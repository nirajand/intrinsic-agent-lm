from __future__ import annotations
from pathlib import Path
import copy, json, random, yaml
import torch
from torch.utils.data import DataLoader, Dataset
from .config import ModelConfig, TrainConfig, RecipeConfig
from .model import IntrinsicAgentLM
from .data import DatasetManager, DatasetSpec, DatasetRegistry
from .processors import pretrain,sft,dpo,reward,rlvr,tool_trace
from .tokenize_data import HFTokenizerAdapter
from .losses import lm_loss
from .rl import RewardModelTrainer,DPOTrainer,PPOTrainer,GRPOTrainer,DAPOTrainer,make_group_advantages
from .safetensor_io import save_checkpoint,load_checkpoint

PROCESSORS={"pretrain":pretrain,"sft":sft,"dpo":dpo,"reward_model":reward,"rlaif":dpo,"rlhf":rlvr,"rlvr":rlvr,"grpo":rlvr,"ppo":rlvr,"dapo":rlvr,"tool_sft":tool_trace}

class JsonlDataset(Dataset):
    def __init__(self,path): self.rows=[json.loads(x) for x in Path(path).open(encoding="utf-8")]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def pad_batch(rows,key,pad=0):
    xs=[torch.tensor(r[key],dtype=torch.long) for r in rows]; m=max(len(x) for x in xs); out=torch.full((len(xs),m),pad,dtype=torch.long); mask=torch.zeros_like(out)
    for i,x in enumerate(xs): out[i,:len(x)]=x; mask[i,:len(x)]=1
    return out,mask

class TrainingPipeline:
    def __init__(self,recipe_path):
        raw=yaml.safe_load(Path(recipe_path).read_text()); self.raw=raw
        self.model_cfg=ModelConfig(**raw["model"]); self.train_cfg=TrainConfig(**raw["train"]); self.datasets={k:[DatasetSpec(**d) for d in v] for k,v in raw.get("datasets",{}).items()}; self.stages=raw.get("stages",{})
        self.manager=DatasetManager(raw.get("data_root","data")); self.tokenizer=None
        tok=raw.get("tokenizer",{}).get("id")
        if tok:self.tokenizer=HFTokenizerAdapter(tok,raw.get("tokenizer",{}).get("revision"))
        self.device = self.train_cfg.device if torch.cuda.is_available() or self.train_cfg.device=="cpu" else "cpu"
        self.model=IntrinsicAgentLM(self.model_cfg).to(self.device)
        self.reward_model=None
    def sync_data(self,stage=None):
        for s in (stage,) if stage else self.datasets:
            for spec in self.datasets.get(s,[]):
                if spec.streaming:
                    # Streaming is intentionally not materialized; prepare() can iterate the Hub stream.
                    self.manager._log({"event":"stream_ready","name":spec.name,"repo":spec.repo,"stage":s})
                else:
                    self.manager.download(spec)
    def prepare_data(self,stage):
        proc=PROCESSORS.get(stage); 
        if not proc: raise KeyError(stage)
        return [self.manager.prepare(spec,stage,proc,self.tokenizer,self.model_cfg.max_seq_len) for spec in self.datasets.get(stage,[])]
    def _optim(self,lr=None): return torch.optim.AdamW(self.model.parameters(),lr=lr or self.train_cfg.lr,weight_decay=self.train_cfg.weight_decay)
    def _rows(self,paths):
        for p in paths:
            yield from JsonlDataset(p).rows

    @staticmethod
    def _batch(rows,key,pad=0):
        xs=[torch.tensor(r[key],dtype=torch.long) for r in rows]; m=max(len(x) for x in xs); ids=torch.full((len(xs),m),pad,dtype=torch.long); labels=torch.full((len(xs),m),-100,dtype=torch.long)
        for i,x in enumerate(xs): ids[i,:len(x)]=x; labels[i,:len(x)]=x
        return ids,labels

    def _run_lm_stage(self,paths,stage,lr_scale=1.0):
        if not paths:return
        ds=list(self._rows(paths)); opt=self._optim(self.train_cfg.lr*lr_scale); bs=max(1,self.train_cfg.micro_batch_size); steps=0
        self.model.train()
        for i in range(0,len(ds),bs):
            rows=ds[i:i+bs]; ids,labels=self._batch(rows,"input_ids"); ids=ids.to(self.device); labels=labels.to(self.device); loss=lm_loss(self.model(ids).logits,labels); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_cfg.grad_clip); opt.step(); steps+=1
            if steps>=self.stages.get(stage,{}).get("max_steps",self.train_cfg.max_steps):break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/(stage+".safetensors"),{"stage":stage,"steps":steps})

    def run_pretrain(self,paths): self._run_lm_stage(paths,"pretrain",1.0)
    def run_sft(self,paths): self._run_lm_stage(paths,"sft",0.5)

    def _pairs(self,paths):
        for p in paths:
            for row in JsonlDataset(p): yield row
    def run_reward_model(self,paths):
        if not paths:return
        import copy
        self.reward_model=copy.deepcopy(self.model).to(self.device); opt=self._optim(self.train_cfg.lr*0.25)
        tr=RewardModelTrainer(self.reward_model,opt); steps=0
        for row in self._rows(paths):
            c=torch.tensor(row["chosen_ids"],device=self.device).unsqueeze(0); r=torch.tensor(row["rejected_ids"],device=self.device).unsqueeze(0); tr.step({"chosen_ids":c,"rejected_ids":r}); steps+=1
            if steps>=self.stages.get("reward_model",{}).get("max_steps",1000):break
        save_checkpoint(self.reward_model,Path(self.train_cfg.output_dir)/"reward_model.safetensors",{"stage":"reward_model","steps":steps})

    def run_dpo(self,paths):
        if not paths:return
        ref=copy.deepcopy(self.model).eval().to(self.device); opt=self._optim(self.train_cfg.lr*0.1); tr=DPOTrainer(self.model,ref,opt,self.train_cfg.alignment_beta); rows=list(self._rows(paths)); bs=max(1,self.train_cfg.micro_batch_size); steps=0
        for i in range(0,len(rows),bs):
            chunk=rows[i:i+bs]; cp,_=self._batch(chunk,"chosen_ids"); rp,_=self._batch(chunk,"rejected_ids"); cp=cp.to(self.device); rp=rp.to(self.device); tr.step({"chosen_ids":cp,"rejected_ids":rp}); steps+=1
            if steps>=self.stages.get("dpo",{}).get("max_steps",1000):break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/"dpo.safetensors",{"stage":"dpo","steps":steps})

    def _rl_rows(self,paths):
        yield from self._rows(paths)

    def _sample_group(self,prompt_ids,group_size,max_new_tokens):
        return [self.model.generate(prompt_ids.clone(),max_new_tokens=max_new_tokens)[0] for _ in range(group_size)]

    def _reward_generated(self,row,seq,prompt_len):
        from .verifiers import verify
        text=self.tokenizer.decode(seq[0,prompt_len:].tolist()) if self.tokenizer is not None else ""
        if row.get("reward") is not None:return float(row["reward"])
        return verify(row.get("verifier","math_exact"),text,row.get("answer","")).reward

    def run_rlhf(self,paths):
        if not paths or self.reward_model is None:return
        opt=self._optim(self.train_cfg.lr*0.03); tr=PPOTrainer(self.model,opt,self.train_cfg.ppo_clip_low,self.train_cfg.ppo_clip_high); steps=0
        self.reward_model.eval()
        for row in self._rl_rows(paths):
            p=torch.tensor(row["prompt_ids"],device=self.device).unsqueeze(0); seq,_=self.model.generate(p.clone(),max_new_tokens=self.stages.get("rlhf",{}).get("max_new_tokens",256));
            with torch.no_grad(): old=self.model.sequence_logprob(seq).mean().unsqueeze(0); rm_reward=self.reward_model(seq).verifier.sigmoid().mean().unsqueeze(0)
            tr.update([seq],old,rm_reward*self.train_cfg.reward_scale); steps+=1
            if steps>=self.stages.get("rlhf",{}).get("max_steps",1000):break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/"rlhf.safetensors",{"stage":"rlhf","steps":steps})

    def run_grpo(self,paths,dapo=False):
        if not paths:return
        ref=copy.deepcopy(self.model).eval().to(self.device); opt=self._optim(self.train_cfg.lr*0.05)
        tr=(DAPOTrainer(self.model,ref,opt,self.train_cfg.dapo_clip_low,self.train_cfg.dapo_clip_high) if dapo else GRPOTrainer(self.model,ref,opt,self.train_cfg.grpo_kl_beta))
        steps=0; stage_name="dapo" if dapo else "grpo"; max_new=self.stages.get(stage_name,{}).get("max_new_tokens",128)
        for row in self._rl_rows(paths):
            p=torch.tensor(row["prompt_ids"],device=self.device).unsqueeze(0); attempts=0
            while True:
                group=self._sample_group(p,self.train_cfg.grpo_group_size,max_new); rewards=[self._reward_generated(row,seq,p.size(1)) for seq in group]
                if not dapo or len(set(round(x,6) for x in rewards))>1 or attempts>=self.train_cfg.dapo_max_resamples: break
                attempts+=1
            adv=make_group_advantages(rewards).to(self.device)
            with torch.no_grad(): old=torch.stack([self.model.sequence_logprob(x).mean() for x in group])
            if dapo:
                length_penalty=torch.tensor(self.train_cfg.overlong_penalty*(max(0,max_new-self.model_cfg.max_seq_len)),device=self.device)
                tr.update(group,old,adv,length_penalty)
            else:
                tr.update(group,adv)
            steps+=1
            if steps>=self.stages.get(stage_name,{}).get("max_steps",1000):break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/("dapo.safetensors" if dapo else "grpo.safetensors"),{"stage":stage_name,"steps":steps})

    def run_ppo(self,paths):
        if not paths:return
        opt=self._optim(self.train_cfg.lr*0.03); tr=PPOTrainer(self.model,opt,self.train_cfg.ppo_clip_low,self.train_cfg.ppo_clip_high); steps=0
        for row in self._rl_rows(paths):
            p=torch.tensor(row["prompt_ids"],device=self.device).unsqueeze(0); seq,_=self.model.generate(p.clone(),max_new_tokens=self.stages.get("ppo",{}).get("max_new_tokens",128))
            with torch.no_grad(): old=self.model.sequence_logprob(seq).mean().unsqueeze(0)
            reward=self._reward_generated(row,seq,p.size(1)); adv=torch.tensor([reward*self.train_cfg.reward_scale],device=self.device); tr.update([seq],old,adv); steps+=1
            if steps>=self.stages.get("ppo",{}).get("max_steps",1000):break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/"ppo.safetensors",{"stage":"ppo","steps":steps})

    def run(self):
        Path(self.train_cfg.output_dir).mkdir(parents=True,exist_ok=True); random.seed(self.train_cfg.seed); torch.manual_seed(self.train_cfg.seed)
        for stage in self.train_cfg.stage_order:
            if stage not in self.datasets: continue
            self.sync_data(stage); paths=self.prepare_data(stage)
            if stage=="pretrain": self.run_pretrain(paths)
            elif stage=="sft": self.run_sft(paths)
            elif stage=="reward_model": self.run_reward_model(paths)
            elif stage in {"dpo","rlaif"}: self.run_dpo(paths)
            elif stage=="grpo": self.run_grpo(paths,False)
            elif stage=="dapo": self.run_grpo(paths,True)
            elif stage=="ppo": self.run_ppo(paths)
            elif stage=="rlhf": self.run_rlhf(paths)
            elif stage=="rlvr": self.run_grpo(paths,False)
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/"final.safetensors",{"stage":"final","pipeline":"automatic"})
