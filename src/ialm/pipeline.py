from __future__ import annotations
from pathlib import Path
import copy, json, random, yaml
import torch
from torch.utils.data import Dataset
from .config import ModelConfig, TrainConfig
from .model import IntrinsicAgentLM
from .data import DatasetManager, DatasetSpec
from .processors import pretrain,sft,dpo,reward,rlvr,tool_trace
from .tokenize_data import HFTokenizerAdapter
from .losses import lm_loss
from .rl import RewardModelTrainer,DPOTrainer,PPOTrainer,GRPOTrainer,DAPOTrainer,make_group_advantages
from .safetensor_io import save_checkpoint

PROCESSORS={"pretrain":pretrain,"sft":sft,"dpo":dpo,"reward_model":reward,"rlaif":dpo,"rlhf":rlvr,"rlvr":rlvr,"grpo":rlvr,"ppo":rlvr,"dapo":rlvr,"tool_sft":tool_trace}

class JsonlDataset(Dataset):
    def __init__(self,path): self.rows=[json.loads(x) for x in Path(path).open(encoding="utf-8")]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

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
                self.manager.download(spec)
    def prepare_data(self,stage):
        proc=PROCESSORS.get(stage)
        if not proc: raise KeyError(stage)
        return [self.manager.prepare(spec,stage,proc,self.tokenizer,self.model_cfg.max_seq_len) for spec in self.datasets.get(stage,[])]
    def _optim(self,params,lr=None): return torch.optim.AdamW(params,lr=lr or self.train_cfg.lr,weight_decay=self.train_cfg.weight_decay)
    def _lr_at(self,step):
        base=self.train_cfg.lr; warm=self.train_cfg.warmup_steps; total=max(1,self.train_cfg.max_steps)
        if warm and step<warm: return base*(step+1)/warm
        if total<=warm: return base
        t=min(1.0,max(0.0,(step-warm)/max(1,total-warm)))
        return base*(1-t)+base*self.train_cfg.min_lr_ratio*t
    def _rows(self,paths):
        for p in paths:
            yield from JsonlDataset(p).rows

    @staticmethod
    def _batch(rows,key,pad=0):
        xs=[torch.tensor(r[key],dtype=torch.long) for r in rows]; m=max(len(x) for x in xs)
        ids=torch.full((len(xs),m),pad,dtype=torch.long); labels=torch.full((len(xs),m),-100,dtype=torch.long); mask=torch.zeros((len(xs),m),dtype=torch.long)
        for i,x in enumerate(xs): ids[i,:len(x)]=x; labels[i,:len(x)]=x; mask[i,:len(x)]=1
        return ids,labels,mask

    @staticmethod
    def _ensure_ids(row,max_seq_len):
        if "input_ids" in row: return row
        text=row.get("text") or row.get("prompt") or ""
        ids=[1+(ord(c)%262) for c in str(text)[:max_seq_len]]
        if not ids: ids=[1]
        out=dict(row); out["input_ids"]=ids; out["attention_mask"]=[1]*len(ids)
        if "text" not in out: out["text"]=str(text)
        return out

    def _run_lm_stage(self,paths,stage,lr_scale=1.0):
        if not paths:return
        ds=[self._ensure_ids(r,self.model_cfg.max_seq_len) for r in self._rows(paths)]
        if not ds:return
        opt=self._optim(self.model.parameters(),self.train_cfg.lr*lr_scale)
        bs=max(1,self.train_cfg.micro_batch_size); accum=max(1,self.train_cfg.grad_accum_steps); steps=0
        self.model.train(); opt.zero_grad(); micro=0
        for i in range(0,len(ds),bs):
            rows=ds[i:i+bs]; ids,labels,_=self._batch(rows,"input_ids"); ids=ids.to(self.device); labels=labels.to(self.device)
            loss=lm_loss(self.model(ids).logits,labels)/accum; loss.backward(); micro+=1
            if micro>=accum or i+bs>=len(ds):
                for g in opt.param_groups: g["lr"]=self._lr_at(steps)*lr_scale
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_cfg.grad_clip); opt.step(); opt.zero_grad(); micro=0; steps+=1
                if steps>=self.stages.get(stage,{}).get("max_steps",self.train_cfg.max_steps):break
            if steps>=self.stages.get(stage,{}).get("max_steps",self.train_cfg.max_steps):break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/(stage+".safetensors"),{"stage":stage,"steps":steps})

    def run_pretrain(self,paths): self._run_lm_stage(paths,"pretrain",1.0)
    def run_sft(self,paths): self._run_lm_stage(paths,"sft",0.5)

    def run_reward_model(self,paths):
        if not paths:return
        self.reward_model=copy.deepcopy(self.model).to(self.device)
        opt=self._optim(self.reward_model.parameters(),self.train_cfg.lr*0.25)
        tr=RewardModelTrainer(self.reward_model,opt); steps=0
        max_steps=self.stages.get("reward_model",{}).get("max_steps",1000)
        for row in self._rows(paths):
            if "chosen_ids" not in row or "rejected_ids" not in row: continue
            c=torch.tensor(row["chosen_ids"],device=self.device).unsqueeze(0); r=torch.tensor(row["rejected_ids"],device=self.device).unsqueeze(0)
            tr.step({"chosen_ids":c,"rejected_ids":r}); steps+=1
            if steps>=max_steps:break
        save_checkpoint(self.reward_model,Path(self.train_cfg.output_dir)/"reward_model.safetensors",{"stage":"reward_model","steps":steps})

    def run_dpo(self,paths,stage="dpo"):
        if not paths:return
        ref=copy.deepcopy(self.model).eval().to(self.device); opt=self._optim(self.model.parameters(),self.train_cfg.lr*0.1)
        tr=DPOTrainer(self.model,ref,opt,self.train_cfg.alignment_beta)
        rows=[r for r in self._rows(paths) if "chosen_ids" in r and "rejected_ids" in r]
        bs=max(1,self.train_cfg.micro_batch_size); steps=0; max_steps=self.stages.get(stage,{}).get("max_steps",1000)
        for i in range(0,len(rows),bs):
            chunk=rows[i:i+bs]
            cp,_,cm=self._batch(chunk,"chosen_ids"); rp,_,rm=self._batch(chunk,"rejected_ids")
            tr.step({"chosen_ids":cp.to(self.device),"rejected_ids":rp.to(self.device),"chosen_mask":cm.to(self.device),"rejected_mask":rm.to(self.device)})
            steps+=1
            if steps>=max_steps:break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/(stage+".safetensors"),{"stage":stage,"steps":steps})

    def _sample_group(self,prompt_ids,group_size,max_new_tokens):
        return [self.model.generate(prompt_ids.clone(),max_new_tokens=max_new_tokens)[0] for _ in range(group_size)]

    def _reward_generated(self,row,seq,prompt_len):
        from .verifiers import verify
        text=self.tokenizer.decode(seq[0,prompt_len:].tolist()) if self.tokenizer is not None else ""
        if row.get("reward") is not None:return float(row["reward"])
        return verify(row.get("verifier","math_exact"),text,row.get("answer","")).reward

    def run_rlhf(self,paths,stage="rlhf"):
        if not paths or self.reward_model is None:return
        opt=self._optim(self.model.parameters(),self.train_cfg.lr*0.03)
        tr=PPOTrainer(self.model,opt,self.train_cfg.ppo_clip_low,self.train_cfg.ppo_clip_high)
        steps=0; max_steps=self.stages.get(stage,{}).get("max_steps",1000); epochs=max(1,self.train_cfg.ppo_epochs)
        self.reward_model.eval()
        for row in self._rl_rows(paths):
            if "prompt_ids" not in row: continue
            p=torch.tensor(row["prompt_ids"],device=self.device).unsqueeze(0); seq,_=self.model.generate(p.clone(),max_new_tokens=self.stages.get(stage,{}).get("max_new_tokens",256))
            with torch.no_grad(): old=self.model.sequence_logprob(seq).mean().unsqueeze(0); rm_reward=self.reward_model(seq).verifier.sigmoid().mean().unsqueeze(0)
            adv=rm_reward*self.train_cfg.reward_scale
            for _ in range(epochs): tr.update([seq],old,adv)
            steps+=1
            if steps>=max_steps:break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/(stage+".safetensors"),{"stage":stage,"steps":steps})

    def run_grpo(self,paths,dapo=False,stage=None):
        if not paths:return
        stage_name=stage or ("dapo" if dapo else "grpo")
        ref=copy.deepcopy(self.model).eval().to(self.device); opt=self._optim(self.model.parameters(),self.train_cfg.lr*0.05)
        tr=(DAPOTrainer(self.model,ref,opt,self.train_cfg.dapo_clip_low,self.train_cfg.dapo_clip_high) if dapo else GRPOTrainer(self.model,ref,opt,self.train_cfg.grpo_kl_beta))
        steps=0; max_new=self.stages.get(stage_name,{}).get("max_new_tokens",128); max_steps=self.stages.get(stage_name,{}).get("max_steps",1000); epochs=max(1,self.train_cfg.ppo_epochs)
        for row in self._rl_rows(paths):
            if "prompt_ids" not in row: continue
            p=torch.tensor(row["prompt_ids"],device=self.device).unsqueeze(0); attempts=0
            while True:
                group=self._sample_group(p,self.train_cfg.grpo_group_size,max_new); rewards=[self._reward_generated(row,seq,p.size(1)) for seq in group]
                if not dapo or len(set(round(x,6) for x in rewards))>1 or attempts>=self.train_cfg.dapo_max_resamples: break
                attempts+=1
            adv=make_group_advantages(rewards).to(self.device)
            if dapo:
                with torch.no_grad(): old=torch.stack([self.model.sequence_logprob(x).mean() for x in group])
                gen_lens=[int(x.size(1)-p.size(1)) for x in group]
                pen=torch.tensor([self.train_cfg.overlong_penalty*max(0,gl-max_new) for gl in gen_lens],device=self.device,dtype=torch.float32)
                for _ in range(epochs): tr.update(group,old,adv,pen)
            else:
                for _ in range(epochs): tr.update(group,adv)
            steps+=1
            if steps>=max_steps:break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/(stage_name+".safetensors"),{"stage":stage_name,"steps":steps})

    def run_ppo(self,paths):
        if not paths:return
        opt=self._optim(self.model.parameters(),self.train_cfg.lr*0.03)
        tr=PPOTrainer(self.model,opt,self.train_cfg.ppo_clip_low,self.train_cfg.ppo_clip_high); steps=0
        max_steps=self.stages.get("ppo",{}).get("max_steps",1000); epochs=max(1,self.train_cfg.ppo_epochs)
        for row in self._rl_rows(paths):
            if "prompt_ids" not in row: continue
            p=torch.tensor(row["prompt_ids"],device=self.device).unsqueeze(0); seq,_=self.model.generate(p.clone(),max_new_tokens=self.stages.get("ppo",{}).get("max_new_tokens",128))
            with torch.no_grad(): old=self.model.sequence_logprob(seq).mean().unsqueeze(0)
            reward=self._reward_generated(row,seq,p.size(1)); adv=torch.tensor([reward*self.train_cfg.reward_scale],device=self.device)
            for _ in range(epochs): tr.update([seq],old,adv)
            steps+=1
            if steps>=max_steps:break
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/"ppo.safetensors",{"stage":"ppo","steps":steps})

    def _rl_rows(self,paths):
        yield from self._rows(paths)

    def run(self):
        Path(self.train_cfg.output_dir).mkdir(parents=True,exist_ok=True); random.seed(self.train_cfg.seed); torch.manual_seed(self.train_cfg.seed)
        for stage in self.train_cfg.stage_order:
            if stage not in self.datasets: continue
            self.sync_data(stage); paths=self.prepare_data(stage)
            if stage=="pretrain": self.run_pretrain(paths)
            elif stage in {"sft","tool_sft"}: self.run_sft(paths)
            elif stage=="reward_model": self.run_reward_model(paths)
            elif stage=="dpo": self.run_dpo(paths,"dpo")
            elif stage=="rlaif": self.run_dpo(paths,"rlaif")
            elif stage=="grpo": self.run_grpo(paths,False,"grpo")
            elif stage=="dapo": self.run_grpo(paths,True,"dapo")
            elif stage=="ppo": self.run_ppo(paths)
            elif stage=="rlhf": self.run_rlhf(paths,"rlhf")
            elif stage=="rlvr": self.run_grpo(paths,False,"rlvr")
            else: raise KeyError(f"unknown stage: {stage}")
        save_checkpoint(self.model,Path(self.train_cfg.output_dir)/"final.safetensors",{"stage":"final","pipeline":"automatic"})
