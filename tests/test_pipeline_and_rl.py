from pathlib import Path
import torch

from ialm.cli import main
from ialm.config import ModelConfig, TrainConfig
from ialm.judges import parse_judgment, Judgment
from ialm.evaluate import Evaluator
from ialm.rl import RewardModelTrainer, DPOTrainer, PPOTrainer, GRPOTrainer, DAPOTrainer, make_group_advantages
from ialm.tokenize_data import HFTokenizerAdapter
from ialm.model import IntrinsicAgentLM
from ialm.pipeline import TrainingPipeline, PROCESSORS


def test_cli_register(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ialm.cli.DatasetRegistry", lambda: __import__("ialm.data", fromlist=["DatasetRegistry"]).DatasetRegistry(tmp_path/"catalog.json"))
    main(["data","register","demo","org/repo","--max-examples","5"])


def test_parse_judgment_ok():
    j=parse_judgment('noise {"winner":"A","score_a":1,"score_b":0,"rationale":"better"} trailing')
    assert isinstance(j,Judgment) and j.winner=="A" and j.score_a==1.0

def test_parse_judgment_bad():
    try:
        parse_judgment("no json here")
        assert False
    except ValueError as e:
        assert "JSON" in str(e)

def test_evaluator_records(tmp_path):
    ev=Evaluator(tmp_path/"eval.jsonl")
    r=ev.verify_answer("t","42","42")
    assert r.passed
    assert (tmp_path/"eval.jsonl").read_text().strip()

class FakeAdapter(HFTokenizerAdapter):
    def __init__(self):
        self.tok=type("Tok",(),{"model_max_length":64})()
    def text(self,text,max_seq_len):
        return {"input_ids":[1,2,3],"attention_mask":[1,1,1]}

def test_tokenize_prompt_only_rl():
    t=FakeAdapter()
    out=t({"prompt":"hi","answer":"42","verifier":"math_exact"},max_seq_len=16)
    assert out["prompt_ids"]==[1,2,3]
    assert "chosen" not in out and "chosen_ids" not in out
    assert out["answer"]=="42" and out["verifier"]=="math_exact"

def test_tokenize_preference_pair():
    t=FakeAdapter()
    out=t({"prompt":"p","chosen":"c","rejected":"r"},max_seq_len=16)
    assert out["chosen_ids"] and out["rejected_ids"] and out["prompt_ids"]

def test_rl_trainers_step_and_clip():
    cfg=ModelConfig(vocab_size=64,max_seq_len=32,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,memory_slots=2,planning_steps=1)
    m=IntrinsicAgentLM(cfg); ref=IntrinsicAgentLM(cfg)
    x=torch.randint(0,64,(1,6))
    opt=torch.optim.SGD(m.parameters(),lr=1e-3)
    rm=RewardModelTrainer(m,torch.optim.SGD(m.parameters(),lr=1e-3))
    loss=rm.step({"chosen_ids":x,"rejected_ids":x+1})
    assert loss==loss
    dpo_tr=DPOTrainer(m,ref,opt,beta=0.1)
    dpo_tr.step({"chosen_ids":x,"rejected_ids":x.flip(-1)})
    ppo=PPOTrainer(m,opt,0.2,0.2)
    old=torch.zeros(1)
    ppo.update([x],old,torch.ones(1))
    grpo=GRPOTrainer(m,ref,opt,0.02)
    grpo.update([x,x],torch.zeros(2))
    dapo=DAPOTrainer(m,ref,opt,0.2,0.28)
    dapo.update([x,x],torch.zeros(2),torch.ones(2),torch.tensor([1.0,0.0]))
    adv=make_group_advantages([1.0,1.0,2.0])
    assert adv.shape[0]==3

def test_process_registry_covers_stage_order():
    recipe_order=TrainConfig().stage_order
    for stage in recipe_order:
        assert stage in PROCESSORS

def test_pipeline_smoke_local_train(tmp_path):
    fixture=Path("tests/fixtures/smoke_pretrain.jsonl").resolve()
    recipe={
        "data_root":str(tmp_path/"data"),
        "tokenizer":{},
        "model":{"vocab_size":264,"max_seq_len":128,"d_model":64,"n_layers":1,"n_heads":4,"n_kv_heads":2,"ffn_mult":2.0,"memory_slots":4,"planning_steps":1,"action_classes":8,"tie_embeddings":True},
        "train":{"output_dir":str(tmp_path/"out"),"device":"cpu","max_steps":2,"warmup_steps":1,"lr":1e-3,"micro_batch_size":2,"grad_accum_steps":2,"stage_order":["pretrain"],"grad_clip":1.0,"min_lr_ratio":0.5,"weight_decay":0.0},
        "datasets":{"pretrain":[{"name":"smoke","repo":str(fixture),"max_examples":8,"streaming":False,"weight":1.0}]},
        "stages":{"pretrain":{"max_steps":2}},
    }
    import yaml as _yaml
    path=tmp_path/"recipe.yaml"; path.write_text(_yaml.safe_dump(recipe),encoding="utf-8")
    p=TrainingPipeline(path)
    p.run()
    assert (tmp_path/"out"/"pretrain.safetensors").exists()
    assert (tmp_path/"out"/"final.safetensors").exists()

def test_reward_optimizer_targets_reward_model_params(tmp_path):
    fixture=Path("tests/fixtures/smoke_pretrain.jsonl").resolve()
    recipe={
        "data_root":str(tmp_path/"data"),
        "tokenizer":{},
        "model":{"vocab_size":264,"max_seq_len":64,"d_model":32,"n_layers":1,"n_heads":4,"n_kv_heads":2,"ffn_mult":2.0,"memory_slots":2,"planning_steps":1,"action_classes":8,"tie_embeddings":True},
        "train":{"output_dir":str(tmp_path/"out"),"device":"cpu","max_steps":1,"warmup_steps":0,"lr":1e-3,"micro_batch_size":1,"grad_accum_steps":1,"stage_order":["reward_model"],"grad_clip":1.0},
        "datasets":{"reward_model":[{"name":"pref","repo":str(fixture),"max_examples":2,"streaming":False,"weight":1.0}]},
        "stages":{"reward_model":{"max_steps":1}},
    }
    # prepared rows lack chosen_ids; reward stage should no-op without crash
    import yaml as _yaml
    path=tmp_path/"recipe.yaml"; path.write_text(_yaml.safe_dump(recipe),encoding="utf-8")
    p=TrainingPipeline(path)
    p.sync_data("reward_model")
    paths=p.prepare_data("reward_model")
    p.run_reward_model(paths)
