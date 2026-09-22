import torch
from ialm.config import ModelConfig
from ialm.model import IntrinsicAgentLM
from ialm.self_align import generate_self_preference,self_dpo_loss

def test_self_align_shapes():
    m=IntrinsicAgentLM(ModelConfig(vocab_size=264,max_seq_len=64,d_model=64,n_layers=1,n_heads=4,n_kv_heads=2,memory_slots=4,planning_steps=1))
    p=torch.randint(0,256,(2,5)); c,r,s=generate_self_preference(m,p,samples=2,max_new_tokens=2); assert c.shape==r.shape and s.shape[1]==2
    assert torch.isfinite(self_dpo_loss(m,c,r))
