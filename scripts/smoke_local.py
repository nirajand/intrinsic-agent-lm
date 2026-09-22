from pathlib import Path
import sys, torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from ialm.config import ModelConfig
from ialm.model import IntrinsicAgentLM
from ialm.safetensor_io import save_checkpoint,load_checkpoint
cfg=ModelConfig(vocab_size=264,max_seq_len=128,d_model=128,n_layers=2,n_heads=4,n_kv_heads=2, memory_slots=8,planning_steps=2)
m=IntrinsicAgentLM(cfg); x=torch.randint(0,256,(2,16)); o=m(x); print(o.logits.shape,o.action_logits.shape,o.memory.shape); save_checkpoint(m,"runs/smoke/model.safetensors",{"test":"true"}); m2,md=load_checkpoint("runs/smoke/model.safetensors"); print(md); print(torch.allclose(m2(x).logits,m(x).logits,atol=1e-5))
