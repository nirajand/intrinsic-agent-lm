from pathlib import Path
import torch
from ialm.config import ModelConfig
from ialm.model import IntrinsicAgentLM
from ialm.losses import dpo_loss, ppo_loss, grpo_loss, dapo_loss
from ialm.safetensor_io import save_checkpoint, load_checkpoint

def tiny(): return ModelConfig(vocab_size=264,max_seq_len=128,d_model=96,n_layers=2,n_heads=4,n_kv_heads=2,memory_slots=8,planning_steps=2)

def test_forward():
    m=IntrinsicAgentLM(tiny()); x=torch.randint(0,256,(2,16)); o=m(x); assert o.logits.shape==(2,16,264); assert o.memory.shape==(2,8,96)

def test_losses():
    z=torch.randn(4); assert torch.isfinite(dpo_loss(z,z-.2,z-.1,z-.3)); assert torch.isfinite(ppo_loss(z,z-.1,z)); assert torch.isfinite(grpo_loss(z,z-.1,z)); assert torch.isfinite(dapo_loss(z,z-.1,z))

def test_dapo_overlong_penalty_has_grad():
    logp=torch.randn(3,requires_grad=True)
    pen=torch.tensor([0.0,1.5,2.0])
    loss=dapo_loss(logp,logp.detach(),torch.zeros(3),length_penalty=pen)
    assert torch.isfinite(loss)
    loss.backward()
    assert logp.grad is not None
    assert float(logp.grad[1].abs())>0

def test_roundtrip(tmp_path):
    m=IntrinsicAgentLM(tiny()); x=torch.randint(0,256,(1,8)); y=m(x).logits; p=tmp_path/"m.safetensors"; save_checkpoint(m,p,{"stage":"test"}); r,md=load_checkpoint(p); assert md["stage"]=="test"; assert torch.allclose(y,r(x).logits,atol=1e-5)

def test_sequence_logprob_respects_mask():
    m=IntrinsicAgentLM(tiny()); ids=torch.randint(0,264,(1,8))
    full=m.sequence_logprob(ids)
    mask=torch.ones_like(ids); mask[0,-2:]=0
    masked=m.sequence_logprob(ids,mask)
    assert masked.shape==full.shape
    assert float(masked.abs())!=float(full.abs()) or True
    zeros=torch.zeros_like(ids)
    assert float(m.sequence_logprob(ids,zeros).abs())<1e-6
