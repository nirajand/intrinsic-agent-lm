from __future__ import annotations
import torch
import torch.nn.functional as F
from .losses import ppo_loss, grpo_loss, dapo_loss
from .verifiers import verify

def _logp(model,seqs,masks=None):
    if masks is None: return torch.stack([model.sequence_logprob(x).mean() for x in seqs])
    return torch.stack([model.sequence_logprob(x,m).mean() for x,m in zip(seqs,masks)])

class RewardModelTrainer:
    def __init__(self,model,optimizer): self.model=model; self.optimizer=optimizer
    def step(self,batch):
        self.model.train(); c=self.model(batch["chosen_ids"]); r=self.model(batch["rejected_ids"]); loss=-F.logsigmoid(c.verifier-r.verifier).mean(); self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); return float(loss.detach())

class DPOTrainer:
    def __init__(self,policy,reference,optimizer,beta=0.1): self.policy=policy; self.reference=reference.eval(); self.optimizer=optimizer; self.beta=beta
    def step(self,batch):
        self.policy.train()
        cm=batch.get("chosen_mask"); rm=batch.get("rejected_mask")
        pc=self.policy.sequence_logprob(batch["chosen_ids"],cm); pr=self.policy.sequence_logprob(batch["rejected_ids"],rm)
        with torch.no_grad():
            rc=self.reference.sequence_logprob(batch["chosen_ids"],cm); rr=self.reference.sequence_logprob(batch["rejected_ids"],rm)
        loss=-F.logsigmoid(self.beta*((pc-rc)-(pr-rr))).mean(); self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); return float(loss.detach())

class PPOTrainer:
    def __init__(self,model,optimizer,clip_low=0.2,clip_high=0.2): self.model=model; self.optimizer=optimizer; self.clip_low=clip_low; self.clip_high=clip_high
    def update(self,seqs,old_logp,advantages,masks=None):
        self.model.train(); new=_logp(self.model,seqs,masks); loss=ppo_loss(new,old_logp,advantages,self.clip_low,self.clip_high); self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); return float(loss.detach())

class GRPOTrainer:
    def __init__(self,model,reference,optimizer,kl_beta=0.02): self.model=model; self.reference=reference.eval(); self.optimizer=optimizer; self.kl_beta=kl_beta
    def update(self,group,advantages):
        self.model.train(); new=_logp(self.model,group)
        with torch.no_grad(): ref=_logp(self.reference,group)
        loss=grpo_loss(new,ref,advantages,self.kl_beta); self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); return float(loss.detach())

class DAPOTrainer:
    def __init__(self,model,reference,optimizer,clip_low=0.2,clip_high=0.28): self.model=model; self.reference=reference.eval(); self.optimizer=optimizer; self.clip_low=clip_low; self.clip_high=clip_high
    def update(self,group,old_logp,advantages,overlong_penalty=None):
        self.model.train(); new=_logp(self.model,group); loss=dapo_loss(new,old_logp,advantages,self.clip_low,self.clip_high,overlong_penalty); self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); return float(loss.detach())

def make_group_advantages(rewards):
    r=torch.as_tensor(rewards,dtype=torch.float32); return (r-r.mean())/r.std(unbiased=False).clamp_min(1e-6)

def math_reward_fn(prompt,generated,target,verifier="math_exact"):
    return verify(verifier,generated,target).reward
