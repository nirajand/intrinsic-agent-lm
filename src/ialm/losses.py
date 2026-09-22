from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import IntrinsicAgentLM

def masked_mean(x,mask=None):
    if mask is None:return x.mean()
    m=mask.to(x); return (x*m).sum()/m.sum().clamp_min(1)

def lm_loss(logits,labels): return F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1),ignore_index=-100)

def dpo_loss(pi_c,pi_r,ref_c,ref_r,beta=0.1): return -F.logsigmoid(beta*((pi_c-ref_c)-(pi_r-ref_r))).mean()

def reward_margin_loss(chosen,rejected): return -F.logsigmoid(chosen-rejected).mean()

def ppo_loss(logp,old_logp,adv,clip_low=0.2,clip_high=0.2):
    r=torch.exp((logp-old_logp).clamp(-20,20)); a=adv.detach(); unclipped=r*a; clipped=torch.clamp(r,1-clip_low,1+clip_high)*a; return -torch.minimum(unclipped,clipped).mean()

def grpo_loss(logp,ref_logp,advantages,kl_beta=0.02):
    ratio=torch.exp((logp.detach()*0 + logp-ref_logp).clamp(-20,20)); kl=(logp-ref_logp).mean(); return -(advantages.detach()*logp).mean()+kl_beta*kl

def dapo_loss(logp,old_logp,advantages,clip_low=0.2,clip_high=0.28,length_penalty=None):
    ratio=torch.exp((logp-old_logp).clamp(-20,20)); a=advantages.detach(); unclipped=ratio*a; clipped=torch.clamp(ratio,1-clip_low,1+clip_high)*a; loss=-torch.minimum(unclipped,clipped).mean()
    if length_penalty is not None: loss=loss+length_penalty
    return loss
