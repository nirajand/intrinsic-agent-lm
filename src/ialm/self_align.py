from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import IntrinsicAgentLM  # noqa: F401 — re-export type for callers

@torch.no_grad()
def generate_self_preference(model:IntrinsicAgentLM,prompts:torch.Tensor,samples:int=2,max_new_tokens:int=128):
    candidates=[]; scores=[]
    for _ in range(samples):
        seq,out=model.generate(prompts.clone(),max_new_tokens=max_new_tokens)
        score=out.verifier.sigmoid()-out.risk.sigmoid().mean(-1)
        candidates.append(seq); scores.append(score)
    idx=torch.stack(scores)
    hi=idx.argmax(0); lo=idx.argmin(0)
    chosen=torch.stack([candidates[hi[i]][i] for i in range(prompts.size(0))])
    rejected=torch.stack([candidates[lo[i]][i] for i in range(prompts.size(0))])
    return chosen,rejected,idx

def self_dpo_loss(model,chosen,rejected,beta=0.1):
    pc=model.sequence_logprob(chosen); pr=model.sequence_logprob(rejected)
    return -F.logsigmoid(beta*(pc-pr)).mean()
