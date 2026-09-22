from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.weight=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

def rotate_half(x):
    a,b=x.chunk(2,dim=-1); return torch.cat((-b,a),dim=-1)

def rope_cos_sin(seq_len, head_dim, theta, device, dtype):
    inv=1.0/(theta**(torch.arange(0,head_dim,2,device=device,dtype=torch.float32)/head_dim))
    t=torch.arange(seq_len,device=device,dtype=torch.float32)
    f=torch.outer(t,inv); emb=torch.cat((f,f),dim=-1)
    return emb.cos()[None,None].to(dtype), emb.sin()[None,None].to(dtype)

class GQA(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.d_model; h=cfg.n_heads; kv=cfg.n_kv_heads; hd=d//h
        self.h,self.kv,self.hd=h,kv,hd; self.theta=cfg.rope_theta; self.use_qk_norm=cfg.use_qk_norm
        self.q=nn.Linear(d,h*hd,bias=False); self.k=nn.Linear(d,kv*hd,bias=False); self.v=nn.Linear(d,kv*hd,bias=False); self.o=nn.Linear(h*hd,d,bias=False)
        self.qn=RMSNorm(hd) if cfg.use_qk_norm else nn.Identity(); self.kn=RMSNorm(hd) if cfg.use_qk_norm else nn.Identity()
    def forward(self,x):
        b,t,_=x.shape; q=self.q(x).view(b,t,self.h,self.hd).transpose(1,2); k=self.k(x).view(b,t,self.kv,self.hd).transpose(1,2); v=self.v(x).view(b,t,self.kv,self.hd).transpose(1,2)
        q=self.qn(q); k=self.kn(k); c,s=rope_cos_sin(t,self.hd,self.theta,x.device,q.dtype); q=q*c+rotate_half(q)*s; k=k*c+rotate_half(k)*s
        rep=self.h//self.kv
        if rep>1: k=k.repeat_interleave(rep,dim=1); v=v.repeat_interleave(rep,dim=1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).contiguous().view(b,t,-1))

class SwiGLU(nn.Module):
    def __init__(self,d,mult): super().__init__(); h=int(d*mult); self.g=nn.Linear(d,h,bias=False); self.u=nn.Linear(d,h,bias=False); self.o=nn.Linear(h,d,bias=False)
    def forward(self,x): return self.o(F.silu(self.g(x))*self.u(x))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n1=RMSNorm(cfg.d_model); self.a=GQA(cfg); self.n2=RMSNorm(cfg.d_model); self.m=SwiGLU(cfg.d_model,cfg.ffn_mult)
    def forward(self,x): return x+self.a(self.n1(x)) + self.m(self.n2(x))

class LatentMemory(nn.Module):
    def __init__(self,d,slots):
        super().__init__(); self.slots=slots; self.initial=nn.Parameter(torch.randn(1,slots,d)*0.02); self.r=nn.Linear(d,d); self.w=nn.Linear(d,d); self.g=nn.Linear(2*d,d); self.n=RMSNorm(d)
    def init(self,b,device,dtype): return self.initial.to(device=device,dtype=dtype).expand(b,-1,-1).clone()
    def read(self,m,c):
        q=self.r(c).unsqueeze(1); a=(q@m.transpose(-1,-2))/math.sqrt(m.size(-1)); return torch.bmm(a.softmax(-1),m).squeeze(1)
    def write(self,m,c):
        cc=c.unsqueeze(1).expand_as(m); g=torch.sigmoid(self.g(torch.cat([m,cc],-1))); return self.n((1-g)*m+g*self.w(cc))

class Planner(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.d_model; self.steps=cfg.planning_steps; self.cell=nn.GRUCell(d,d); self.read=nn.Linear(d,d); self.score=nn.Linear(d,1); self.state=nn.Parameter(torch.randn(1,d)*0.02)
    def forward(self,context):
        s=self.state.to(context).expand(context.size(0),-1); traces=[]
        for _ in range(self.steps):
            s=self.cell(context+s,s); traces.append(s)
        tr=torch.stack(traces,1); weights=self.score(tr).squeeze(-1).softmax(-1); return (tr*weights.unsqueeze(-1)).sum(1), tr

@dataclass
class ModelOutput:
    logits: torch.Tensor
    action_logits: torch.Tensor
    risk: torch.Tensor
    verifier: torch.Tensor
    confidence: torch.Tensor
    termination: torch.Tensor
    value: torch.Tensor
    memory: torch.Tensor
    plan: torch.Tensor
    hidden: torch.Tensor

class AgentCore(nn.Module):
    ACTIONS=("respond","tool","retrieve","remember","delegate","verify","continue","finish")
    def __init__(self,cfg):
        super().__init__(); d=cfg.d_model; self.memory=LatentMemory(d,cfg.memory_slots); self.planner=Planner(cfg); self.fuse=nn.Linear(2*d,d); self.action=nn.Linear(d,cfg.action_classes); self.risk=nn.Linear(d,1); self.verifier=nn.Linear(d,1); self.confidence=nn.Linear(d,1); self.term=nn.Linear(d,1); self.value=nn.Linear(d,1)
    def forward(self,x,memory_state=None):
        b=x.size(0); m=self.memory.init(b,x.device,x.dtype) if memory_state is None else memory_state; ctx=x.mean(1); mr=self.memory.read(m,ctx); p,tr=self.planner(ctx+mr); h=x+self.fuse(torch.cat([x,p.unsqueeze(1).expand_as(x)],-1)); mn=self.memory.write(m,ctx+mr+p); pooled=h[:,-1]
        return h,self.action(h),self.risk(h).squeeze(-1),self.verifier(pooled).squeeze(-1),self.confidence(pooled).squeeze(-1),self.term(h).squeeze(-1),self.value(pooled).squeeze(-1),mn,p

class IntrinsicAgentLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); cfg.validate(); self.cfg=cfg
        self.tok=nn.Embedding(cfg.vocab_size,cfg.d_model); self.role=nn.Embedding(cfg.role_classes,cfg.d_model); self.drop=nn.Dropout(cfg.dropout); self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm=RMSNorm(cfg.d_model); self.agent=AgentCore(cfg); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self._init();
        if cfg.tie_embeddings: self.lm_head.weight=self.tok.weight
    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02); 
            elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self,input_ids,role_ids=None,memory_state=None):
        x=self.tok(input_ids); x=x+(self.role(role_ids) if role_ids is not None else 0); x=self.drop(x)
        for b in self.blocks: x=b(x)
        x=self.norm(x); h,a,r,v,c,t,val,m,p=self.agent(x,memory_state); return ModelOutput(self.lm_head(h),a,r,v,c,t,val,m,p,h)
    def sequence_logprob(self,ids,mask=None):
        o=self(ids); lp=F.log_softmax(o.logits[:,:-1],-1); tok=ids[:,1:].unsqueeze(-1); vals=lp.gather(-1,tok).squeeze(-1)
        if mask is not None: vals=vals*mask[:,1:].to(vals)
        return vals.sum(-1)
    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens=128,temperature=1.0,top_p=0.95,role_ids=None,memory_state=None):
        ids=input_ids; out=None
        for _ in range(max_new_tokens):
            out=self(ids,role_ids=role_ids,memory_state=memory_state); memory_state=out.memory; logits=out.logits[:,-1]/max(temperature,1e-4); p=logits.softmax(-1)
            if top_p<1:
                sp,si=torch.sort(p,descending=True); cp=sp.cumsum(-1); mask=cp>top_p; mask[...,1:]=mask[...,:-1]; mask[...,0]=False; sp=sp.masked_fill(mask,0); sp=sp/sp.sum(-1,keepdim=True); nxt=si.gather(-1,torch.multinomial(sp,1))
            else: nxt=torch.multinomial(p,1)
            ids=torch.cat([ids,nxt],1)
        return ids,out
