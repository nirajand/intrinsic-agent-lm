from __future__ import annotations
import json, re
from dataclasses import dataclass

@dataclass
class VerificationResult:
    reward: float
    passed: bool
    details: str

def numeric_answer(text):
    nums=re.findall(r"[-+]?\d+(?:\.\d+)?",text.replace(",","")); return float(nums[-1]) if nums else None

def math_exact(pred,target,tol=1e-8):
    p=numeric_answer(pred); t=numeric_answer(target)
    if p is None or t is None:return VerificationResult(0.0,False,"numeric answer missing")
    ok=abs(p-t)<=tol*max(1,abs(t)); return VerificationResult(1.0 if ok else 0.0,ok,f"pred={p}, target={t}")

def regex_exact(pred,target):
    ok=pred.strip()==target.strip(); return VerificationResult(1.0 if ok else 0.0,ok,"exact string")

def json_valid(pred,target=None):
    try: json.loads(pred); return VerificationResult(1.0,True,"valid JSON")
    except Exception as e:return VerificationResult(0.0,False,str(e))

REGISTRY={"math_exact":math_exact,"regex_exact":regex_exact,"json_valid":json_valid}

def verify(name,pred,target):
    if name not in REGISTRY: raise KeyError(f"unknown verifier: {name}")
    return REGISTRY[name](pred,target)
