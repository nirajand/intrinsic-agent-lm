from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import json,time
import torch
from .verifiers import verify

@dataclass
class EvalRecord:
    task: str
    reward: float
    passed: bool
    details: str
    timestamp: float

class Evaluator:
    def __init__(self,output="runs/eval.jsonl"): self.output=Path(output); self.output.parent.mkdir(parents=True,exist_ok=True)
    def record(self,rec:EvalRecord):
        with self.output.open("a",encoding="utf-8") as f:f.write(json.dumps(asdict(rec),ensure_ascii=False)+"\n")
    def verify_answer(self,task,pred,target,verifier="math_exact"):
        r=verify(verifier,pred,target); self.record(EvalRecord(task,r.reward,r.passed,r.details,time.time())); return r
