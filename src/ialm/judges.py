from __future__ import annotations
import json,re
from dataclasses import dataclass

@dataclass
class Judgment:
    winner: str
    score_a: float
    score_b: float
    rationale: str

class AIFeedbackJudge:
    """Interface for an independently versioned AI evaluator used by RLAIF.

    The evaluator should be kept separate from the policy being optimized. The default implementation
    uses a Transformers causal LM and asks for machine-readable scores; for high-stakes research, use
    a separately managed judge service with held-out calibration and audit logs.
    """
    def __init__(self,model_id:str,revision=None,device="auto"):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
        self.model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,device_map=device,torch_dtype="auto")
    def judge(self,prompt,a,b):
        text=("You are an evaluator. Compare two candidate answers for the same user request. "
              "Return ONLY JSON with keys winner (A/B/TIE), score_a, score_b, rationale.\n\n"
              f"USER:\n{prompt}\n\nA:\n{a}\n\nB:\n{b}\n")
        ids=self.tokenizer(text,return_tensors="pt").to(self.model.device)
        out=self.model.generate(**ids,max_new_tokens=160,do_sample=False)
        raw=self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True)
        m=re.search(r"\{.*\}",raw,re.S)
        if not m: raise ValueError(f"judge did not return JSON: {raw[:200]}")
        obj=json.loads(m.group(0)); return Judgment(str(obj["winner"]),float(obj["score_a"]),float(obj["score_b"]),str(obj.get("rationale","")))
