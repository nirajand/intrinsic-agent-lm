from __future__ import annotations
import json

from .data import choose_text, normalize_chat

def pretrain(row):
    text=choose_text(row, ["text","content","paper_text","full_content","solution","code"])
    if not text:
        task=row.get("task"); solution=row.get("solution"); analysis=row.get("analysis")
        parts=[x for x in (task,analysis,solution) if isinstance(x,str) and x.strip()]
        text="\n\n".join(parts) if parts else None
    return {"text":text.strip()} if isinstance(text,str) and len(text.strip())>=8 else None

def sft(row):
    msgs=normalize_chat(row)
    if msgs and len(msgs)>=2: return {"messages":msgs,"tools":row.get("tools",[]) or row.get("tools_json",[]) or []}
    instruction=row.get("instruction") or row.get("question") or row.get("task") or row.get("prompt")
    output=row.get("output") or row.get("response") or row.get("answer") or row.get("solution")
    if isinstance(instruction,str) and isinstance(output,str) and output.strip():
        return {"messages":[{"role":"user","content":instruction},{"role":"assistant","content":output}],"tools":row.get("tools",[])}
    return None

def scientific_sft(row):
    instruction=row.get("instruction") or row.get("question")
    output=row.get("output") or row.get("answer")
    if not isinstance(instruction,str) or not isinstance(output,str): return None
    return {"messages":[{"role":"user","content":instruction},{"role":"assistant","content":output}]}

def code_task_sft(row):
    task=row.get("task"); solution=row.get("solution") or row.get("code")
    if not isinstance(task,str) or not isinstance(solution,str): return None
    analysis=row.get("analysis")
    assistant=(analysis.strip()+"\n\n" if isinstance(analysis,str) and analysis.strip() else "")+solution
    return {"messages":[{"role":"user","content":task},{"role":"assistant","content":assistant}]}

def _response_text(value):
    if isinstance(value,str): return value
    if isinstance(value,list):
        for item in reversed(value):
            if isinstance(item,dict) and item.get("role") in {"assistant","gpt","model"}: return str(item.get("content") or item.get("value") or "")
        for item in reversed(value):
            if isinstance(item,dict): return str(item.get("content") or item.get("value") or "")
    return ""

def dpo(row):
    prompt=row.get("prompt") or row.get("instruction") or row.get("question")
    chosen=row.get("chosen") or row.get("preferred")
    rejected=row.get("rejected") or row.get("dispreferred")
    if chosen is None and row.get("response_1") is not None and row.get("response_2") is not None:
        strength=float(row.get("preference_strength",0))
        if strength>0: chosen,rejected=row["response_2"],row["response_1"]
        elif strength<0: chosen,rejected=row["response_1"],row["response_2"]
        else: return None
    chosen=_response_text(chosen); rejected=_response_text(rejected)
    if prompt and chosen and rejected: return {"prompt":str(prompt),"chosen":chosen,"rejected":rejected}
    return None

def _first_present(row,keys):
    for k in keys:
        if row.get(k) is not None: return row[k]
    return None

def reward(row):
    d=dpo(row)
    if not d:return None
    score=_first_present(row,("score","overall_score","reward"))
    if score is None:
        attrs=[row.get(k) for k in ("helpfulness","correctness","coherence")]
        if all(v is not None for v in attrs): score=sum(float(v) for v in attrs)/3.0
    out={**d}
    if score is not None:
        try: out["score"]=float(score)
        except Exception: pass
    return out

def rlvr(row):
    prompt=row.get("problem") or row.get("prompt") or row.get("question") or row.get("query")
    answer=row.get("answer") or row.get("solution") or row.get("expected_answer") or row.get("ground_truth")
    if not prompt:return None
    verifier=row.get("verifier","math_exact")
    # Preserve structured code test cases for the downstream sandbox verifier.
    if isinstance(answer,(dict,list)): answer=json.dumps(answer,ensure_ascii=False)
    return {"prompt":str(prompt),"answer":str(answer or ""),"verifier":verifier,"domain":row.get("domain"),"source":row.get("source")}

def tool_trace(row):
    msgs=normalize_chat(row)
    return {"messages":msgs,"tools":row.get("tools",[]) or row.get("tools_json",[])} if msgs else None
