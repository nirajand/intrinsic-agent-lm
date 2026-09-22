from pathlib import Path
from ialm.data import DatasetManager, DatasetSpec
from ialm.processors import dpo, reward, rlvr
from ialm.runtime import ToolRegistry, AgentRuntime
from ialm.verifiers import verify

def test_helpsteer_pair_processor():
    row={"prompt":"p","response_1":"bad","response_2":"good","preference_strength":2}
    x=dpo(row); assert x["chosen"]=="good" and x["rejected"]=="bad"
    y=reward(row); assert y["chosen"]=="good"

def test_reward_score_zero_is_kept():
    row={"prompt":"p","response_1":"a","response_2":"b","preference_strength":1,"score":0}
    y=reward(row); assert y["score"]==0.0

def test_runtime():
    r=ToolRegistry(); r.register("add","add two numbers",lambda a,b:a+b)
    out=AgentRuntime(r).execute_tool("add",{"a":2,"b":3}); assert out["ok"] and out["result"]==5

def test_runtime_max_steps():
    r=ToolRegistry(); r.register("add","add",lambda a,b:a+b)
    rt=AgentRuntime(r,max_steps=1)
    rt.execute_tool("add",{"a":1,"b":1})
    try:
        rt.execute_tool("add",{"a":1,"b":1})
        assert False, "expected max_steps error"
    except RuntimeError as e:
        assert "max_steps" in str(e)

def test_verifiers():
    assert verify("math_exact","The answer is 42","42").passed
    assert verify("json_valid",'{"x": 1}','').passed

def test_prepare_dotted_name_and_local_jsonl(tmp_path):
    src=tmp_path/"openhermes-2.5.jsonl"
    src.write_text('{"text":"hello world fixture"}\n',encoding="utf-8")
    mgr=DatasetManager(tmp_path/"data")
    spec=DatasetSpec(name="openhermes-2.5",repo=str(src),max_examples=10)
    def proc(row): return {"text":row["text"]}
    out=mgr.prepare(spec,"sft",proc)
    assert out.name=="openhermes-2.5.jsonl"
    assert out.exists()
    rows=list(mgr.iter_rows(DatasetSpec(name="openhermes-2.5",repo=str(src),max_examples=1)))
    assert rows==[{"text":"hello world fixture"}]

def test_download_requires_cap(tmp_path):
    mgr=DatasetManager(tmp_path/"data")
    spec=DatasetSpec(name="uncapped",repo="org/repo",streaming=True,max_examples=None)
    try:
        mgr.download(spec)
        assert False, "expected cap error"
    except ValueError as e:
        assert "max_examples" in str(e)
