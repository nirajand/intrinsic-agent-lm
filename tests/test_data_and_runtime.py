from ialm.processors import dpo, reward, rlvr
from ialm.runtime import ToolRegistry, AgentRuntime
from ialm.verifiers import verify

def test_helpsteer_pair_processor():
    row={"prompt":"p","response_1":"bad","response_2":"good","preference_strength":2}
    x=dpo(row); assert x["chosen"]=="good" and x["rejected"]=="bad"
    y=reward(row); assert y["chosen"]=="good"

def test_runtime():
    r=ToolRegistry(); r.register("add","add two numbers",lambda a,b:a+b)
    out=AgentRuntime(r).execute_tool("add",{"a":2,"b":3}); assert out["ok"] and out["result"]==5

def test_verifiers():
    assert verify("math_exact","The answer is 42","42").passed
    assert verify("json_valid",'{"x": 1}','').passed
