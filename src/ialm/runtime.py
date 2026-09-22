from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import json

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    risk: float = 0.0

class ToolRegistry:
    def __init__(self): self._tools: dict[str,Tool] = {}
    def register(self,name,description,fn,risk=0.0): self._tools[name]=Tool(name,description,fn,risk); return fn
    def get(self,name): return self._tools[name]
    def schemas(self):
        return [{"name":t.name,"description":t.description,"risk":t.risk} for t in self._tools.values()]
    def execute(self,name,arguments):
        t=self.get(name)
        if not isinstance(arguments,dict): raise TypeError("tool arguments must be an object")
        return t.fn(**arguments)

class AgentRuntime:
    """Minimal execution boundary for the model-native action interface.

    The neural checkpoint does not execute arbitrary code. This runtime validates and executes
    only registered tools, then returns a structured observation to the model.
    """
    def __init__(self,registry:ToolRegistry, max_steps:int=16, max_tool_risk:float=0.5):
        self.registry=registry; self.max_steps=max_steps; self.max_tool_risk=max_tool_risk
    def execute_tool(self,name,arguments):
        tool=self.registry.get(name)
        if tool.risk>self.max_tool_risk: raise PermissionError(f"tool risk {tool.risk} exceeds runtime budget")
        result=self.registry.execute(name,arguments)
        try: payload=json.loads(json.dumps(result,ensure_ascii=False))
        except Exception: payload=str(result)
        return {"tool":name,"ok":True,"result":payload}
