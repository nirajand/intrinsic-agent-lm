from __future__ import annotations
from pathlib import Path
import json
from safetensors.torch import save_file, load_file
from .config import ModelConfig
from .model import IntrinsicAgentLM

def _materialize(sd): return {k:v.detach().contiguous().clone() for k,v in sd.items()}

def save_checkpoint(model,path,metadata=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    meta={"format":"ialm.safetensors.v2","model_config":json.dumps(model.cfg.to_dict(),sort_keys=True)}
    if metadata: meta.update({str(k):str(v) for k,v in metadata.items()})
    save_file(_materialize(model.state_dict()),str(path),metadata=meta)

def load_checkpoint(path,device="cpu"):
    tensors=load_file(str(path),device=device)
    from safetensors import safe_open
    with safe_open(str(path),framework="pt",device=device) as f: md=f.metadata() or {}
    cfg=ModelConfig(**json.loads(md["model_config"])); model=IntrinsicAgentLM(cfg); model.load_state_dict(tensors,strict=True); return model,md
