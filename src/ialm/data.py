from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json, os, time

_FULL_OK = "IALM_ALLOW_FULL_DOWNLOAD"

@dataclass
class DatasetSpec:
    name: str
    repo: str
    subset: str | None = None
    split: str = "train"
    revision: str | None = None
    streaming: bool = False
    trust_remote_code: bool = False
    max_examples: int | None = None
    weight: float = 1.0
    text_fields: list[str] | None = None
    format: str = "auto"
    license: str = "unknown"
    commercial_use: str = "unknown"
    requires_hf_acceptance: bool = False
    provenance_review: str = "required"
    intended_stage: str | None = None
    enabled_in_recipe: bool | None = None
    notes: str = ""

class DatasetRegistry:
    def __init__(self, root="data/catalog.json"):
        self.root=Path(root); self.root.parent.mkdir(parents=True,exist_ok=True); self.items=self._load()
    def _load(self): return json.loads(self.root.read_text()) if self.root.exists() else {}
    def add(self,spec): self.items[spec.name]=asdict(spec); self.root.write_text(json.dumps(self.items,indent=2,sort_keys=True))
    def add_many(self,specs):
        for spec in specs: self.items[spec.name]=asdict(spec)
        self.root.write_text(json.dumps(self.items,indent=2,sort_keys=True))
    def get(self,name):
        if name not in self.items: raise KeyError(name)
        return DatasetSpec(**self.items[name])
    def all(self): return {name: DatasetSpec(**spec) for name,spec in self.items.items()}

class DatasetManager:
    def __init__(self,root="data"):
        self.root=Path(root); self.raw=self.root/"raw"; self.prepared=self.root/"prepared"; self.manifest=self.root/"manifest.jsonl"
        self.raw.mkdir(parents=True,exist_ok=True); self.prepared.mkdir(parents=True,exist_ok=True)
    def _log(self,event):
        with self.manifest.open("a",encoding="utf-8") as f:f.write(json.dumps({"time":time.time(),**event},ensure_ascii=False)+"\n")
    def _check_cap(self,spec,lim):
        if lim is None and not os.environ.get(_FULL_OK):
            raise ValueError(
                f"dataset {spec.name!r} has no max_examples; set max_examples in the recipe "
                f"or set {_FULL_OK}=1 to allow full materialization"
            )
    def _local_path(self,spec):
        p=Path(spec.repo)
        return p if p.suffix==".jsonl" and p.is_file() else None
    def load(self,spec):
        if self._local_path(spec): return None
        try: from datasets import load_dataset
        except ImportError as e: raise RuntimeError("Install optional data dependencies: pip install -e '.[data]'") from e
        kwargs={"split":spec.split,"streaming":spec.streaming,"trust_remote_code":spec.trust_remote_code}
        if spec.subset: kwargs["name"]=spec.subset
        if spec.revision: kwargs["revision"]=spec.revision
        ds=load_dataset(spec.repo,**kwargs)
        self._log({"event":"load","name":spec.name,"repo":spec.repo,"subset":spec.subset,"split":spec.split,"streaming":spec.streaming,"license":spec.license,"requires_hf_acceptance":spec.requires_hf_acceptance})
        return ds
    def download(self,spec,limit=None):
        lim=limit if limit is not None else spec.max_examples
        self._check_cap(spec,lim)
        local=self._local_path(spec)
        if local is not None:
            self._log({"event":"download_local","name":spec.name,"path":str(local)}); return local
        ds=self.load(spec)
        out=self.raw/spec.name
        if spec.streaming:
            out.mkdir(parents=True,exist_ok=True); path=out/"data.jsonl"; n=0
            with path.open("w",encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(row,ensure_ascii=False)+"\n"); n+=1
                    if lim and n>=lim: break
            self._log({"event":"download_stream","name":spec.name,"rows":n,"path":str(path)}); return path
        if lim is not None:
            try:
                if len(ds)>lim: ds=ds.select(range(lim))
            except TypeError: pass
        out.mkdir(parents=True,exist_ok=True); ds.save_to_disk(str(out))
        self._log({"event":"download","name":spec.name,"rows":len(ds),"path":str(out)}); return out
    def _iter_jsonl(self,path,max_examples):
        with path.open(encoding="utf-8") as f:
            for i,line in enumerate(f):
                if max_examples and i>=max_examples: break
                line=line.strip()
                if line: yield json.loads(line)
    def iter_rows(self,spec):
        local=self._local_path(spec)
        if local is not None:
            yield from self._iter_jsonl(local,spec.max_examples); return
        path=self.raw/spec.name
        if path.exists() and path.is_dir():
            jsonl=path/"data.jsonl"
            if jsonl.is_file():
                yield from self._iter_jsonl(jsonl,spec.max_examples); return
            from datasets import load_from_disk
            ds=load_from_disk(str(path))
            for i,row in enumerate(ds):
                if spec.max_examples and i>=spec.max_examples: break
                yield row
            return
        if path.exists() and path.is_file():
            yield from self._iter_jsonl(path,spec.max_examples); return
        self._check_cap(spec,spec.max_examples)
        ds=self.load(spec)
        if ds is None: return
        for i,row in enumerate(ds):
            if spec.max_examples and i>=spec.max_examples: break
            yield row
    def prepare(self,spec,stage,processor,tokenizer=None,max_seq_len=None):
        out=self.prepared/stage/spec.name; out.parent.mkdir(parents=True,exist_ok=True)
        path=out.parent/(out.name+".jsonl")
        n=0
        with path.open("w",encoding="utf-8") as f:
            for row in self.iter_rows(spec):
                item=processor(row)
                if item is None: continue
                if tokenizer is not None: item=tokenizer(item,max_seq_len=max_seq_len)
                f.write(json.dumps(item,ensure_ascii=False)+"\n"); n+=1
        sha=hashlib.sha256(path.read_bytes()).hexdigest(); self._log({"event":"prepare","name":spec.name,"stage":stage,"rows":n,"sha256":sha,"path":str(path),"license":spec.license}); return path

def choose_text(row,fields=None):
    fields=fields or ["text","content","paper_text","full_content","prompt"]
    for k in fields:
        v=row.get(k)
        if isinstance(v,str) and v.strip(): return v
    return None

def normalize_chat(row):
    msgs=row.get("messages") or row.get("conversations") or row.get("chat")
    if isinstance(msgs,list):
        out=[]
        for m in msgs:
            if not isinstance(m,dict): continue
            role=m.get("role") or m.get("from")
            content=m.get("content") or m.get("value") or ""
            role={"human":"user","gpt":"assistant","system":"system","tool":"tool","model":"assistant"}.get(role,role)
            if role not in {"system","user","assistant","tool","developer"}: continue
            item={"role":role,"content":str(content)}
            for key in ("tool_calls","tool_call_id","name"):
                if key in m: item[key]=m[key]
            out.append(item)
        if out:return out
    text=choose_text(row); return [{"role":"user","content":text}] if text else []
