from __future__ import annotations
import argparse, json
from pathlib import Path
from .pipeline import TrainingPipeline
from .data import DatasetRegistry,DatasetSpec

def main():
    ap=argparse.ArgumentParser(prog="ialm"); sub=ap.add_subparsers(dest="cmd",required=True)
    d=sub.add_parser("data"); ds=d.add_subparsers(dest="sub",required=True)
    a=ds.add_parser("register"); a.add_argument("name"); a.add_argument("repo"); a.add_argument("--subset"); a.add_argument("--split",default="train"); a.add_argument("--streaming",action="store_true"); a.add_argument("--max-examples",type=int)
    a=ds.add_parser("download"); a.add_argument("recipe"); a.add_argument("--stage")
    a=ds.add_parser("prepare"); a.add_argument("recipe"); a.add_argument("stage")
    t=sub.add_parser("train"); t.add_argument("recipe")
    args=ap.parse_args()
    if args.cmd=="data" and args.sub=="register":
        r=DatasetRegistry(); r.add(DatasetSpec(name=args.name,repo=args.repo,subset=args.subset,split=args.split,streaming=args.streaming,max_examples=args.max_examples)); print("registered",args.name); return
    if args.cmd=="data" and args.sub=="download":
        p=TrainingPipeline(args.recipe); p.sync_data(args.stage); return
    if args.cmd=="data" and args.sub=="prepare":
        p=TrainingPipeline(args.recipe); print([str(x) for x in p.prepare_data(args.stage)]); return
    if args.cmd=="train": TrainingPipeline(args.recipe).run(); return

if __name__=="__main__": main()
