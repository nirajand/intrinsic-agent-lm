from __future__ import annotations
from typing import Any

class HFTokenizerAdapter:
    def __init__(self, tokenizer_id:str, revision=None):
        try: from transformers import AutoTokenizer
        except ImportError as e: raise RuntimeError("Install transformers: pip install -e '.[data]'") from e
        self.tok=AutoTokenizer.from_pretrained(tokenizer_id,revision=revision,use_fast=True)
        if self.tok.pad_token_id is None: self.tok.pad_token=self.tok.eos_token
    def text(self,text,max_seq_len):
        x=self.tok(text,truncation=True,max_length=max_seq_len,return_attention_mask=True)
        return {"input_ids":x["input_ids"],"attention_mask":x["attention_mask"]}
    def chat(self,messages,max_seq_len):
        text=self.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        return self.text(text,max_seq_len)|{"messages":messages}
    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)
    def __call__(self,item,max_seq_len=None):
        m=max_seq_len or self.tok.model_max_length
        if "text" in item:return self.text(item["text"],m)|{"text":item["text"]}
        if "messages" in item:return self.chat(item["messages"],m)
        if "prompt" in item:
            p=self.text(item["prompt"],m)
            c=self.text(item["chosen"],m); r=self.text(item["rejected"],m)
            return {"prompt":item["prompt"],"chosen":item["chosen"],"rejected":item["rejected"],"prompt_ids":p["input_ids"],"chosen_ids":c["input_ids"],"rejected_ids":r["input_ids"]}
        if "answer" in item:
            return item | {"prompt_ids":self.text(item["prompt"],m)["input_ids"],"answer_ids":self.text(item["answer"],m)["input_ids"]}
        return item
