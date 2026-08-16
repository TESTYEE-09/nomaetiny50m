import argparse, json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/processed/chat.jsonl"); p.add_argument("--tokenizer",default="tokenizer/tokenizer.json"); p.add_argument("--seq",type=int,default=2048); p.add_argument("--output",default="data/processed/train.bin"); a=p.parse_args()
    tok=Tokenizer.from_file(a.tokenizer); ids=[]
    for line in Path(a.input).read_text(encoding="utf-8").splitlines(): ids.extend(tok.encode(json.loads(line)["text"]).ids)
    usable=(len(ids)//(a.seq+1))*(a.seq+1)
    if usable==0: raise SystemExit(f"Need at least {a.seq+1} tokens; found {len(ids)}")
    array=np.asarray(ids[:usable],dtype=np.uint16); Path(a.output).parent.mkdir(parents=True,exist_ok=True); array.tofile(a.output)
    manifest={"token_count":int(usable),"sequence_length":a.seq,"dtype":"uint16","source":a.input,"tokenizer":a.tokenizer}
    Path(a.output+".json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest))
if __name__=="__main__": main()

