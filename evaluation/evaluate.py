import argparse, json, time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.infer import load,generate
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/latest.pt"); p.add_argument("--tokenizer",default="tokenizer/tokenizer.json"); p.add_argument("--output",default="evaluation/results.jsonl"); p.add_argument("--max-new",type=int,default=512); a=p.parse_args()
m,t=load(a.checkpoint,a.tokenizer); prompts=json.loads(Path("evaluation/prompts.json").read_text()); out=Path(a.output)
with out.open("w",encoding="utf-8") as f:
    for prompt in prompts:
        began=time.perf_counter(); answer=generate(m,t,prompt,a.max_new); record={"prompt":prompt,"answer":answer,"seconds":time.perf_counter()-began}; f.write(json.dumps(record,ensure_ascii=False)+"\n"); print(prompt,repr(answer[:100]))

