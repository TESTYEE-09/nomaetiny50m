import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.model import TinyLM,ModelConfig

p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/latest.pt"); p.add_argument("--output",default="export/model-fp16.pt"); a=p.parse_args()
ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=TinyLM(ModelConfig(**ck["config"])); model.load_state_dict(ck["model"]); model.half()
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
torch.save({"format":"tiny50m-fp16-state-dict-v1","config":ck["config"],"model":model.state_dict(),"trained_tokens":ck["tokens"],"step":ck["step"]},out)
meta={"file":out.name,"bytes":out.stat().st_size,"parameters":model.parameter_count(),"precision":"fp16","trained_tokens":ck["tokens"],"step":ck["step"]}
out.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta))

