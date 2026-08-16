import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np, torch
from model import TinyLM, ModelConfig

def save(path,m,opt,step,tokens,args):
    path.parent.mkdir(parents=True,exist_ok=True)
    state={"model":m.state_dict(),"optimizer":opt.state_dict(),"step":step,"tokens":tokens,"args":vars(args),"config":m.config.__dict__,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state()}
    temp=path.with_suffix(".tmp"); torch.save(state,temp); os.replace(temp,path)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/processed/train.bin"); p.add_argument("--steps",type=int,default=1000); p.add_argument("--seq",type=int,default=2048); p.add_argument("--grad-accum",type=int,default=16); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--warmup",type=float,default=.02); p.add_argument("--save-every",type=int,default=100); p.add_argument("--resume"); p.add_argument("--output",default="checkpoints/latest.pt"); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("GPU REQUIRED: refusing CPU fallback")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device="cuda"
    data=np.memmap(a.data,dtype=np.uint16,mode="r")
    if len(data) < a.seq + 1: raise SystemExit(f"Need at least {a.seq + 1} tokens; found {len(data)}")
    m=TinyLM().to(device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.1); start_step=tokens=0
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start_step=ck["step"]; tokens=ck["tokens"]
    torch.cuda.reset_peak_memory_stats(); began=time.perf_counter()
    for step in range(start_step,a.steps):
        opt.zero_grad(set_to_none=True); total_loss=0
        local=step-start_step+1; span=max(1,a.steps-start_step); progress=local/span; warm=max(1,int(span*a.warmup)); scale=local/warm if local<warm else .1+.9*.5*(1+math.cos(math.pi*(local-warm)/max(1,span-warm)))
        for group in opt.param_groups: group["lr"]=a.lr*scale
        for _ in range(a.grad_accum):
            pos=random.randrange(0,len(data)-a.seq); ids=torch.from_numpy(np.array(data[pos:pos+a.seq+1],dtype=np.int64)).unsqueeze(0).to(device)
            with torch.autocast("cuda",dtype=torch.float16): _,loss=m(ids,ids); scaled=loss/a.grad_accum
            scaled.backward(); total_loss += loss.item(); tokens += a.seq
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if (step+1)%10==0 or step==start_step:
            torch.cuda.synchronize(); elapsed=time.perf_counter()-began
            print(json.dumps({"label":"MEASURED","step":step+1,"tokens":tokens,"loss":total_loss/a.grad_accum,"lr":opt.param_groups[0]["lr"],"tokens_per_second":(tokens-(0 if start_step==0 else ck["tokens"]))/elapsed,"peak_vram_bytes":torch.cuda.max_memory_allocated()}),flush=True)
        if (step+1)%a.save_every==0: save(Path(a.output),m,opt,step+1,tokens,a); save(Path(a.output).with_name(f"step-{step+1}.pt"),m,opt,step+1,tokens,a)
    save(Path(a.output),m,opt,a.steps,tokens,a)
if __name__=="__main__": main()
