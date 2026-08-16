import argparse, json, time, torch
from model import TinyLM

def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=5); p.add_argument("--seq",type=int,default=256); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("GPU REQUIRED: torch.cuda.is_available() is false; refusing CPU fallback")
    device=torch.device("cuda"); m=TinyLM().to(device); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    amp=torch.float16; torch.cuda.reset_peak_memory_stats(); tokens=0; torch.cuda.synchronize(); start=time.perf_counter()
    for step in range(a.steps):
        ids=torch.randint(0,m.config.vocab_size,(1,a.seq),device=device); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=amp): _,loss=m(ids,ids)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step(); tokens += ids.numel()
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    print(json.dumps({"label":"MEASURED","gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"hip":torch.version.hip,"precision":"fp16","steps":a.steps,"sequence_length":a.seq,"tokens_per_second":tokens/elapsed,"peak_vram_bytes":torch.cuda.max_memory_allocated(),"loss":loss.item()}))
if __name__=="__main__": main()

