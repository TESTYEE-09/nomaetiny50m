import argparse, sys, torch
from pathlib import Path
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizer.chat import format_chat
from training.model import TinyLM, ModelConfig

def load(checkpoint,tokenizer):
    if not torch.cuda.is_available(): raise SystemExit("GPU REQUIRED: refusing CPU fallback")
    ck=torch.load(checkpoint,map_location="cuda",weights_only=False); m=TinyLM(ModelConfig(**ck["config"])).cuda(); m.load_state_dict(ck["model"]); m.eval(); return m,Tokenizer.from_file(tokenizer)

@torch.no_grad()
def generate(m,tok,prompt,max_new=256,temp=.8,top_k=40):
    text=format_chat([{"role":"user","content":prompt}],True); ids=torch.tensor([tok.encode(text).ids],device="cuda")
    eos=tok.token_to_id("<|eos|>")
    for _ in range(max_new):
        logits,_=m(ids[:,-m.config.max_position_embeddings:]); next_logits=logits[:,-1]/max(temp,1e-5); values,indices=torch.topk(next_logits,min(top_k,next_logits.size(-1))); probs=torch.softmax(values,-1); nxt=indices.gather(-1,torch.multinomial(probs,1)); ids=torch.cat((ids,nxt),1)
        if nxt.item()==eos: break
    return tok.decode(ids[0,tok.encode(text).ids.__len__():].tolist(),skip_special_tokens=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("prompt"); p.add_argument("--checkpoint",default="checkpoints/latest.pt"); p.add_argument("--tokenizer",default="tokenizer/tokenizer.json"); p.add_argument("--max-new",type=int,default=256); a=p.parse_args(); m,t=load(a.checkpoint,a.tokenizer); print(generate(m,t,a.prompt,a.max_new))

