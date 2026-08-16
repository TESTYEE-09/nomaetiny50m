import torch, sys
sys.path.insert(0, ".")
import training.model as mm
from tokenizers import Tokenizer

ck = torch.load("checkpoints/mixed2-production.pt", map_location="cpu", weights_only=False)
m = mm.TinyLM(mm.ModelConfig(**ck["config"])).eval()
m.load_state_dict(ck["model"])
m = m.to("cuda").half()
tok = Tokenizer.from_file("tokenizer/tokenizer.json")


def gen(prompt, max_new=40, temp=0.78, topk=36):
    ids = torch.tensor([tok.encode(prompt).ids], device="cuda")
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = m(ids)[0][:, -1, :].float()
            l = logits[0] / temp
            k = min(topk, l.shape[0])
            top = torch.topk(l, k).indices
            w = torch.softmax(l[top], dim=0)
            ti = top[torch.multinomial(w, 1)].item()
            if ti == 1:
                break
            out.append(ti)
            ids = torch.cat([ids, torch.tensor([[ti]], device="cuda")], dim=1)
    return tok.decode(out, skip_special_tokens=True)


q = "whats 1+1"
print("WITH system (app.js):", gen(f"<|bos|><|system|>You are Tiny50M, a concise helpful assistant.<|eos|><|user|>{q}<|eos|><|assistant|>"))
print("NO system (training):", gen(f"<|bos|><|user|>{q}<|eos|><|assistant|>"))
print("training fmt france:", gen(f"<|bos|><|user|>What is the capital of France?<|eos|><|assistant|>"))
print("training fmt add func:", gen(f"<|bos|><|user|>Write a python function that adds two numbers.<|eos|><|assistant|>"))
