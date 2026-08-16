import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

tok = Tokenizer.from_file(str(ROOT / "website" / "model" / "tokenizer.json"))

def generate(model_path, prompt, dtype, max_new=40, temperature=0.78, top_k=36, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    prompt = f"<|bos|><|system|>You are Tiny50M, a concise helpful assistant.<|eos|><|user|>{prompt}<|eos|><|assistant|>"
    ids = tok.encode(prompt, add_special_tokens=False).ids
    feeds = {"input_ids": np.array([ids], dtype=np.int64)}
    for layer in range(12):
        for kind in ("key", "value"):
            feeds[f"past_{layer}_{kind}"] = np.zeros((1, 2, 0, 64), dtype=dtype)
    output = session.run(None, feeds)
    names = [o.name for o in session.get_outputs()]
    generated = []
    for _ in range(max_new):
        logits = output[0][0]
        logits = logits.astype(np.float64) / temperature
        k = min(top_k, len(logits))
        top = np.argpartition(logits, -k)[-k:]
        top = top[np.argsort(logits[top])]
        weights = np.exp(logits[top] - logits[top[-1]])
        token = int(rng.choice(top, p=weights / weights.sum()))
        if token == 1:
            break
        generated.append(token)
        next_feeds = {"input_ids": np.array([[token]], dtype=np.int64)}
        for index, name in enumerate(names[1:]):
            next_feeds[name.replace("present_", "past_")] = output[index + 1]
        output = session.run(None, next_feeds)
    return tok.decode(generated, skip_special_tokens=True)

for name, path, dtype in [
    ("fp16", ROOT / "website" / "model" / "tiny50m-fp16.onnx", np.float16),
    ("int8", ROOT / "website" / "model" / "tiny50m-int8.onnx", np.float32),
]:
    print(f"--- {name} ---")
    print("hello:", generate(path, "hello", dtype)[:120])
    print("add:", generate(path, "write a python function that adds two numbers", dtype)[:120])