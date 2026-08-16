import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def generate(tokenizer, model_path, prompt, dtype, max_new=40, temperature=0.78, top_k=36, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    prompt = f"<|bos|><|system|>You are Tiny50M, a concise helpful assistant.<|eos|><|user|>{prompt}<|eos|><|assistant|>"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
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
    return tokenizer.decode(generated, skip_special_tokens=True)

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "website/model-wiki"
    model_dir = Path(target)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    print(f"tokenizer vocab: {tokenizer.get_vocab_size()}")
    for name, path, dtype in [
        ("fp16", model_dir / "tiny50m-wiki-fp16.onnx", np.float16),
        ("int8", model_dir / "tiny50m-wiki-int8.onnx", np.float32),
    ]:
        if not path.exists():
            print(f"skip {name}: missing {path}")
            continue
        print(f"--- {name} ---")
        for probe in ["what is 1 + 1?", "write a python function that adds two numbers", "What is the capital of France?"]:
            print(f"{probe} -> {generate(tokenizer, path, probe, dtype, max_new=40)[:140]}")

if __name__ == "__main__":
    main()