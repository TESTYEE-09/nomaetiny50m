import json, subprocess, sys, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "overnight_pipeline.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as f: f.write(line + "\n")
    print(line, flush=True)

def run(args, timeout=None):
    log("RUN " + " ".join(args))
    r = subprocess.run([PY] + args, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "").strip().splitlines()
    for l in out[-3:]: log("  " + l)
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        for l in err[-8:]: log("  ERR " + l)
        raise SystemExit(f"failed: {args[0]} (exit {r.returncode})")
    return "\n".join(out)

def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase in ("all", "prepare"):
        for _ in range(300):
            if (ROOT / "data" / "wiki_done.json").exists(): break
            time.sleep(20)
        done = json.loads((ROOT / "data" / "wiki_done.json").read_text(encoding="utf-8"))
        log(f"downloads done: {done}")

        (ROOT / "data" / "empty_raw").mkdir(exist_ok=True)
        (ROOT / "data" / "empty_luna").mkdir(exist_ok=True)
        run(["data/process.py", "--raw", "data/empty_raw", "--luna", "data/empty_luna", "--hf", "data/wiki_raw", "--output", "data/processed-wiki"])

        log("retraining tokenizer on new corpus...")
        run(["tokenizer/train_tokenizer.py", "--input", "data/processed-wiki/tokenizer_corpus.txt", "--output", "tokenizer/tokenizer-wiki.json"])

        run(["data/pack.py", "--input", "data/processed-wiki/chat.jsonl", "--tokenizer", "tokenizer/tokenizer-wiki.json", "--seq", "512", "--output", "data/processed/wiki.bin"])
        manifest = json.loads((ROOT / "data" / "processed" / "wiki.bin.json").read_text(encoding="utf-8"))
        tokens = manifest["token_count"]
        steps = int(tokens / 2048 * 2.5)
        log(f"corpus tokens={tokens} steps={steps}")
        (ROOT / "data" / "wiki_steps.json").write_text(json.dumps({"tokens": tokens, "steps": steps}))

    if phase in ("all", "train"):
        steps = json.loads((ROOT / "data" / "wiki_steps.json").read_text(encoding="utf-8"))["steps"]
        run(["training/train.py", "--data", "data/processed/wiki.bin", "--steps", str(steps), "--seq", "512", "--grad-accum", "4", "--save-every", str(max(2000, steps // 8)), "--output", "checkpoints/wiki-production.pt"])

    if phase in ("all", "ship"):
        run(["export/export_weights.py", "--checkpoint", "checkpoints/wiki-production.pt", "--output", "export/model-wiki-fp16.pt"])
        run(["scripts/export_onnx.py", "--fp32", "--model", "export/model-wiki-fp16.pt", "--name", "tiny50m-wiki", "--target", "site/public/model-wiki"])
        run(["scripts/export_onnx.py", "--model", "export/model-wiki-fp16.pt", "--name", "tiny50m-wiki-fp16", "--target", "site/public/model-wiki"])
        run(["scripts/shard_onnx.py", "--src", "site/public/model-wiki/tiny50m-wiki-fp16.onnx", "--dst", "website/model-wiki", "--name", "tiny50m-wiki"])
        run(["scripts/quantize_onnx.py", "--src", "site/public/model-wiki/tiny50m-wiki-fp32.onnx", "--dst", "website/model-wiki/tiny50m-wiki-int8.onnx"])
        for f in ("tokenizer.json", "tokenizer_config.json"):
            src = ROOT / "website" / "model" / f if f == "tokenizer_config.json" else ROOT / "tokenizer" / "tokenizer-wiki.json"
            (ROOT / "website" / "model-wiki" / f).parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, ROOT / "website" / "model-wiki" / f)

    log("PIPELINE OK")

if __name__ == "__main__":
    main()