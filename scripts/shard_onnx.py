import argparse, json
from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "site" / "public" / "model" / "tiny50m.onnx"))
    p.add_argument("--dst", default=str(ROOT / "website" / "model"))
    p.add_argument("--name", default="tiny50m-fp16")
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    graph = dst / f"{a.name}.onnx"
    manifest = dst / f"{a.name}-manifest.json"
    model = onnx.load(str(src), load_external_data=False)
    onnx.save(
        model,
        str(graph),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
        convert_attribute=False,
    )
    files = sorted(p.name for p in dst.iterdir() if p.is_file() and not p.name.endswith((".json", ".onnx", ".nojekyll")))
    (dst / manifest.name).write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")
    total = sum((dst / f).stat().st_size for f in files)
    print(f"shards: {len(files)}  total: {total/1e6:.1f} MB  graph: {graph.stat().st_size} bytes")

if __name__ == "__main__":
    main()