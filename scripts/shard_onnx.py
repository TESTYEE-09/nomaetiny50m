import json, sys
from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "site" / "public" / "model" / "tiny50m.onnx"
DST = ROOT / "website" / "model"
GRAPH = "tiny50m-fp16.onnx"
MANIFEST = "tiny50m-fp16-manifest.json"

model = onnx.load(str(SRC), load_external_data=False)
onnx.save(
    model,
    str(DST / GRAPH),
    save_as_external_data=True,
    all_tensors_to_one_file=False,
    size_threshold=0,
    convert_attribute=False,
)
files = sorted(p.name for p in DST.iterdir() if p.is_file() and p.name != GRAPH and p.name != MANIFEST and not p.name.endswith((".json", ".nojekyll")))
manifest = {"files": files}
(DST / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
total = sum((DST / f).stat().st_size for f in files)
print(f"shards: {len(files)}  total: {total/1e6:.1f} MB  graph: {(DST/GRAPH).stat().st_size} bytes")