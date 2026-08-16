import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data" / "hf_raw" / "synthetic.jsonl"
records = [json.loads(line) for line in out.open(encoding="utf-8")]
print(f"base synthetic: {len(records)}")

lines = []
for rec in records:
    for _ in range(8):
        lines.append(json.dumps(rec, ensure_ascii=False))
with (ROOT / "data" / "hf_raw" / "synthetic.jsonl").open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"after 8x: {len(lines)}")