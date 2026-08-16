import json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw_hf"
DST = ROOT / "data" / "hf_raw"
DST.mkdir(parents=True, exist_ok=True)


def record(messages, category, source):
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:24]
    return {"id": f"hf-{source}-{digest}", "category": category, "messages": messages}


def main():
    out = []
    # dolly
    for line in (SRC / "dolly.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        user = o["instruction"] + (f"\n\nContext: {o['context']}" if o.get("context") else "")
        out.append(record([{"role": "user", "content": user}, {"role": "assistant", "content": o["response"]}], o.get("category", "knowledge"), "dolly"))
    # alpaca
    for o in json.loads((SRC / "alpaca.json").read_text(encoding="utf-8")):
        user = o["instruction"] + (f"\n\n{o['input']}" if o.get("input") else "")
        out.append(record([{"role": "user", "content": user}, {"role": "assistant", "content": o["output"]}], "instruction", "alpaca"))
    # gsm8k
    import pyarrow.parquet as pq
    table = pq.read_table(str(SRC / "gsm8k-train.parquet"))
    for question, answer in zip(table["question"].to_pylist(), table["answer"].to_pylist()):
        out.append(record([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], "math", "gsm8k"))
    with (DST / "hf.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"records: {len(out)}  dolly/alpaca/gsm8k: {len(out) and 'ok'}")


if __name__ == "__main__":
    main()