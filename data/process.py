"""Mechanical-only processing: parse, deduplicate pipeline retries, format, and report."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer.chat import format_chat

def main():
    p=argparse.ArgumentParser(); p.add_argument("--raw",default="data/raw"); p.add_argument("--output",default="data/processed"); a=p.parse_args()
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    records=[]; seen=set(); categories=Counter(); duplicates=0
    for path in sorted(Path(a.raw).glob("*.json")):
        obj=json.loads(path.read_text(encoding="utf-8")); messages=obj["teacher"]["messages"]
        canonical=json.dumps(messages,ensure_ascii=False,sort_keys=True); digest=hashlib.sha256(canonical.encode()).hexdigest()
        if digest in seen: duplicates += 1; continue
        seen.add(digest); categories[obj["category"]] += 1
        records.append({"id":obj["id"],"category":obj["category"],"messages":messages,"text":format_chat(messages)})
    if not records: raise SystemExit("No teacher records found")
    with (output/"chat.jsonl").open("w",encoding="utf-8") as f:
        for record in records: f.write(json.dumps(record,ensure_ascii=False)+"\n")
    (output/"tokenizer_corpus.txt").write_text("\n".join(r["text"] for r in records),encoding="utf-8")
    stats={"records":len(records),"mechanical_duplicates_removed":duplicates,"categories":dict(categories),"characters":sum(len(r["text"]) for r in records)}
    Path("data/stats").mkdir(parents=True,exist_ok=True); Path("data/stats/processed.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats))
if __name__=="__main__": main()

