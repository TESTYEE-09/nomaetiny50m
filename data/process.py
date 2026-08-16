"""Mechanical-only processing: parse, deduplicate pipeline retries, format, and report."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer.chat import format_chat

def normalize_messages(messages):
    """Parse both documented role/content and teacher-emitted {role: text} shapes."""
    normalized=[]
    for message in messages:
        if "role" in message and "content" in message:
            normalized.append({"role":message["role"],"content":str(message["content"])})
            continue
        if "role" in message and "text" in message:
            normalized.append({"role":message["role"],"content":str(message["text"])})
            continue
        role_keys=[role for role in ("system","user","assistant") if role in message]
        if len(role_keys)!=1: raise ValueError(f"mechanically unparseable message keys: {sorted(message)}")
        role=role_keys[0]; normalized.append({"role":role,"content":str(message[role])})
    return normalized

def main():
    p=argparse.ArgumentParser(); p.add_argument("--raw",default="data/raw"); p.add_argument("--luna",default="data/luna_raw"); p.add_argument("--hf",default="data/hf_raw"); p.add_argument("--output",default="data/processed"); a=p.parse_args()
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    records=[]; seen=set(); categories=Counter(); sources=Counter(); duplicates=0
    for path in sorted(Path(a.raw).glob("*.json")):
        obj=json.loads(path.read_text(encoding="utf-8")); messages=normalize_messages(obj["teacher"]["messages"])
        canonical=json.dumps(messages,ensure_ascii=False,sort_keys=True); digest=hashlib.sha256(canonical.encode()).hexdigest()
        if digest in seen: duplicates += 1; continue
        seen.add(digest); categories[obj["category"]] += 1
        sources["deepseek-v4-flash"] += 1
        records.append({"id":obj["id"],"source":"deepseek-v4-flash","category":obj["category"],"messages":messages,"text":format_chat(messages)})
    for shard in sorted(Path(a.luna).glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            obj=json.loads(line); messages=normalize_messages(obj["messages"])
            canonical=json.dumps(messages,ensure_ascii=False,sort_keys=True); digest=hashlib.sha256(canonical.encode()).hexdigest()
            if digest in seen: duplicates += 1; continue
            seen.add(digest); categories[obj["category"]] += 1; sources[obj.get("source","gpt-5.6-luna")] += 1
            records.append({"id":obj["id"],"source":obj.get("source","gpt-5.6-luna"),"category":obj["category"],"messages":messages,"text":format_chat(messages)})
    for shard in sorted(Path(a.hf).glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").split("\n"):
            if not line.strip(): continue
            obj=json.loads(line); messages=normalize_messages(obj["messages"])
            canonical=json.dumps(messages,ensure_ascii=False,sort_keys=True); digest=hashlib.sha256(canonical.encode()).hexdigest()
            if digest in seen: duplicates += 1; continue
            seen.add(digest); categories[obj["category"]] += 1; sources[obj["id"].split("-")[1]] += 1
            records.append({"id":obj["id"],"source":obj["id"].split("-")[1],"category":obj["category"],"messages":messages,"text":format_chat(messages)})
    if not records: raise SystemExit("No teacher records found")
    with (output/"chat.jsonl").open("w",encoding="utf-8") as f:
        for record in records: f.write(json.dumps(record,ensure_ascii=False)+"\n")
    (output/"tokenizer_corpus.txt").write_text("\n".join(r["text"] for r in records),encoding="utf-8")
    stats={"records":len(records),"mechanical_duplicates_removed":duplicates,"sources":dict(sources),"categories":dict(categories),"characters":sum(len(r["text"]) for r in records)}
    Path("data/stats").mkdir(parents=True,exist_ok=True); Path("data/stats/processed.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats))
if __name__=="__main__": main()
