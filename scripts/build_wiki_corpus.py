import json, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "wiki_raw"
OUT.mkdir(parents=True, exist_ok=True)

def digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def write(records, name):
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{name}: {len(records)} records -> {path.name}", flush=True)

def chat_records(items, source, category, user_field, assistant_field):
    records = []
    for item in items:
        u = item.get(user_field)
        a = item.get(assistant_field)
        if not u or not a: continue
        messages = [{"role": "user", "content": str(u)}, {"role": "assistant", "content": str(a)}]
        records.append({"id": f"hf-{source}-{digest(messages)[:16]}", "category": category, "messages": messages})
    return records

def code_records(items, source):
    records = []
    for item in items:
        prompt = item.get("prompt") or item.get("instruction")
        code = item.get("code") or item.get("canonical_solution") or item.get("solution") or item.get("output") or item.get("completion")
        if not prompt or not code: continue
        messages = [{"role": "user", "content": str(prompt).strip()}, {"role": "assistant", "content": str(code).strip()}]
        records.append({"id": f"hf-{source}-{digest(messages)[:16]}", "category": "coding", "messages": messages})
    return records

def main():
    from datasets import load_dataset

    done = []

    # --- Wikipedia: parquet-based sources with fallbacks ---
    wiki = []
    seen = set()
    print("downloading wikipedia slice...", flush=True)
    wiki_sources = [("wikimedia/wikipedia", "20231101.en")]
    for name, config in wiki_sources:
        for attempt in range(4):
            try:
                ds = load_dataset(name, name=config, split="train", streaming=True)
                for i, article in enumerate(ds):
                    title, text = article.get("title", ""), article.get("text", "")
                    if not text: continue
                    key = hashlib.sha256(text.encode()).hexdigest()
                    if key in seen: continue
                    seen.add(key)
                    messages = [{"role": "user", "content": f"Write a Wikipedia article about {title}."}, {"role": "assistant", "content": text[:4000]}]
                    wiki.append({"id": f"hf-wikipedia-{key[:16]}", "category": "general_qa", "messages": messages})
                    if len(wiki) >= 60000: break
                    if (i + 1) % 10000 == 0: print(f"  [{name}] streamed {i+1}, kept {len(wiki)}", flush=True)
                print(f"  [{name}] done, kept {len(wiki)}", flush=True)
                if wiki: break
            except Exception as e:
                print(f"  [{name}] attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                import time as _t; _t.sleep(5 * (attempt + 1))
    write(wiki, "wikipedia")
    done.append("wikipedia")

    # --- Light coding: CodeAlpaca_20K ---
    try:
        ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
        write(code_records(ds, "codealpaca"), "codealpaca")
        done.append("codealpaca")
    except Exception as e:
        print("codealpaca failed:", e, flush=True)

    # --- Chat: ultrachat slice ---
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        chat = []
        for i, item in enumerate(ds):
            msgs = item.get("messages", [])
            if len(msgs) < 2: continue
            pairs = []
            for m in msgs:
                if m.get("role") in ("user", "assistant"):
                    pairs.append({"role": m["role"], "content": str(m.get("content", ""))})
            if len(pairs) < 2: continue
            chat.append({"id": f"hf-ultrachat-{digest(pairs)[:16]}", "category": "conversation", "messages": pairs[:8]})
            if len(chat) >= 20000: break
        write(chat, "ultrachat")
        done.append("ultrachat")
    except Exception as e:
        print("ultrachat failed:", e, flush=True)

    # --- GSM8K (math) ---
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        recs = []
        for item in ds:
            messages = [{"role": "user", "content": item["question"]}, {"role": "assistant", "content": item["answer"]}]
            recs.append({"id": f"hf-gsm8k-{digest(messages)[:16]}", "category": "math", "messages": messages})
        write(recs, "gsm8k")
        done.append("gsm8k")
    except Exception as e:
        print("gsm8k failed:", e, flush=True)

    # --- Dolly 15k (QA/instruction) ---
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        recs = []
        for item in ds:
            context = str(item.get("context", "") or "")
            q = str(item.get("instruction", "") or "")
            a = str(item.get("response", "") or "")
            if not q or not a: continue
            full_q = f"{context}\n{q}" if context else q
            messages = [{"role": "user", "content": full_q}, {"role": "assistant", "content": a}]
            recs.append({"id": f"hf-dolly-{digest(messages)[:16]}", "category": item.get("category", "general_qa"), "messages": messages})
        write(recs, "dolly")
        done.append("dolly")
    except Exception as e:
        print("dolly failed:", e, flush=True)

    # --- Alpaca cleaned (instruction) ---
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        recs = []
        for item in ds:
            inst = str(item.get("instruction", "") or "")
            inp = str(item.get("input", "") or "")
            out = str(item.get("output", "") or "")
            if not inst or not out: continue
            full_q = f"{inst}\n{inp}" if inp else inst
            messages = [{"role": "user", "content": full_q}, {"role": "assistant", "content": out}]
            recs.append({"id": f"hf-alpaca-{digest(messages)[:16]}", "category": "instruction", "messages": messages})
        write(recs, "alpaca")
        done.append("alpaca")
    except Exception as e:
        print("alpaca failed:", e, flush=True)

    (ROOT / "data" / "wiki_done.json").write_text(json.dumps(done), encoding="utf-8")
    print("DONE:", done, flush=True)

if __name__ == "__main__":
    main()