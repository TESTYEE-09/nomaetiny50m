"""Resumable DeepSeek-only teacher-data generator using explicit non-thinking mode."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, random, time
from pathlib import Path
import httpx

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
INPUT_USD_PER_MILLION = 0.14
OUTPUT_USD_PER_MILLION = 0.28
SYSTEM = """Generate one realistic training example for a tiny specialist assistant. All user and assistant text must be authored by you. The assistant is direct, plainspoken, concise by default, practical, and never corporate. Return JSON with keys category and messages; messages is an array of role/content objects. Include complete runnable code when appropriate. Do not include hidden reasoning."""
CATEGORIES = ["conversation"]*30 + ["web"]*20 + ["browser_game"]*15 + ["debugging"]*10 + ["code_edit"]*10 + ["python"]*5 + ["explanation"]*5 + ["instruction"]*5

def stable_id(seed: int, index: int) -> str:
    return hashlib.sha256(f"tiny50m:{seed}:{index}".encode()).hexdigest()[:24]

def prompt_for(seed: int, index: int) -> tuple[str, str]:
    rng = random.Random(f"{seed}:{index}")
    category = rng.choice(CATEGORIES)
    snake = category == "browser_game" and rng.random() < .55
    focus = "Snake with varied wording or a realistic follow-up modification" if snake else category
    return category, f"Category: {category}. Focus: {focus}. Variation seed: {seed}-{index}. Invent diverse natural user wording, sometimes messy or multi-turn."

def cost_usd(usage: dict) -> float:
    # Conservatively treats every input token as a cache miss.
    return usage.get("prompt_tokens",0)/1_000_000*INPUT_USD_PER_MILLION + usage.get("completion_tokens",0)/1_000_000*OUTPUT_USD_PER_MILLION

def prior_spend(out: Path) -> float:
    total=0.0
    for path in out.glob("*.json"):
        try: total += cost_usd(json.loads(path.read_text(encoding="utf-8")).get("usage",{}))
        except (OSError,json.JSONDecodeError): pass
    return total

async def call(client: httpx.AsyncClient, key: str, prompt: str, max_tokens: int, max_retries: int) -> dict:
    payload={"model":MODEL,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],"thinking":{"type":"disabled"},"response_format":{"type":"json_object"},"max_tokens":max_tokens}
    for attempt in range(max_retries):
        try:
            response=await client.post(API,headers={"Authorization":f"Bearer {key}"},json=payload)
            response.raise_for_status(); raw=response.json()
            choice=raw.get("choices",[{}])[0].get("message",{})
            content=choice.get("content","")
            if not content.strip(): raise ValueError("empty final response")
            if choice.get("reasoning_content"): raise RuntimeError("DeepSeek returned reasoning_content despite thinking=disabled; refusing to save")
            json.loads(content)  # Mechanical completeness check only; no quality judgment.
            return raw
        except (httpx.TimeoutException,httpx.NetworkError,httpx.HTTPStatusError,ValueError,json.JSONDecodeError) as exc:
            if attempt+1 >= max_retries: raise
            retry_after=response.headers.get("retry-after") if "response" in locals() else None
            delay=float(retry_after) if retry_after and retry_after.isdigit() else min(60,2**attempt+random.random())
            print(json.dumps({"event":"mechanical_retry","attempt":attempt+1,"delay_seconds":delay,"error_type":type(exc).__name__}))
            await asyncio.sleep(delay)

async def main_async(args):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key: raise SystemExit("DEEPSEEK_API_KEY is not set; no request was made.")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    spent=prior_spend(out)
    print(json.dumps({"event":"reasoning_contract","model":MODEL,"thinking":"disabled","reasoning_effort":"omitted","prior_estimated_spend_usd":spent,"max_spend_usd":args.max_spend}))
    if spent >= args.max_spend: raise SystemExit("Spending cap already reached; no request was made")
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(args.count):
            eid = stable_id(args.seed, i); path = out / f"{eid}.json"
            if path.exists(): continue
            if spent >= args.max_spend: print(json.dumps({"event":"spending_cap_reached","estimated_spend_usd":spent})); break
            category, prompt = prompt_for(args.seed, i)
            raw = await call(client, key, prompt, args.max_tokens, args.max_retries)
            message=raw["choices"][0]["message"]
            record={"id":eid,"category":category,"request":{"model":MODEL,"thinking":"disabled"},"teacher":json.loads(message["content"]),"usage":raw.get("usage",{}),"api_metadata":{"id":raw.get("id"),"created":raw.get("created"),"finish_reason":raw["choices"][0].get("finish_reason")}}
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
            spent += cost_usd(record["usage"])
            print(json.dumps({"event":"saved","id":eid,"category":category,"total_tokens":record["usage"].get("total_tokens"),"estimated_cumulative_spend_usd":spent}),flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=1); p.add_argument("--seed",type=int,default=20260816); p.add_argument("--max-tokens",type=int,default=4096); p.add_argument("--max-retries",type=int,default=6); p.add_argument("--max-spend",type=float,default=1.0); p.add_argument("--output",default="data/raw")
    asyncio.run(main_async(p.parse_args()))
