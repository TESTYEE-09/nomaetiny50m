import subprocess, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run(args, cwd=ROOT):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        log(f"FAIL {args}: {r.stderr[-400:]}")
        raise SystemExit(1)
    return r.stdout.strip()

def main():
    log("verifying wiki model...")
    run([PY, "scripts/verify_site_models.py", "website/model-wiki"])

    log("git add...")
    run(["git", "add", "-A"])
    run(["git", "status", "--short"])
    run(["git", "commit", "-m", "Add Wikipedia-trained model + model selector, train from scratch on 81.7M-token corpus"])
    run(["git", "push", "origin", "codex/end-to-end"])
    run(["git", "push", "origin", "codex/end-to-end:main"])
    log("pushed. waiting for pages run...")
    time.sleep(45)
    r = subprocess.run(["git", "log", "--oneline", "-1", "origin/main"], capture_output=True, text=True)
    log(f"origin/main: {r.stdout.strip()}")
    log("DONE")

if __name__ == "__main__":
    main()