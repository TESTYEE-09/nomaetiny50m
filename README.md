# Tiny50M

A native-Windows, random-initialized decoder-only language-model project targeting conversation and simple coding, especially small browser games and Snake.

> Current artifact status: Phase 0 pipeline proof only. The included checkpoint was trained for 12,800 tokens on 14 DeepSeek records and does not yet produce useful language. It must not be presented as the intended finished specialist model.

## Architecture

16,384-token byte-level BPE; 512 hidden; 1,792 SwiGLU intermediate; 12 layers; 8 query heads; 2 KV heads; RoPE; RMSNorm; tied embeddings; 2,048 context. Run `python training/model.py` for the exact parameter count.

## Data contract

Every training character, including tokenizer training text, must be authored by `deepseek-v4-flash`. Because DeepSeek currently maps LOW to HIGH, the approved fallback is explicit non-thinking mode: `thinking.type=disabled`, with `reasoning_effort` omitted. The generator rejects unexpected reasoning content and never quality-filters or rewrites teacher output.

## Native Windows AMD status

AMD ROCm/PyTorch 7.2.1 supports RX 9070 XT on Windows 11 for PyTorch inference, using Python 3.12 and PyTorch 2.9.1 wheels. AMD's current Windows limitations say ML training is unsupported, so any native-Windows training success is experimental, not officially supported.

## Included Phase-0 weights

`export/model-fp16.pt` contains the tied-weight 49,295,872-parameter model in FP16. It is a 98.6 MB inference export, trained for only 12,800 tokens. The resumable optimizer checkpoint remains local because it is roughly 592 MB. `website/index.html` is currently a WebGPU capability/UI shell; browser model execution is not yet wired up.

## Commands

Activate with `.venv\Scripts\Activate.ps1`. Install generic dependencies with `pip install -r requirements.txt`, then install AMD's ROCm 7.2.1 SDK and PyTorch wheels from its official Windows guide. Generate safely with `powershell -ExecutionPolicy Bypass -File scripts\generate_with_key.ps1 -Count 1`; the masked replacement key exists only in that child process and is cleared afterward. Generation resumes using stable IDs. Train the tokenizer only after teacher data exists. Check model size with `python training\model.py`. Run the strict GPU smoke test with `python training\smoke_train.py`; it refuses CPU fallback. Serve `website` with `python -m http.server 8000 -d website`.

## Chat template

`<|bos|><|system|>...<|user|>...<|assistant|>...<|eos|>` is the single template for SFT, evaluation, inference, and export.
