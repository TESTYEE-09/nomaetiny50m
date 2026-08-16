# Experiments

## 2026-08-16 — Foundation inspection

- OS: Windows 11 Home 10.0.26200, 64-bit
- GPU: AMD Radeon RX 9070 XT, driver 32.0.31035.1003
- VRAM: **MEASURED** by DxDiag: 16,188 MB dedicated
- Python: 3.12.10, project-local virtual environment
- DeepSeek: non-thinking mode approved after LOW was found to map to HIGH; replacement key still needs to be supplied through the environment
- PyTorch: 2.9.1+rocm7.2.1; HIP 7.2.53211-158bd99533
- Model parameters: **MEASURED** 49,295,872
- GPU tensor operations: **MEASURED** pass (FP32)
- Mixed precision operations: **MEASURED** pass (FP16)
- Experimental training: **MEASURED** pass; 3 forward/backward/AdamW steps
- Smoke configuration: batch 1, sequence 128, FP16, 384 total tokens
- Throughput: **MEASURED** 168.12 tokens/sec (short warm-up-free smoke test, not a steady-state benchmark)
- Peak allocated VRAM: **MEASURED** 1,151,593,984 bytes (1.072 GiB)
- Final random-data loss: **MEASURED** 9.87285
- Note: success is experimental because AMD officially documents no Windows ML-training support.

## 2026-08-16 — Phase 0 end-to-end

- DeepSeek records: 1 (502 API tokens; 376 teacher output tokens)
- Mechanical processing: pass
- Tiny proof tokenizer: **MEASURED** 422 tokens (not the production tokenizer)
- Packed dataset: **MEASURED** 387 tokens at sequence length 128
- Training: **MEASURED** checkpoint saved after 3 RX 9070 XT steps
- Step-one throughput: **MEASURED** 81.83 tokens/sec
- Peak allocated VRAM: **MEASURED** 1,151,595,008 bytes
- Inference path: pass (empty random-weight output is expected at this scale)

## 2026-08-16 — Mixed DeepSeek + Luna one-epoch run

- DeepSeek records: **MEASURED** 4,465
- Luna records: **MEASURED** 172 from 10 lightweight agents
- Exact duplicates removed mechanically: **MEASURED** 4
- Mixed tokenizer: **MEASURED** 16,384 tokens
- Packed dataset: **MEASURED** 2,554,740 tokens
- Model: **MEASURED** 49,295,872 parameters, random initialization
- Training: **MEASURED** 1,247 steps; 2,553,856 tokens; sequence 512; gradient accumulation 4; FP16
- Sustained throughput: **MEASURED** approximately 19,400 tokens/sec late in the run
- Peak allocated VRAM: **MEASURED** 1,352,043,008 bytes
- Late-batch loss range: **MEASURED** approximately 2.05–3.58
- FP16 inference export: **MEASURED** 98,630,719 bytes

## 2026-08-16 — Ten-epoch retraining on mixed corpus

- Root cause of live-site gibberish: `train.bin` was packed with the pre-retraining tokenizer; retraining on it corrupted the model. Reverted to `checkpoints/step-1200.pt` (last clean checkpoint trained on `mixed.bin`).
- Training: **MEASURED** resumed `step-1200.pt` for 12,470 steps; 25,538,560 tokens; sequence 512; gradient accumulation 4; FP16
- Sustained throughput: **MEASURED** approximately 22,058 tokens/sec on RX 9070 XT
- Peak allocated VRAM: **MEASURED** approximately 1.5 GiB
- Final checkpoint: `checkpoints/mixed-production.pt` (step 12470)
- Held-out corpus loss: **MEASURED** approximately 0.7–1.0 (down from 2.2–4.8 at step 1200)
- Web GPU export: FP16 ONNX graph + 111 per-tensor shards, 115.4 MB total
- Web CPU export: **MEASURED** self-contained int8 ONNX, 83.4 MB (WASM fallback, much faster than FP16 on CPU)
- Site prompt format fixed to match training chat format (`<|eos|>` after each turn)
- ONNX vs PyTorch parity: **MEASURED** identical argmax, max abs diff 0.0527 (FP16)

## 2026-08-16 — Hugging Face data expansion + continued training

- New sources (Hugging Face): databricks-dolly-15k (14,996), yahma/alpaca-cleaned (51,756), openai/gsm8k (7,473)
- Merged corpus: **MEASURED** 78,862 records; 18,142,245 tokens (up from 2.55M)
- Training: **MEASURED** resumed `mixed-production.pt` for 13,284 steps; 52,744,192 total tokens; sequence 512; gradient accumulation 4; FP16; ~13.6 minutes
- Sustained throughput: **MEASURED** approximately 21,978 tokens/sec on RX 9070 XT
- Peak allocated VRAM: **MEASURED** 1,556,622,336 bytes (1.45 GiB)
- Final checkpoint: `checkpoints/mixed2-production.pt` (step 25754)
- ONNX vs PyTorch parity: **MEASURED** identical argmax, max abs diff 0.0234 (FP16); int8 identical argmax, top-5 overlap 5/5
