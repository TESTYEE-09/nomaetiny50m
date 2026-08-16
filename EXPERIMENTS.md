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
