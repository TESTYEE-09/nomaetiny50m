$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$py=Join-Path $root '.venv\Scripts\python.exe'
Push-Location $root
try {
  & $py data\process.py
  & $py tokenizer\train_tokenizer.py --input data\processed\tokenizer_corpus.txt --output tokenizer\tokenizer.json
  & $py data\pack.py --seq 128 --output data\processed\phase0.bin
  & $py training\train.py --data data\processed\phase0.bin --seq 128 --grad-accum 1 --steps 3 --save-every 3 --output checkpoints\phase0.pt
  & $py training\infer.py 'hello' --checkpoint checkpoints\phase0.pt --max-new 20
} finally { Pop-Location }

