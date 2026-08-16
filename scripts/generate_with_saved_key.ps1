param(
    [int]$Count = 1000000,
    [int]$Seed = 20260816,
    [int]$MaxTokens = 4096,
    [double]$MaxSpend = 1.0,
    [int]$Concurrency = 32
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$generator = Join-Path $root 'generator\generate.py'
$secretFile = Join-Path $root '.secrets\deepseek-key.dpapi'
$output = Join-Path $root 'data\raw'
if (-not (Test-Path -LiteralPath $secretFile)) { throw "Encrypted key not found. Run scripts\save_deepseek_key.ps1 first." }
$encrypted = (Get-Content -Raw -LiteralPath $secretFile).Trim()
$secure = $encrypted | ConvertTo-SecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    & $python $generator --count $Count --seed $Seed --max-tokens $MaxTokens --max-spend $MaxSpend --concurrency $Concurrency --output $output
    if ($LASTEXITCODE -ne 0) { throw "Generator exited with code $LASTEXITCODE" }
} finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
}
