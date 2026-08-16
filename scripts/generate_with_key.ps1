param(
    [int]$Count = 1,
    [int]$Seed = 20260816,
    [int]$MaxTokens = 4096,
    [double]$MaxSpend = 1.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$generator = Join-Path $projectRoot 'generator\generate.py'
$output = Join-Path $projectRoot 'data\raw'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found: $python"
}

$secureKey = Read-Host 'Replacement DeepSeek API key' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    & $python $generator --count $Count --seed $Seed --max-tokens $MaxTokens --max-spend $MaxSpend --output $output
    if ($LASTEXITCODE -ne 0) { throw "Generator exited with code $LASTEXITCODE" }
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
}
