$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$secretDir = Join-Path $root '.secrets'
$secretFile = Join-Path $secretDir 'deepseek-key.dpapi'
New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
$secure = Read-Host 'Replacement DeepSeek API key' -AsSecureString
$secure | ConvertFrom-SecureString | Set-Content -LiteralPath $secretFile -Encoding ascii
Write-Output "Encrypted key saved for the current Windows account. It is ignored by Git: $secretFile"

