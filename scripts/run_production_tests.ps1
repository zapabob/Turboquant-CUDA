#Requires -Version 5.1
# Run env_check + pytest; skip uv sync when CUDA torch already OK in uv venv.

param(
    [switch]$ForceSync,
    [ValidateSet("cu128", "cpu")]
    [string]$TorchExtra = "cu128"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

function Test-CudaTorchReadyViaUv {
    if (-not (Test-Path (Join-Path $RepoRoot ".venv"))) {
        return $false
    }
    $snippet = 'import sys,torch; from turboquant.runtime import cuda_matches_target; sys.exit(0 if cuda_matches_target(torch) else 1)'
    $null = & uv run python -c $snippet 2>&1
    return ($LASTEXITCODE -eq 0)
}

$skipSync = $false
if (-not $ForceSync) {
    if ($TorchExtra -eq "cu128" -and (Test-CudaTorchReadyViaUv)) {
        Write-Host "Torch CUDA already OK in uv environment; skipping uv sync."
        $skipSync = $true
    }
}

if (-not $skipSync) {
    Write-Host "Running uv sync --extra $TorchExtra --extra dev ..."
    & uv sync --extra $TorchExtra --extra dev
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Running scripts\env_check.py ..."
& uv run python scripts\env_check.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running pytest ..."
& uv run python -m pytest -q
exit $LASTEXITCODE