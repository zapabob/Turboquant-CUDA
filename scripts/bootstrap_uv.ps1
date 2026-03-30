param(
    [string]$PythonVersion = "3.12.9",
    [ValidateSet("cu128", "cpu")]
    [string]$TorchExtra = "cu128",
    [switch]$SkipSyncIfCudaReady
)

$ErrorActionPreference = "Stop"

function Test-CudaTorchReadyViaUv {
    if (-not (Test-Path (Join-Path (Get-Location) ".venv"))) {
        return $false
    }
    $snippet = 'import sys,torch; from turboquant.runtime import cuda_matches_target; sys.exit(0 if cuda_matches_target(torch) else 1)'
    $null = & uv run python -c $snippet 2>&1
    return ($LASTEXITCODE -eq 0)
}

Write-Host "Installing Python $PythonVersion via uv..."
uv python install $PythonVersion

Write-Host "Creating project virtual environment..."
uv venv --python $PythonVersion

$doSync = $true
if ($SkipSyncIfCudaReady -and $TorchExtra -eq "cu128" -and (Test-CudaTorchReadyViaUv)) {
    Write-Host "Torch CUDA already OK in uv environment; skipping uv sync."
    $doSync = $false
}

if ($doSync) {
    Write-Host "Syncing dependencies for extra '$TorchExtra' plus dev..."
    uv sync --extra $TorchExtra --extra dev
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Use 'uv run ...' for all project commands."