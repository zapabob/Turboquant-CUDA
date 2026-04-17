param(
    [string]$PythonVersion = "3.12.9",
    [ValidateSet("cu128", "cpu")]
    [string]$TorchExtra = "cu128",
    [switch]$SkipSyncIfCudaReady,
    [switch]$SkipHfQwen
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Uv {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & uv @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "uv $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
        }
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Test-CudaTorchReadyViaUv {
    if (-not (Test-Path (Join-Path $repoRoot ".venv"))) {
        return $false
    }
    $snippet = 'import sys,torch; from turboquant.runtime import cuda_matches_target; sys.exit(0 if cuda_matches_target(torch) else 1)'
    Push-Location $repoRoot
    try {
        Invoke-Uv @("run", "python", "-c", $snippet)
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        Pop-Location
    }
}

Push-Location $repoRoot
try {
    Write-Host "Installing Python $PythonVersion via uv..."
    Invoke-Uv @("python", "install", $PythonVersion)

    Write-Host "Creating project virtual environment..."
    Invoke-Uv @("venv", "--python", $PythonVersion)

    $doSync = $true
    if ($SkipSyncIfCudaReady -and $TorchExtra -eq "cu128" -and (Test-CudaTorchReadyViaUv)) {
        Write-Host "Torch CUDA already OK in uv environment; skipping uv sync."
        $doSync = $false
    }

    if ($doSync) {
        $syncArgs = @("sync", "--extra", $TorchExtra, "--extra", "dev")
        if (-not $SkipHfQwen) {
            $syncArgs += @("--extra", "hf_qwen")
        }
        Write-Host ("Syncing dependencies: uv " + ($syncArgs -join " "))
        Invoke-Uv $syncArgs
    }

    Write-Host ""
    Write-Host "Bootstrap complete."
    Write-Host "Use 'uv run ...' for all project commands."
} finally {
    Pop-Location
}
