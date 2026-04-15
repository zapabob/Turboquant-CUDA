#Requires -Version 5.1
# Validate the repo contract, then build the Rust workspace against the expected llama.cpp checkout.

param(
    [string]$Package,
    [switch]$Release,
    [switch]$CheckOnly,
    [switch]$NoCuda,
    [string]$LlamaCppDir,
    [string]$TargetDir,
    [int]$Jobs = 1,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

$resolvedLlamaCppDir = $null
if ($LlamaCppDir) {
    $resolvedLlamaCppDir = (Resolve-Path -LiteralPath $LlamaCppDir).Path
}

$validateArgs = @("run", "python", "scripts/validate_repo_contract.py")
if ($resolvedLlamaCppDir) {
    $validateArgs += @("--llama-cpp-dir", $resolvedLlamaCppDir)
}

Write-Host "Running scripts\validate_repo_contract.py ..."
& uv @validateArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$previousLlamaCppDir = $env:LLAMA_CPP_DIR
$previousHypuraLlamaCppDir = $env:HYPURA_LLAMA_CPP_DIR
$previousNoCuda = $env:HYPURA_NO_CUDA
$previousCargoTargetDir = $env:CARGO_TARGET_DIR

try {
    if ($resolvedLlamaCppDir) {
        $env:LLAMA_CPP_DIR = $resolvedLlamaCppDir
        $env:HYPURA_LLAMA_CPP_DIR = $resolvedLlamaCppDir
    }

    if ($NoCuda) {
        $env:HYPURA_NO_CUDA = "1"
    }

    if ($TargetDir) {
        $resolvedTargetDir = [System.IO.Path]::GetFullPath($TargetDir)
        New-Item -ItemType Directory -Force -Path $resolvedTargetDir | Out-Null
        $env:CARGO_TARGET_DIR = $resolvedTargetDir
    }

    Push-Location -LiteralPath (Join-Path $RepoRoot "rust")
    try {
        $cargoCommand = if ($CheckOnly) { "check" } else { "build" }
        $buildArgs = @($cargoCommand)
        if ($Package) {
            $buildArgs += @("-p", $Package)
        }
        if ($Release) {
            $buildArgs += "--release"
        }
        if ($Jobs -gt 0) {
            $buildArgs += @("-j", $Jobs.ToString())
        }
        if ($CargoArgs) {
            $buildArgs += $CargoArgs
        }

        Write-Host ("Running cargo " + ($buildArgs -join " ") + " ...")
        & cargo @buildArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $previousLlamaCppDir) {
        Remove-Item Env:LLAMA_CPP_DIR -ErrorAction SilentlyContinue
    } else {
        $env:LLAMA_CPP_DIR = $previousLlamaCppDir
    }

    if ($null -eq $previousHypuraLlamaCppDir) {
        Remove-Item Env:HYPURA_LLAMA_CPP_DIR -ErrorAction SilentlyContinue
    } else {
        $env:HYPURA_LLAMA_CPP_DIR = $previousHypuraLlamaCppDir
    }

    if ($null -eq $previousNoCuda) {
        Remove-Item Env:HYPURA_NO_CUDA -ErrorAction SilentlyContinue
    } else {
        $env:HYPURA_NO_CUDA = $previousNoCuda
    }

    if ($null -eq $previousCargoTargetDir) {
        Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
    } else {
        $env:CARGO_TARGET_DIR = $previousCargoTargetDir
    }
}
