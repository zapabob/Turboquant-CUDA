$ErrorActionPreference = "Stop"

param(
    [string]$PythonVersion = "3.12.9",
    [ValidateSet("cu128", "cpu")]
    [string]$TorchExtra = "cu128"
)

Write-Host "Installing Python $PythonVersion via uv..."
uv python install $PythonVersion

Write-Host "Creating project virtual environment..."
uv venv --python $PythonVersion

Write-Host "Syncing dependencies for extra '$TorchExtra' plus dev..."
uv sync --extra $TorchExtra --extra dev

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Use 'uv run ...' for all project commands."
