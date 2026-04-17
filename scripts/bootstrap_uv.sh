#!/usr/bin/env bash
set -euo pipefail

python_version="3.12.9"
torch_extra="cu128"
skip_sync_if_cuda_ready=0
skip_hf_qwen=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version)
      python_version="$2"
      shift 2
      ;;
    --torch-extra)
      torch_extra="$2"
      shift 2
      ;;
    --skip-sync-if-cuda-ready)
      skip_sync_if_cuda_ready=1
      shift
      ;;
    --skip-hf-qwen)
      skip_hf_qwen=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$torch_extra" != "cu128" && "$torch_extra" != "cpu" ]]; then
  echo "Unsupported --torch-extra '$torch_extra' (expected cu128 or cpu)" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test_cuda_torch_ready_via_uv() {
  if [[ ! -d .venv ]]; then
    return 1
  fi
  local snippet
  snippet='import sys,torch; from turboquant.runtime import cuda_matches_target; sys.exit(0 if cuda_matches_target(torch) else 1)'
  uv run python -c "$snippet" >/dev/null 2>&1
}

echo "Installing Python $python_version via uv..."
uv python install "$python_version"

echo "Creating project virtual environment..."
uv venv --python "$python_version"

do_sync=1
if [[ "$skip_sync_if_cuda_ready" -eq 1 && "$torch_extra" == "cu128" ]] && test_cuda_torch_ready_via_uv; then
  echo "Torch CUDA already OK in uv environment; skipping uv sync."
  do_sync=0
fi

if [[ "$do_sync" -eq 1 ]]; then
  sync_args=(sync --extra "$torch_extra" --extra dev)
  if [[ "$skip_hf_qwen" -eq 0 ]]; then
    sync_args+=(--extra hf_qwen)
  fi
  echo "Syncing dependencies: uv ${sync_args[*]}"
  uv "${sync_args[@]}"
fi

echo
echo "Bootstrap complete."
echo "Use 'uv run ...' for all project commands."
