# 2026-04-19 uv CUDA 12.8 torch environment

## Overview

- Provisioned a working `uv` virtual environment with CUDA 12.8 PyTorch on
  `C:` because the workspace drive `H:` did not have enough free space for a
  local in-repo CUDA PyTorch install.
- Confirmed `torch`, `torch.nn`, and CUDA visibility work in the new
  environment.
- Re-ran `scripts\validate_repo_contract.py` successfully with that Python.

## Background / requirement

- The existing local `.venv` and `.venv-cu128` states exposed only a broken
  namespace-style `torch` package with no `torch.nn`.
- A direct `uv sync --extra cu128 --extra dev` against the project environment
  failed while copying `torch_cpu.dll` with Windows error 112 (`not enough disk
  space`) on `H:`.

## Decision

- Use `uv` with an active virtual environment located on `C:`:
  `C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv`
- Keep the project metadata (`pyproject.toml`) unchanged because it already
  points `torch` extra `cu128` at the correct PyTorch CUDA 12.8 index.

## Commands run

```powershell
uv venv --python 3.12.9 C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv

& C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv\Scripts\Activate.ps1
uv sync --active --extra cu128 --extra dev --link-mode copy

C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv\Scripts\python.exe - <<'PY'
import torch
import torch.nn as nn
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(nn.Module.__name__)
PY

C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv\Scripts\python.exe scripts\validate_repo_contract.py
```

## Verification results

- Python: `C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv\Scripts\python.exe`
- `torch.__version__`: `2.11.0+cu128`
- `torch.version.cuda`: `12.8`
- `torch.cuda.is_available()`: `True`
- `import torch.nn as nn`: succeeded
- `scripts\validate_repo_contract.py`: `Repository contract OK.`

## Residual risks

- The working CUDA PyTorch environment is not the default in-repo `.venv`; it is
  currently the external `C:`-hosted environment above.
- If future commands use plain `uv sync` without `--active` or without
  activating this environment first, `uv` may target the project-default `.venv`
  again.

## Recommended next actions

1. Activate the `C:` environment before PyTorch/CUDA work:
   `& C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv\Scripts\Activate.ps1`
2. Use `uv sync --active --extra cu128 --extra dev --link-mode copy` when
   refreshing that environment.
3. If a repo-local `.venv` is required later, free enough space on `H:` first.
