# 2026-04-18 TurboQuant Studio v1 Foundation

## Scope

- implement the first local `TurboQuant Studio` workbench inside this repo
- keep the quantization engine unchanged and wrap existing scripts instead of replacing them
- preserve artifact roots and CLI compatibility
- add tests for the new API/job orchestration layer and the initial SPA shell

## Backend

Added `turboquant.studio_api` with:

- `app.py`: FastAPI app factory and `/api/...` routes
- `config.py`: local path settings for artifacts, jobs, DB, and built frontend assets
- `storage.py`: SQLite job metadata in `artifacts\studio\studio.db`
- `jobs.py`: single-worker local queue, log capture, process launch, and cancellation
- `artifacts.py`: artifact summary and tree indexing
- `setup.py`: environment snapshot for Python / uv / Node / CUDA / torch / repo contract
- `models.py`: typed request and response models for Studio workflows
- `tasks.py`: preview builders plus task wrappers over the existing scripts

The backend routes now cover:

- setup and run discovery
- artifact summary/tree browsing
- capture
- paper validate
- matrix validate
- runtime eval
- report export
- online report export
- GGUF packaging
- local llama.cpp serving
- local Hypura serving
- cancellation

## CLI preservation

The existing script entrypoints were kept, but they now expose `main(argv)` / `parse_args(argv)` style seams so Studio can call them as reusable task functions without shelling out for every Python-native workflow.

Touched entrypoints:

- `scripts\capture_qwen_kv.py`
- `scripts\validate_qwen_3060_matrix.py`
- `scripts\paper_validate_captured_qwen.py`
- `scripts\paper_validate_attention.py`
- `scripts\export_report.py`
- `scripts\export_online_eval_report.py`
- `scripts\pack_turboquant_gguf.py`
- `scripts\eval_runtime_qwen.py`

## Frontend

Added `studio-web\` as a React + TypeScript + Vite SPA with:

- top tab shell
- metadata strip
- right-side Tool Outputs drawer
- bottom Run History panel
- tabs for:
  - Setup
  - Capture
  - Offline Validate
  - Compare
  - Runtime Eval
  - Package & Export
  - Serve

The UI is intentionally local-operator focused rather than chat-first and keeps the repo research rules visible:

- Stage 1 / Stage 2 distinction remains explicit
- exact / estimated scoring is not merged into a single label
- reconstruction metrics remain separated from attention/logit metrics
- the production canonical K path stays `key_only_block_so8_triality_vector`

## Verification

```powershell
uv run pytest tests\test_studio_api.py -q
Set-Location .\studio-web
npm run test
npm run build
```

## Result

- Studio backend and SPA shell are now present in-repo
- dry-run orchestration works for the major workflows
- built frontend assets can be served by the FastAPI app
- README now includes the local Studio startup and build flow
