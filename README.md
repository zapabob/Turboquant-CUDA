# TurboQuant on Qwen3.5-9B

Research-grade TurboQuant prototype for KV-cache compression experiments on
Qwen3.5-9B in a Windows-native Codex workflow.

## Focus

- Paper-faithful separation between `TurboQuantMSE` and `TurboQuantProd`
- Offline validation before online Hugging Face integration
- Windows and PowerShell friendly tooling
- Reproducible experiment artifacts with explicit metadata

## Quick Start

```powershell
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev
uv run python scripts\env_check.py
uv run python -m pytest -q
```

## Main Entry Points

- `scripts\env_check.py`
- `scripts\test_synthetic.py`
- `scripts\capture_qwen_kv.py`
- `scripts\validate_attention_scores.py`
- `scripts\benchmark_encode_decode.py`

## Qwen Smoke Examples

```powershell
uv run python scripts\capture_qwen_kv.py --model-id Qwen/Qwen3.5-9B --weight-load 4bit --max-length 96
uv run python scripts\capture_qwen_kv.py --model-id Qwen/Qwen3.5-9B-Base --weight-load 4bit --max-length 96
uv run python scripts\validate_attention_scores.py --model-id Qwen/Qwen3.5-9B --query-source synthetic
```
