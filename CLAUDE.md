# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Paper-faithful TurboQuant prototype for KV-cache compression experiments on Qwen3.5-9B. Python package: `turboquant-qwen35`. Platform: **Windows + `uv` + Python 3.12.x** (strict — no other Python version, no bash scripts for portability-sensitive logic).

## Environment Setup

Run all commands from `hub_Qwen3.5-9B-SOT/` (the directory containing `pyproject.toml`). Use `uv run python ...` — never bare `py -3`.

```powershell
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev --extra hf_qwen
uv run python scripts\env_check.py
```

- `--extra cu128`: required for CUDA PyTorch (omitting gives CPU-only torch)
- `--extra hf_qwen`: required for `capture_qwen_kv.py` (transformers, accelerate, bitsandbytes)

## Commands

```powershell
# Run all tests
uv run python -m pytest -q

# Run a single test file
uv run python -m pytest tests/test_kv_codec.py -q

# Offline validation (no GPU / model required)
uv run python scripts\paper_validate_synthetic.py --trials 8
uv run python scripts\paper_validate_attention.py --trials 8 --synthetic-layers 4
uv run python scripts\research_validate_v_codecs.py --query-source synthetic --trials 3
uv run python scripts\research_value_sensitivity.py --trials 3 --synthetic-layers 4

# KV capture from local model weights (~18GB VRAM for bf16 full load)
uv run python scripts\capture_qwen_kv.py --weight-load none --dtype bfloat16 --trust-remote-code --model-id "H:\Qwen3.5-9B-official-hf" --output-dir artifacts\kv_full_bf16 --max-length 96

# Paper baseline on captured KV
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv_full_bf16

# Triality full pipeline (train then eval)
uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv_full_bf16 --train-output-dir artifacts\research_extension\triality_full_train_prod_bf16 --eval-output-dir artifacts\research_extension\triality_full_eval_prod_bf16

# Triality eval only (with resume for long runs)
uv run python scripts\research_validate_k_triality.py --resume --output-dir artifacts\research_extension\triality_full_eval_prod_bf16

# Production bundle (env_check + pytest)
.\scripts\run_production_tests.ps1
```

**Note**: Never run `research_train_k_triality.py` and `research_validate_k_triality.py` concurrently against the same `--output-dir`.

## Architecture

Three layers with a deliberate separation of concerns:

### 1. `turboquant.paper_baseline`
Paper-faithful quantizer. Two stages kept **strictly separate**:
- **Stage 1** (`TurboQuantMSE` / `PaperTurboQuantMSE`): MSE-optimized sphere Lloyd-Max codebook quantizer. Encodes to `QuantizedMSEBatch` (norms + indices + bitwidths).
- **Stage 2** (`TurboQuantProd` / `PaperTurboQuantProd`): Inner-product estimator adding a 1-bit QJL sketch (`QJLSketch`) for residual correction. Combined output: `QuantizedProdBatch`.

Three evaluation modes: `exact` (unquantized baseline), `key_only_random`, `full_kv`.

### 2. `turboquant.research_extension`
K/V research extensions:
- **Triality proxy** (`triality_proxy.py`): SO(8) triality views applied to rotation matrices (5 views: `identity`, `s`, `v`, `c`, `c_inv`). `TrialityProxyMSE` / `TrialityProxyProd` wrap Stage 1/2 with a chosen view.
- **K triality fitting** (`k_triality.py`): fits and evaluates triality proxy rotations per-layer; `--resume` reloads from disk. Rotation objects saved/loaded via `save_triality_proxy_rotations` / `load_triality_proxy_rotations`.
- **V codecs / sensitivity** (`evaluation.py`, `value_codecs.py`): `ProtectedValueCodec` with policies (`none`, `sensitivity_mixed`, `sensitive_layer_exact`, `protected_low_rank`).

### 3. `turboquant.adapters.hf_qwen`
Optional — requires `--extra hf_qwen`. Handles HF/Qwen model loading (bf16, 4bit, 8bit via BitsAndBytes) and per-layer KV capture to `.pt` files with a `capture_manifest.json`.

### Config families
- `turboquant_config.paper.json` — paper baseline and HF replay (`PAPER_SCHEMA_KIND`)
- `turboquant_config.research.json` — research extensions (`RESEARCH_SCHEMA_KIND`)

Schema helpers in `turboquant.schema`: `build_*_config`, `read_turboquant_config`, `write_turboquant_config`, `validate_*_config`.

### Artifacts
- `artifacts/paper_baseline/` — paper baseline validation results
- `artifacts/kv_full_bf16/<capture_id>/` — captured KV tensors (`layer_*_{key,value}.pt` + `capture_manifest.json`)
- `artifacts/research_extension/` — research eval outputs

## Code Conventions (from AGENTS.md)

- All public functions require type hints; nontrivial public APIs require docstrings.
- Add shape comments (`# [..., heads, head_dim]`) on tensor-heavy code.
- **Fail loudly** on invalid shapes or unsupported dtypes — no silent coercions.
- Never silently move tensors across devices or cast dtype in quantization paths.
- Keep math correspondence to the paper visible in comments.
- Every experiment artifact must record: model name, seed, dtype, device, quantization config, prompt hash, timestamp.
- Use `pathlib` for all paths; assume paths may contain spaces.

## Testing Policy

After changing quantization logic: run unit tests for touched modules + `paper_validate_synthetic.py` if estimator logic changed.

After changing capture or offline validation logic: run `env_check.py` + capture script + offline attention validation on a reduced case.
