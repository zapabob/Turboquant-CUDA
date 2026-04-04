# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Paper-faithful TurboQuant prototype for KV-cache compression experiments on Qwen3.5-9B. Python package: `turboquant-qwen35`. Platform: **Windows + `uv` + Python 3.12.x** (strict — no other Python version, no bash scripts for portability-sensitive logic).

**Primary dev GPU (this repo): RTX 3060 12GB.** Prefer **4-bit or 8-bit weight load** and a **conservative `--max-length`** for KV capture; bf16 full-model load is documented as **~24GB-class** VRAM, not the default path on 12GB.

### Production canonical K-side TurboQuant (実用正系)

For **practical / shipping-style** key quantization on captured (or runtime) KV, treat **Triality SO(8) proxy + TurboQuant** as the reference path:

- **Mode name:** `key_only_block_so8_triality_vector` (`PRODUCTION_K_TURBOQUANT_MODE` in `turboquant.research_extension.k_triality`).
- **View:** `vector` (`PRODUCTION_K_TURBOQUANT_VIEW`) — per-layer fitted rotations from `fit_triality_proxy_rotations` / `research_train_k_triality.py`.
- **Default rotation artifact dir:** `artifacts/research_extension/triality_full_train/rotations` (`DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR`).
- **Scripts:** `scripts/research_validate_multiscreen_kv.py` defaults `--mode` to this triality vector mode (ablation modes remain available explicitly).

Paper modes (`key_only_random`, static SO8 without triality, `full_kv`, etc.) stay **reproducibility and ablation** baselines, not the production K default.

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

# KV capture — 12GB default path (4-bit weights; tune --max-length if OOM)
uv run python scripts\capture_qwen_kv.py --weight-load 4bit --dtype bfloat16 --trust-remote-code --model-id "H:\Qwen3.5-9B-official-hf" --output-dir artifacts\kv_4bit --max-length 64

# Alternative 12GB path: 8-bit weights (often more VRAM than 4-bit)
# uv run python scripts\capture_qwen_kv.py --weight-load 8bit --dtype bfloat16 --trust-remote-code --model-id "H:\Qwen3.5-9B-official-hf" --output-dir artifacts\kv_8bit --max-length 64

# bf16 / full weight load (~24GB-class VRAM; not typical on 12GB)
# uv run python scripts\capture_qwen_kv.py --weight-load none --dtype bfloat16 --trust-remote-code --model-id "H:\Qwen3.5-9B-official-hf" --output-dir artifacts\kv_full_bf16 --max-length 96

# Paper baseline on captured KV
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv_4bit

# Production K path (Triality SO8 + TurboQuant vector): after training rotations, same capture root
# uv run python scripts\research_validate_multiscreen_kv.py --captured-dir artifacts\kv_4bit --eval-device cuda

# Ablation: Multiscreen relevance + mixed-bit K eval
uv run python scripts\research_validate_multiscreen_kv.py --captured-dir artifacts\kv_4bit --mode multiscreen_relevance --bits 3 --max-layers 4

# Triality full pipeline (train then eval); use the same --kv-dir you captured with
uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv_4bit --train-output-dir artifacts\research_extension\triality_full_train --eval-output-dir artifacts\research_extension\triality_full_eval

# Triality eval only (with resume for long runs)
uv run python scripts\research_validate_k_triality.py --resume --output-dir artifacts\research_extension\triality_full_eval

# VRAM / KV footprint multi-group study (PyTorch TurboQuant + Multiscreen on CUDA; needs captured KV first)
# After capture from H:\Qwen3.5-9B-official-hf, compare modes with SEM error bars (writes CSV/MD + 3 PNGs):
# uv run python scripts\research_vram_multigroup_qwen.py --kv-dir artifacts\kv_4bit --model-id "H:\Qwen3.5-9B-official-hf" --eval-device cuda --trials 8
# Learned rotation + Multiscreen mixed bits (add to --modes; --bits must match trained artifact, e.g. 3.0):
# uv run python scripts\research_vram_multigroup_qwen.py --kv-dir artifacts\kv_4bit --eval-device cuda --trials 8 --modes exact,multiscreen_relevance,multiscreen_triality_vector --rotation-dir artifacts\research_extension\triality_full_train\rotations
# Vector-only Triality (no multiscreen): add key_only_block_so8_triality_vector and same --rotation-dir if not default.

# Rust workspace (Hypura + llama.cpp FFI): semantic versions hypura 0.4.0 / hypura-sys 0.3.0 / kobold_gguf_gui 0.2.0; incremental in rust/.cargo/config.toml
# Mirror copy: `C:\Users\downl\Desktop\hypura-main\hypura-main` (same workspace + `vendor/llama.cpp`). Override llama path: `LLAMA_CPP_DIR` or `HYPURA_LLAMA_CPP_DIR`.
# cd rust; $env:HYPURA_NO_CUDA=1; cargo build -p hypura   # CPU-only check without CUDA toolkit

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
- **K triality fitting** (`k_triality.py`): fits and evaluates triality proxy rotations per-layer; `--resume` reloads from disk. Rotation objects saved/loaded via `save_triality_proxy_rotations` / `load_triality_proxy_rotations`. **Production canonical** K eval mode: `PRODUCTION_K_TURBOQUANT_MODE` / `PRODUCTION_K_TURBOQUANT_VIEW` (vector + `key_only_block_so8_triality_vector`).
- **V codecs / sensitivity** (`evaluation.py`, `value_codecs.py`): `ProtectedValueCodec` with policies (`none`, `sensitivity_mixed`, `sensitive_layer_exact`, `protected_low_rank`).

### 3. `turboquant.adapters.hf_qwen`
Optional — requires `--extra hf_qwen`. Handles HF/Qwen model loading (bf16, 4bit, 8bit via BitsAndBytes) and per-layer KV capture to `.pt` files with a `capture_manifest.json`.

### Config families
- `turboquant_config.paper.json` — paper baseline and HF replay (`PAPER_SCHEMA_KIND`)
- `turboquant_config.research.json` — research extensions (`RESEARCH_SCHEMA_KIND`)

Schema helpers in `turboquant.schema`: `build_*_config`, `read_turboquant_config`, `write_turboquant_config`, `validate_*_config`.

### Artifacts
- `artifacts/paper_baseline/` — paper baseline validation results
- `artifacts/kv_4bit/` (or your capture root) — captured KV tensors (`layer_*_{key,value}.pt` + `capture_manifest.json`)
- `artifacts/research_extension/` — research eval outputs (incl. `multiscreen_kv/`, `vram_multigroup/` from VRAM comparison script)
- `rust/hypura`, `rust/hypura-sys`, `rust/kobold_gguf_gui` — Hypura + Kobold互換GUI（`vendor/llama.cpp` または `LLAMA_CPP_DIR`）。GGUF切替はGUIの Recent + `%APPDATA%\hypura\kobold_gguf_gui_settings.json`

### References (external)
- [Screening Is Enough](https://www.alphaxiv.org/abs/2604.01178) — Multiscreen / screening background (repo stays dependency-free; logic in `turboquant.research_extension.multiscreen_kv`).
- Upstream `zapabob/multiscreen-pytorch` — optional parity tests only (`multiscreen_parity` extra in `pyproject.toml`).

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
