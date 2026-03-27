# Turboquant-CUDA

PyTorch-first TurboQuant paper reproduction plus K/V-separated research
extensions, with Qwen3.5-9B captured replay kept as a secondary adapter layer.

This repository now has three explicit layers:

- `turboquant.paper_baseline`: paper-faithful PyTorch-only Stage 1 / Stage 2
- `turboquant.research_extension`: K/V-separated research codecs and ablations
- `turboquant.adapters.hf_qwen`: optional Hugging Face / Qwen capture + replay

The main implementation work is no longer defined by Hugging Face runtime
integration. The first-class goal is a PyTorch-only baseline that reproduces the
paper's quantization math before any runtime-specific integration.

Licensed under the MIT License. See `LICENSE`.

## Focus

- Paper-faithful `TurboQuantMSE` and `TurboQuantProd` in PyTorch
- Explicit norm handling for the unit-sphere baseline
- Research split between `K = score problem` and `V = state transport problem`
- Offline-first validation before runtime integration
- Qwen capture / replay retained as a secondary adapter validation path
- Reproducible CSV, Markdown, Matplotlib, and Plotly artifacts

## Quick Start

```powershell
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev
uv run python scripts\env_check.py
uv run python -m pytest -q
uv run python scripts\paper_validate_synthetic.py --trials 8
uv run python scripts\paper_validate_attention.py --trials 8 --synthetic-layers 4
uv run python scripts\research_validate_v_codecs.py --query-source synthetic --trials 3
uv run python scripts\research_value_sensitivity.py --trials 3 --synthetic-layers 4
```

To use the Qwen adapter path as well:

```powershell
uv sync --extra cu128 --extra dev --extra hf_qwen
uv run python scripts\capture_qwen_kv.py --weight-load 4bit --max-length 96
uv run python scripts\validate_attention_scores.py --query-source captured --kv-dir artifacts\kv --trials 1 --max-layers 1 --bits 2,2.5,3.5,4 --eval-device auto
```

## Main Entry Points

### Paper Baseline

- `scripts\paper_validate_synthetic.py`
- `scripts\paper_validate_attention.py`

### Research Extension

- `scripts\research_validate_v_codecs.py`
- `scripts\research_value_sensitivity.py`

### HF/Qwen Adapter

- `scripts\capture_qwen_kv.py`
- `scripts\validate_attention_scores.py`
- `scripts\export_report.py`

- `scripts\env_check.py`
- `scripts\benchmark_encode_decode.py`

## PyTorch-First Design

- `paper_baseline` is the canonical implementation target.
- `research_extension` is where K/V objective splitting lives.
- `adapters.hf_qwen` is intentionally optional and secondary.
- `bitsandbytes`, `transformers`, `llama.cpp`, and `Unsloth` are not part of the
  paper baseline acceptance criteria.

## Current Report

The current combined report still reflects the older integrated harness. The new
PyTorch-first split makes the interpretation explicit:

- `paper_baseline` answers whether Stage 1 / Stage 2 are reproduced faithfully.
- `research_extension` answers whether `K` and `V` should share one codec family.
- `adapters.hf_qwen` is used only after the PyTorch baseline is already stable.

The current mathematical bottleneck remains the value path: `full_kv` saves
more memory, but hidden-state drift is consistently larger than the stronger
key-only baselines.

### Paper Baseline Qwen3.5-9B Captured Replay

The paper-baseline-only Qwen replay now has its own extracted artifact set under
`artifacts/paper_baseline/qwen_captured_reported/`. This section uses only the
paper modes:

- `exact`
- `key_only_random`
- `full_kv`

Generated files:

- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured_mean_pm_sd.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured_mean_pm_sd.md`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.html`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_runtime_tradeoffs_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_runtime_tradeoffs_captured.html`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_mean_pm_sd_captured.png`

Representative captured values:

| Mode | Bits | Logit Cosine | Hidden Cosine | Memory / Exact |
| --- | ---: | ---: | ---: | ---: |
| key-only random | 2.0 | 0.997070 | 0.997070 | 0.566406 |
| key-only random | 4.0 | 0.998047 | 0.999023 | 0.628906 |
| full-KV | 2.0 | 0.997070 | 0.940430 | 0.130859 |
| full-KV | 4.0 | 0.998047 | 0.995117 | 0.255859 |

Mean +/- SD summary from the raw replay aggregation:

| Mode | Bits | Logit Cosine (mean +/- SD) | Hidden Cosine (mean +/- SD) | Memory Ratio (mean +/- SD) |
| --- | ---: | --- | --- | --- |
| key-only random | 2.0 | 0.995117 +/- 0.001953 | 0.997070 +/- 0.003740 | 0.566406 +/- 0.000000 |
| key-only random | 2.5 | 0.998047 +/- 0.002255 | 0.999023 +/- 0.001953 | 0.574219 +/- 0.000000 |
| key-only random | 3.0 | 1.000000 +/- 0.000000 | 0.998047 +/- 0.002255 | 0.597656 +/- 0.000000 |
| key-only random | 3.5 | 0.999023 +/- 0.001953 | 0.999023 +/- 0.001953 | 0.605469 +/- 0.000000 |
| key-only random | 4.0 | 0.998047 +/- 0.003906 | 0.999023 +/- 0.001953 | 0.628906 +/- 0.000000 |
| full-KV | 2.0 | 0.995117 +/- 0.001953 | 0.939453 +/- 0.006766 | 0.130859 +/- 0.000000 |
| full-KV | 2.5 | 0.998047 +/- 0.002255 | 0.957031 +/- 0.003189 | 0.146484 +/- 0.000000 |
| full-KV | 3.0 | 1.000000 +/- 0.000000 | 0.980469 +/- 0.005524 | 0.193359 +/- 0.000000 |
| full-KV | 3.5 | 0.999023 +/- 0.001953 | 0.988281 +/- 0.003189 | 0.208984 +/- 0.000000 |
| full-KV | 4.0 | 0.998047 +/- 0.003906 | 0.995117 +/- 0.001953 | 0.255859 +/- 0.000000 |

Note: the error bars in this section use replay-sample standard deviation across the captured prompt panel.
