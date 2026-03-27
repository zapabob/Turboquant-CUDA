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

This repo now treats TurboQuant config as dual-schema output:

- `turboquant_config.paper.json`: paper-faithful HF/Qwen baseline
- `turboquant_config.research.json`: K/V-separated research and future Hypura/GGUF sidecar

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
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv --bits 2,2.5,3,3.5,4 --write-config
uv run python scripts\research_validate_v_codecs.py --query-source captured --kv-dir artifacts\kv --trials 1 --max-layers 1 --bits 2,2.5,3.5,4 --write-config
```

## Main Entry Points

### Paper Baseline

- `scripts\paper_validate_synthetic.py`
- `scripts\paper_validate_attention.py`
- `scripts\paper_validate_captured_qwen.py`

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

Mixed-bit points `2.5` and `3.5` follow the paper policy:

- `2.5 bit = 32 channels @ 3 bit + 96 channels @ 2 bit`
- `3.5 bit = 32 channels @ 4 bit + 96 channels @ 3 bit`

Generated files:

- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured_mean_pm_sd.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured_mean_pm_sd.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_memory_by_bit_mean_pm_sd.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_memory_by_bit_mean_pm_sd.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_omnibus.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_omnibus.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_pairwise.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_pairwise.md`
- `artifacts/paper_baseline/qwen_captured_reported/turboquant_config.paper.json`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.html`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_runtime_tradeoffs_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_runtime_tradeoffs_captured.html`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_mean_pm_sd_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_v_breakage_by_bit_sd.png`

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

KV cache / memory usage by bit:

| Bits | key-only random memory ratio | full-KV memory ratio | full-KV memory bits (mean +/- SD) |
| --- | --- | --- | --- |
| 2.0 | 0.566406 +/- 0.000000 | 0.130859 +/- 0.000000 | 78256.0 +/- 22246.7 |
| 2.5 | 0.574219 +/- 0.000000 | 0.146484 +/- 0.000000 | 87600.0 +/- 24903.0 |
| 3.0 | 0.597656 +/- 0.000000 | 0.193359 +/- 0.000000 | 115632.0 +/- 32872.0 |
| 3.5 | 0.605469 +/- 0.000000 | 0.208984 +/- 0.000000 | 124976.0 +/- 35528.3 |
| 4.0 | 0.628906 +/- 0.000000 | 0.255859 +/- 0.000000 | 153008.0 +/- 43497.3 |

The practical trade-off is consistent across all bits: paper-faithful `full_kv`
shrinks the KV cache much more aggressively than `key_only_random`, but that
extra V compression shows up mainly in hidden-state and attention-output
transport rather than in logit cosine.

Error-bar figures:

![Paper baseline trade-offs](artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.png)

![Paper baseline mean +/- SD](artifacts/paper_baseline/qwen_captured_reported/plots/attention_mean_pm_sd_captured.png)

![V breakage by bit](artifacts/paper_baseline/qwen_captured_reported/plots/attention_v_breakage_by_bit_sd.png)

### Statistical Treatment

For the statistical summary, the README uses the raw replay samples in
`attention_trials_captured.csv` with `n = 4` per `(mode, bit)` group. Following
the requested framing, we interpret `full_kv` as the "V is broken" condition,
but the reported p-values are still computed against the conventional
no-difference null.

Omnibus multi-group comparison across the 10 quantized groups
(`key_only_random/full_kv × {2, 2.5, 3, 3.5, 4}`):

| Metric | Test | Statistic | p-value |
| --- | --- | ---: | ---: |
| hidden cosine similarity | Kruskal-Wallis | 33.8627 | 9.44e-05 |
| attention output relative error | Kruskal-Wallis | 37.5473 | 2.10e-05 |
| logit cosine similarity | Kruskal-Wallis | 17.2946 | 4.43e-02 |

This gives a clean separation between the "V-sensitive" metrics and the score
metric: hidden cosine and attention-output error vary strongly across groups,
while logit cosine moves much less.

Bit-wise one-sided exact Mann-Whitney comparisons (`key_only_random` versus
`full_kv`):

| Bits | Hidden delta (`key_only - full_kv`) | Hidden p | Attention error delta (`key_only - full_kv`) | Attention error p | Logit delta | Logit p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 0.057617 | 0.0143 | -0.291382 | 0.0143 | 0.000000 | 1.0000 |
| 2.5 | 0.041992 | 0.0143 | -0.254044 | 0.0143 | 0.000000 | 1.0000 |
| 3.0 | 0.017578 | 0.0143 | -0.160980 | 0.0143 | 0.000000 | 1.0000 |
| 3.5 | 0.010742 | 0.0143 | -0.136307 | 0.0143 | 0.000000 | 1.0000 |
| 4.0 | 0.003906 | 0.0571 | -0.087006 | 0.0143 | 0.000000 | 1.0000 |

Interpretation:

- The observed direction matches the requested H0 framing at every bit:
  `full_kv` always lowers hidden cosine and always increases
  `attention_output_relative_error` relative to `key_only_random`.
- Under the conventional null, hidden cosine differences are already visible at
  2.0 to 3.5 bit with raw exact p-values of `0.0143`, and the 4-bit gap is
  still in the same direction but smaller.
- Attention-output error is the clearest signal: every bit setting shows the
  same directional split, and the effect size remains large even at 4 bit.
- Logit cosine does not separate the two modes in this baseline. That is
  exactly the failure mode we care about here: the score path looks stable while
  the value transport path is degraded.

Because `n = 4` per group, Holm-corrected exact pairwise tests are conservative
and do not cross `0.05`; the omnibus Kruskal results and the consistent
directional deltas are therefore the more informative summary.

Paper baseline conclusion:

- `paper baseline result: key_only_random preserves hidden geometry better than full_kv on Qwen3.5-9B captured replay`
- mixed-bit `2.5 / 3.5` are intermediate Pareto points between low integer bits and higher integer bits
- statistically, the captured replay supports the same qualitative conclusion:
  TurboQuant reduces KV cache size as intended, but the paper-faithful
  `full_kv` path damages the V-dependent token-output transport signal before it
  meaningfully damages the logit score signal
