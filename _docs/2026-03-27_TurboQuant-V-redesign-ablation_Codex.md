# TurboQuant V Redesign Ablation Log

## Summary

This update implements the first research-oriented split between `K` and `V` in the offline replay harness.

Working hypothesis:

- `K` is an inner-product / score preservation problem.
- `V` is an attention-output / hidden-state transport problem.
- The current failure mode is more likely to come from reusing a `K`-style geometry and objective on `V` than from the `TurboQuantProd` key-side theory itself.

The runtime default in `qwen35_rtx3060` was intentionally left unchanged.

## Code Changes

### Core analysis

- Restored and expanded [turboquant/analysis.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\turboquant\analysis.py).
- Added new replay metrics:
  - `next_logit_kl`
  - `attention_output_relative_error`
- Added new value-side ablation modes under fixed `K = key_only_block_so8_learned`:
  - `v_mse_random`
  - `v_mse_block_so8`
  - `v_prod_random`
  - `v_prod_block_so8`
  - `protected_v`
  - `protected_v_lowrank`
- Added value sensitivity helpers:
  - `compute_value_sensitivity_rows`
  - `summarize_value_sensitivity`
  - `compose_sensitive_layer_policy_rows`
- Added protection-grid evaluation helper:
  - `evaluate_value_protection_grid`

### Quantization support

- Extended [turboquant/qjl.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\turboquant\qjl.py) with a research-only `decode()` path for residual transport experiments.
- Extended [turboquant/turboquant_prod.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\turboquant\turboquant_prod.py) with `transport_decode()` for value-side Prod ablations.
- Extended [turboquant/types.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\turboquant\types.py) with `channel_group_size` in `ValueCodecConfig`.
- Reworked [turboquant/value_codec.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\turboquant\value_codec.py):
  - grouped channel protection
  - attention-output sensitivity proxy
  - teacher-gradient proxy
  - persistent `channel_sensitivity()` and `group_sensitivity()` accessors
  - optional low-rank residual basis

### Metrics and reporting

- [turboquant/attention_metrics.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\turboquant\attention_metrics.py) now exports:
  - KL divergence from logits
  - relative Frobenius output error
- [scripts/validate_attention_scores.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\scripts\validate_attention_scores.py) now:
  - evaluates the new `V` ablation modes
  - writes value sensitivity CSVs on captured replay
  - writes coarse sensitive-layer policy CSVs
  - writes static protection-grid CSVs and summary CSVs
  - exposes `--sensitivity-group-size`
  - exposes `--protection-grid-layer-limit`
- [scripts/export_report.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\scripts\export_report.py) now recognizes the new ablation modes and includes them in the replay plots/tables.

## Tests

Updated tests:

- [tests/test_analysis.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\tests\test_analysis.py)
- [tests/test_qjl.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\tests\test_qjl.py)
- [tests/test_turboquant_prod.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\tests\test_turboquant_prod.py)
- [tests/test_value_codec.py](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\tests\test_value_codec.py)

Verification commands:

```powershell
uv run python -m compileall turboquant scripts tests
uv run python -m pytest -q
uv run python scripts\validate_attention_scores.py --query-source synthetic --trials 1 --synthetic-layers 1 --batch 1 --heads 2 --seq-len 16 --head-dim 16 --bits 2,3 --eval-device cpu --output-dir artifacts\metrics_smoke
uv run python scripts\validate_attention_scores.py --query-source captured --kv-dir artifacts\kv --trials 1 --max-layers 1 --bits 2 --eval-device cpu --output-dir artifacts\metrics_smoke_captured --protection-grid-layer-limit 1
uv run python scripts\export_report.py
```

Observed outcomes:

- `29 passed in 26.33s`
- synthetic replay smoke completed with the new `V_mse_*` / `V_prod_*` / `protected_v*` modes
- captured replay smoke completed with:
  - attention summary outputs
  - value sensitivity outputs
  - sensitive-layer policy outputs
  - protection-grid outputs

## Notes

- The protection grid is intentionally limited in the validation script by `--protection-grid-layer-limit` so the default path does not explode in runtime on full captured runs.
- Grouped protection is aligned with the current block-SO(8) path to avoid mixing protected and unprotected channels under the value-side research branch.
- This change does not yet separate `paper_baseline` and `research_extension` in the public repo structure; that remains a later cleanup step after the experiments settle.
