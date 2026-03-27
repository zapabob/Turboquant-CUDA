# 2026-03-27 TurboQuant Pareto Comparison / Codex

## Goal

Implemented the Pareto-comparison extension for the hub-side offline replay so
the experiment is explicitly evaluated as:

- same base `Qwen3.5-9B` weights
- only the KV codec changes
- captured KV replay is the decision gate
- quality is reported on both hidden and logit axes
- memory is reported on both KV ratio and runtime-side memory axes

## What Changed

### 1. Replay metrics now include runtime-side memory

Extended `turboquant/analysis.py` so replay rows now emit:

- `prefill_seconds`
- `decode_seconds`
- `peak_vram_mb`

This was added for:

- `exact`
- `key_only_random`
- `key_only_block_so8_static`
- `key_only_block_so8_learned`
- `protected_v`
- `protected_v_lowrank`
- `full_kv`

`exact` is no longer a zero-timing placeholder. It is now measured as the real
baseline for replay timing.

### 2. Explicit eval device support

Added `--eval-device` to `scripts/validate_attention_scores.py`.

Behavior:

- `auto` => `cuda` if available, else `cpu`
- replay tensors are moved explicitly to the chosen device
- CUDA runs measure peak allocated VRAM
- CPU runs report `peak_vram_mb = 0.0`

### 3. Main table vs runtime table split

Updated summary generation so reports are split into:

- primary Pareto table
  - `memory_ratio_vs_exact`
  - `hidden_cosine_similarity`
  - `hidden_mse`
  - `logit_cosine_similarity`
  - `logit_top1_match`
  - `logit_top5_overlap`
- secondary runtime table
  - `prefill_seconds`
  - `decode_seconds`
  - `peak_vram_mb`

This applies to both:

- `artifacts/metrics/attention_summary*.md`
- `artifacts/reports/summary.md`

### 4. Runtime plots added

Updated `scripts/export_report.py` to generate:

- `artifacts/plots/attention_runtime_tradeoffs.png`
- `artifacts/plots/attention_runtime_tradeoffs.html`
- `artifacts/plots/attention_runtime_tradeoffs_captured.png`
- `artifacts/plots/attention_runtime_tradeoffs_captured.html`

The existing quality/memory plots remain unchanged.

### 5. Test updates

Updated `tests/test_analysis.py` so the replay rows are expected to include:

- `prefill_seconds`
- `decode_seconds`
- `peak_vram_mb`

## Verification

### Passed

```powershell
uv run python -m pytest -q
```

Result:

- `25 passed`

### Passed

```powershell
uv run python scripts\validate_attention_scores.py --query-source synthetic --trials 2 --synthetic-layers 2 --batch 1 --heads 2 --seq-len 32 --head-dim 128 --eval-device cpu
```

This regenerated synthetic summaries with the new runtime metrics and the new
primary/secondary report sections.

### Passed

```powershell
uv run python scripts\export_report.py
```

This regenerated `artifacts/reports/summary.md` and the new runtime trade-off
plots.

## Captured Replay Note

The captured replay code path was updated for `--eval-device auto`, but the full
CUDA rerun is still computationally heavy because it recalibrates the
learned-SO(8) and protected-value branches across prompt/layer/mode/bit
combinations.

So the state at the end of this turn is:

- implementation is complete
- synthetic replay and reporting are regenerated with the new Pareto layout
- captured replay implementation is ready for CUDA VRAM measurement
- the previously established runtime recommendation remains unchanged:
  - keep runtime default as `key-only`
  - `protected-V` remains promising but not ready
