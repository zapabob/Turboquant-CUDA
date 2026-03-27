# 2026-03-27 KV-cache reduction research implementation log

## Summary

This log records the implementation of the research-oriented KV-cache reduction plan in
`H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT`.

Goal:

- keep the paper-faithful TurboQuant core as the canonical reference
- split key and value objectives explicitly
- add `block-SO(8)` rotation policies for key compression
- add a sensitivity-protected value codec with optional low-rank residuals
- evaluate the new Pareto points offline before any runtime-default change

Implementation AI: Codex

## Main code changes

### 1. Public types and configuration surface

Updated `turboquant/types.py`:

- added `RotationPolicy`
  - `random_haar`
  - `block_so8_static`
  - `block_so8_learned`
- added `SensitivitySpec`
- added `ValueCodecConfig`
- added `MemoryBudgetSpec`
- added `ProtectedValueBatch`
- extended `TurboQuantMSEConfig` and `TurboQuantProdConfig` with `rotation_policy`

This makes the K/V split explicit instead of burying it in ad-hoc flags.

### 2. Rotation subsystem

Updated `turboquant/rotation.py`:

- added cached block-diagonal SO(8) rotation construction
- added `block_so8_rotation(...)`
- added `block_so8_from_skew(...)`
- added `rotation_from_policy(...)`

The `block_so8_learned` path uses skew-symmetric block generators and
`torch.matrix_exp` to stay on the orthogonal manifold.

### 3. Stage 1 TurboQuant updates

Updated `turboquant/turboquant_mse.py`:

- rotation is now selected via `rotation_policy`
- added `set_rotation(...)`
- added `quantize_with_bitwidths(...)`
- added quantization-aware `fit_rotation(...)` for `block_so8_learned`

The fitting objective combines:

- reconstruction MSE
- optional score cosine preservation
- optional KL term on the softmax ranking proxy

This is still an offline calibration procedure; it is not token-adaptive at runtime.

### 4. Stage 2 / key codec updates

Updated `turboquant/turboquant_prod.py`:

- passed `rotation_policy` through to the Stage 1 quantizer
- added `fit_rotation(...)` forwarding for the learned SO(8) path

Keys remain the paper-faithful `TurboQuantProd` branch.

### 5. Protected value codec

Added `turboquant/value_codec.py`:

- `ProtectedValueCodec`
- calibration from value energy or attention-weighted value energy
- exact/high/low protection masks
- optional low-rank residual basis

Current prototype policy:

- top `protected_fraction` channels per `(layer/head)` prefix are exact
- next `secondary_fraction` channels are high-bit
- remaining channels are low-bit
- optional low-rank coefficients are stored per token when enabled

### 6. KV codec integration

Updated `turboquant/kv_codec.py`:

- `KVCodecConfig` now carries
  - `rotation_policy`
  - `value_codec`
  - `sensitivity`
  - `memory_budget`
- added `calibrate(keys, values, queries)`
- added `encode_protected_values(...)`
- added `decode_protected_values(...)`
- added `protected_value_storage_bits(...)`

This keeps the canonical key path as `TurboQuantProd` while letting the value side use a different objective.

### 7. Offline replay analysis

Updated `turboquant/analysis.py`:

- added mode-aware evaluation for:
  - `exact`
  - `key_only_random`
  - `key_only_block_so8_static`
  - `key_only_block_so8_learned`
  - `protected_v`
  - `protected_v_lowrank`
  - `full_kv`
- added codec calibration before learned/protected modes
- made value-mode handling explicit:
  - exact values
  - full quantized values
  - protected values

### 8. Reporting and summaries

Updated:

- `scripts/validate_attention_scores.py`
- `scripts/export_report.py`
- `README.md`

Changes:

- replay summaries now understand the new research modes
- bottleneck text compares learned-SO(8) key-only against full-KV
- reports include protected-V branches in tables and plots

## Tests added or updated

Updated tests:

- `tests/test_rotation.py`
  - verifies block-SO(8) orthogonality
- `tests/test_kv_codec.py`
  - verifies calibration + protected value path
- `tests/test_analysis.py`
  - verifies the research mode set is emitted

Added:

- `tests/test_value_codec.py`
  - verifies sensitivity mask construction
  - verifies protected value round-trip shape and memory ratio

## Validation run

Executed:

```powershell
uv run python -m pytest -q
uv run python scripts\test_synthetic.py --trials 2 --heads 2 --seq-len 32
uv run python scripts\validate_attention_scores.py --query-source synthetic --trials 2 --synthetic-layers 2 --batch 1 --heads 2 --seq-len 32 --head-dim 128
uv run python scripts\export_report.py
```

Result:

- `pytest`: `21 passed in 39.97s`
- synthetic replay/report generation: succeeded

## Observed synthetic replay result

Representative means from `artifacts/metrics/attention_summary.csv`:

### Learned SO(8) key-only

- `2.0 bits`
  - logit cosine: `0.947570`
  - hidden cosine: `0.999999`
  - memory ratio: `0.539062`
- `2.5 bits`
  - logit cosine: `0.973792`
  - hidden cosine: `1.000000`
  - memory ratio: `0.542969`

### Protected-V

- `2.0 bits`
  - logit cosine: `0.947570`
  - hidden cosine: `0.955686`
  - memory ratio: `0.135498`
- `3.5 bits`
  - logit cosine: `0.987415`
  - hidden cosine: `0.987256`
  - memory ratio: `0.169067`

### Protected-V + low-rank

- `2.0 bits`
  - logit cosine: `0.947570`
  - hidden cosine: `0.962470`
  - memory ratio: `0.151123`
- `3.5 bits`
  - logit cosine: `0.987415`
  - hidden cosine: `0.989023`
  - memory ratio: `0.184692`

### Full-KV

- `2.0 bits`
  - logit cosine: `0.947670`
  - hidden cosine: `0.942180`
  - memory ratio: `0.074219`
- `3.5 bits`
  - logit cosine: `0.986553`
  - hidden cosine: `0.988886`
  - memory ratio: `0.113281`

## Current interpretation

The synthetic evidence still points to the same high-level conclusion:

- the dominant failure mode remains the value path
- `block_so8_learned` does not magically solve value drift, but it gives a stronger key-only baseline
- `protected_v` and `protected_v_lowrank` are now concrete middle Pareto points
- `protected_v_lowrank` is better than naive full-KV on hidden cosine, but it is not yet competitive with key-only on quality

## Next recommended step

Before changing the runtime default in `qwen35_rtx3060`, run the same replay matrix on captured KV tensors:

1. captured exact
2. key-only random
3. key-only block-SO(8)
4. protected-V
5. protected-V + low-rank
6. full-KV

Only after that should the runtime harness be updated beyond the current key-only default.
