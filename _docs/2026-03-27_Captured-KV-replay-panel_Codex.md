# 2026-03-27 Captured KV Replay Panel / Codex

## Scope

Implemented the real captured-KV replay gate for `Qwen3.5-9B` in the hub repo
before any runtime-default change in `qwen35_rtx3060`.

Primary goal:

- capture real `K/V` tensors from the local official Hugging Face export
- store them in prompt-scoped, replay-ready directories
- run the same 7-mode replay matrix used in synthetic analysis
- emit a captured report that answers whether runtime should stay `key-only`

## Model Source

- Canonical local model path: `H:\Qwen3.5-9B-official-hf`
- Capture load mode: `4bit`
- Capture max length: `96`
- Mode: text-only

## Implemented Changes

### 1. Prompt-scoped capture layout

Updated `scripts/capture_qwen_kv.py` and `turboquant/capture.py` so the default
path captures a fixed 4-prompt panel and writes:

- `artifacts/kv/<capture_id>/capture_manifest.json`
- `artifacts/kv/<capture_id>/layer_{idx}_key.pt`
- `artifacts/kv/<capture_id>/layer_{idx}_value.pt`

Added manifest fields:

- `model_source`
- `prompt_label`
- `capture_id`
- `prompt_length`
- existing reproducibility metadata such as device, dtype, package versions, git commit hash

Prompt panel versioned in code:

1. explanatory prompt
2. reasoning prompt
3. coding prompt
4. summarization prompt

### 2. Qwen3.5 cache normalization fix

Extended `normalize_past_key_values(...)` to handle the actual cache object
returned by current `transformers` Qwen3.5 runtime.

This fixed the failure:

`Unsupported cache type: <class 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DynamicCache'>`

### 3. Captured replay loader and validation

Updated `turboquant/analysis.py` to load prompt-scoped captures through
`load_captured_runs(...)`.

Validation now fails early if:

- `capture_manifest.json` is missing
- a manifest-listed key tensor is missing
- a manifest-listed value tensor is missing

Threshold aggregation was also updated so captured rows are grouped by:

- `capture_id`
- `prompt_label`
- `prompt_hash`

This prevents prompt panel runs from being collapsed into one aggregate bucket.

### 4. Captured replay matrix execution

Updated `scripts/validate_attention_scores.py` so captured replay runs the same
mode matrix as synthetic replay:

- `exact`
- `key_only_random`
- `key_only_block_so8_static`
- `key_only_block_so8_learned`
- `protected_v`
- `protected_v_lowrank`
- `full_kv`

Also fixed `--max-layers` semantics so it applies per capture/prompt rather than
globally across all loaded captured layers.

### 5. Captured report outputs

The validate script now writes query-specific outputs:

- `attention_trials_captured.csv`
- `attention_metrics_long_captured.csv`
- `attention_summary_captured.csv`
- `attention_thresholds_captured.csv`
- `attention_summary_captured.md`

`scripts/export_report.py` was updated to:

- keep synthetic and captured summaries separate
- generate `attention_tradeoffs_captured.png`
- generate `attention_tradeoffs_captured.html`
- include the captured runtime recommendation in `artifacts/reports/summary.md`

### 6. Value low-rank replay fix

Updated `turboquant/value_codec.py` so low-rank basis fitting casts residuals to
`float32` before SVD. This fixed the captured replay failure on `bfloat16`.

## Tests Added

Added coverage in `tests/test_capture.py` for:

- missing manifest failure
- missing value tensor failure
- prompt-scoped capture directory loading
- Qwen-style cache normalization

## Commands Run

### Test suite

```powershell
uv run python -m pytest -q
```

Result:

- `24 passed in 31.28s`

### Real captured KV generation

```powershell
uv run python scripts\capture_qwen_kv.py --output-dir artifacts\kv --weight-load 4bit --max-length 96
```

Result:

- `saved 4 prompt captures to artifacts\kv`

Capture directories produced:

- `artifacts/kv/coding-ca766984`
- `artifacts/kv/explain-359133ba`
- `artifacts/kv/reasoning-30ae1199`
- `artifacts/kv/summary-42f56222`

### Captured replay

```powershell
uv run python scripts\validate_attention_scores.py --query-source captured --kv-dir artifacts\kv --trials 1 --max-layers 1 --bits 2,2.5,3.5,4
```

### Synthetic replay refresh

```powershell
uv run python scripts\validate_attention_scores.py --query-source synthetic --trials 2 --synthetic-layers 2 --batch 1 --heads 2 --seq-len 32 --head-dim 128
```

### Report regeneration

```powershell
uv run python scripts\export_report.py
```

## Captured Result Snapshot

From `artifacts/metrics/attention_summary_captured.csv`:

- `key_only_block_so8_learned`, 2.0 bits
  - hidden cosine: `1.001953`
  - memory ratio: `0.566406`
- `protected_v`, 2.0 bits
  - hidden cosine: `0.957031`
  - memory ratio: `0.204352`
- `protected_v_lowrank`, 2.0 bits
  - hidden cosine: `0.960938`
  - memory ratio: `0.212164`
- `full_kv`, 2.0 bits
  - hidden cosine: `0.940430`
  - memory ratio: `0.130859`

## Conclusion

Captured replay does show a real middle Pareto region:

- `protected_v` and `protected_v_lowrank` improve hidden-state stability over `full_kv`
- but neither branch approaches `key_only_block_so8_learned` closely enough to justify changing the runtime default

Current captured recommendation:

- `protected-V is promising but not ready`

Decision:

- keep runtime default as `key-only`
- continue value-side work as a research branch until a captured replay branch materially beats `full_kv` and narrows the remaining gap to `key_only_block_so8_learned`
