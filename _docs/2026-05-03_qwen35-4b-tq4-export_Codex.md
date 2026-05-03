# 2026-05-03 - Qwen3.5-4B TQ4_1S GGUF export

## Goal

Allow the offline GGUF weight converter to produce an authoritative Turboquant
TQ4_1S artifact from the ELT Qwen3.5-4B Q8_0 GGUF, even when the source GGUF
already contains provisional `hypura.turboquant.*` metadata.

## Files touched

- `turboquant/weight_gguf.py`
- `scripts/convert_weight_turboquant_gguf.py`
- `tests/test_eval_scripts.py`
- `tests/test_weight_gguf.py`
- `tests/test_triality_contract.py`

## Key decisions

- Kept the default converter behavior fail-closed when a source GGUF already has
  `hypura.turboquant.*` metadata.
- Added an explicit `--replace-existing-turboquant-metadata` CLI flag for the
  known two-step path where a prior GGUF tool writes provisional metadata and
  this converter writes the authoritative TQ4_1S namespace.
- Skipped copied `general.file_type` before writing the converter's output type
  to avoid duplicate key warnings.
- Added Qwen3.5-4B inference and contract coverage alongside the existing
  Qwen3.5-9B/27B coverage.
- Matched the existing optional-HF test convention by skipping the HF online
  script dry-run test when `transformers` is not installed.

## Tests and verification

- `uv run --no-sync pytest -q`
  - 147 passed, 3 skipped
- `uv run --no-sync pytest -q tests/test_weight_gguf.py::test_convert_weight_turboquant_gguf_can_replace_existing_metadata_and_infer_qwen35_4b tests/test_weight_gguf.py::test_convert_weight_turboquant_gguf_rewrites_selected_q8_tensors tests/test_triality_contract.py::test_qwen35_payload_uses_weight_v1_config_i_contract`
  - 5 passed
- `uv run --no-sync python -m py_compile turboquant/weight_gguf.py scripts/convert_weight_turboquant_gguf.py`
  - passed
- Real conversion:
  `uv run --no-sync python scripts/convert_weight_turboquant_gguf.py --input-gguf H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-Q8_0.gguf --output-gguf H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-TQ4_1S.gguf --model-family Qwen/Qwen3.5-4B --replace-existing-turboquant-metadata --force`
  - converted 84 tensors
  - preserved 343 tensors
  - wrote `H:/elt_data/releases/elt-lm-qwen35-side-stem-v2-bridge-TQ4_1S.gguf`

## Next session notes

- The real TQ4_1S artifact metadata reports
  `hypura.turboquant.weight.policy=qwen35-config-i` and model family
  `Qwen/Qwen3.5-4B`.
- Runtime generation smoke remains separate from the offline byte rewrite.
