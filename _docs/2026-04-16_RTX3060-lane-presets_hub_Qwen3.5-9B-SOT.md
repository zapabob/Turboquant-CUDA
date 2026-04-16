# RTX 3060 Lane Presets Implementation Log

## Overview

Implemented Task 1 from `implementation_plan.md`: explicit RTX 3060 capture
presets plus capture-manifest metadata for lane name, seed, and quantization
configuration.

## Background / requirements

- User request: proceed with Task 1 from the refreshed RTX 3060-first plan.
- Repository rules:
  - correctness and reproducibility first
  - Windows-native / PowerShell-safe workflow
  - explicit metadata for experiment reproduction
- Task 1 acceptance targets:
  - expose `qwen35_9b_12gb` and `qwen35_4b_6gb`
  - record lane name, seed, and quantization config in capture manifests
  - avoid silently mapping the 6 GB lane back onto the existing local 9B path

## Assumptions / decisions

- Preserved the existing `default` / `base` preset names as aliases to avoid
  breaking older capture flows.
- Kept the local 9B Windows path as the resolved load source for the 12 GB
  preset, but now store the canonical model name separately from the source
  path.
- Used HF model ID `Qwen/Qwen3.5-4B` for the new 6 GB preset because no local
  4B mirror was found in the workspace.
- Chose `seed=0` as the capture CLI default and wrote it into the manifest even
  though the current capture flow is not heavily stochastic.

## Changed files

- Modified:
  - `turboquant/runtime.py`
  - `turboquant/schema.py`
  - `turboquant/capture.py`
  - `scripts/capture_qwen_kv.py`
  - `tests/test_capture.py`
- Added:
  - `_docs/2026-04-16_RTX3060-lane-presets_hub_Qwen3.5-9B-SOT.md`

## Implementation details

- Added explicit runtime presets:
  - `qwen35_9b_12gb`
  - `qwen35_9b_base_12gb`
  - `qwen35_4b_6gb`
- Added alias resolution so legacy `default` and `base` still work.
- Added `build_capture_quantization_config(...)` and validation in
  `turboquant/schema.py` so capture-time quantization metadata has one canonical
  JSON shape.
- Extended `CaptureMetadata` with:
  - `model_preset`
  - `lane_name`
  - `seed`
  - `quantization_config`
- Updated capture-manifest loading to validate `quantization_config` when
  present.
- Updated `scripts/capture_qwen_kv.py` to:
  - resolve model name separately from model source
  - accept explicit lane override
  - seed PyTorch / CUDA before model load
  - write manifest metadata for lane, seed, preset, and quantization config
- Added focused tests for:
  - runtime preset resolution
  - 6 GB preset not resolving to the local 9B capture path
  - capture-manifest round-trip for lane / seed / quantization metadata

## Commands run

```powershell
uv run python -m pytest tests\test_capture.py -q
uv run python scripts\env_check.py
git diff --check
```

## Test / verification results

- `uv run python -m pytest tests\test_capture.py -q`
  - passed
  - `6 passed in 5.33s`
- `uv run python scripts\env_check.py`
  - passed
  - `status: ok`
  - CUDA target match confirmed on `NVIDIA GeForce RTX 3060`
- `git diff --check`
  - no whitespace / patch-shape errors
  - CRLF warnings remain informational only in this workspace

## Residual risks

- No end-to-end capture smoke run was performed with the new preset logic
  against a real model load, so the CLI path is only covered by focused unit
  tests in this slice.
- The new 4B preset currently points to the public HF model ID; if a local 4B
  mirror is later added, the preset may need to prefer that path.
- `env_check.py` still reports the default model at a high level and does not
  yet print the named 3060 preset or lane explicitly.

## Recommended next actions

- Implement Task 2: wire the 12 GB / 6 GB lane distinction into the captured
  replay reporting path.
- Add a lightweight script-level smoke test for `resolve_capture_target(...)`
  or the capture CLI argument matrix if Task 2 starts broadening the capture
  surface.
- Keep Gemma and runtime integration out of scope until the Qwen offline lane
  remains green.
