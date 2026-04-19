# 2026-04-19 Triality weight-v1 PR prep (Codex)

## Overview

Prepared the `Turboquant-CUDA` branch for upstream review by completing the
weight-contract v1 follow-up, tightening GGUF packaging details, and verifying
the export and manifest flow for both Qwen 3.5 and Gemma 4 family fixtures.

## Background / requirements

- The branch already carried the paired multimodal manifest export work.
- Remaining local changes completed the stricter weight payload contract and the
  GGUF writer adjustments needed for loader-facing compatibility.
- The branch needed verification before opening a PR to
  `zapabob/Turboquant-CUDA`.

## Assumptions / decisions

- Kept the branch focused on the Triality fixture export and contract surface.
- Promoted `pyyaml` to an explicit dependency because vendored `gguf` imports
  `yaml` during packaging and test execution.
- Verified Qwen and Gemma export paths with targeted smoke runs instead of
  broader offline research jobs.

## Changed files

- `pyproject.toml`
- `requirements.txt`
- `tests/test_turboquant_gguf_profiles.py`
- `tests/test_triality_contract.py`
- `turboquant/gguf_profiles.py`
- `turboquant/triality_contract.py`
- `uv.lock`

## Implementation details

- Added explicit `weight.codec=tq4_1s` handling and validation to the Triality
  metadata and payload contract.
- Switched the default weight policy families to:
  - `qwen35-config-i`
  - `gemma4-kv-first-multimodal-safe`
- Added strict `weight_plan` validation, including schema, codec, and tensor
  plan checks.
- Added Qwen/Gemma contract tests and paired-mmproj export tests.
- Shortened embedded rotation tensor names and added a hashed fallback so names
  stay under GGUF length limits.
- Switched GGUF packaging to explicit tensor-info plus tensor-data writing to
  preserve source tensor metadata cleanly.
- Made `pyyaml` explicit in the project dependency surface so vendored `gguf`
  imports work in a fresh synced environment.

## Commands run

- `uv sync --extra cpu --extra dev`
- `uv pip install pyyaml`
- `uv run python -m pytest tests/test_turboquant_gguf_profiles.py tests/test_triality_contract.py -q`
- `uv run python scripts/verify_triality_export.py --help`
- `uv run python scripts/export_triality_fixture.py --output-dir $tmp --mode triality-proxy-so8-pareto --model-family Qwen/Qwen3.5-27B`
- `uv run python scripts/verify_triality_export.py --manifest ...triality-fixture-manifest.json`
- `uv run python scripts/export_triality_fixture.py --output-dir $tmp --mode triality-proxy-so8-pareto --model-family google/gemma-4-e4b-it`
- `uv run python scripts/verify_triality_export.py --manifest ...triality-fixture-manifest.json`

## Test / verification results

- Passed: `tests/test_turboquant_gguf_profiles.py`
- Passed: `tests/test_triality_contract.py`
- Passed: Qwen 3.5 fixture export and manifest verification smoke
- Passed: Gemma 4 paired-manifest export and manifest verification smoke

## Residual risks

- Verification stayed at fixture export and contract level; no large-model
  capture or offline metric sweep was rerun in this turn.
- `uv pip install pyyaml` was used before tests; the dependency is now explicit
  in project metadata, but a fresh `uv sync` should still be rerun by CI or a
  reviewer after pulling the branch.

## Recommended next actions

- Push the updated branch.
- Open a PR against `zapabob/Turboquant-CUDA`.
- If reviewer confidence needs more evidence, add one reduced real-model export
  smoke on top of the current synthetic fixture validation.
