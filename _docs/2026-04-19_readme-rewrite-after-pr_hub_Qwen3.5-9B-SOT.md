# 2026-04-19 README rewrite after PR sync (Codex)

## Overview

Pulled the current `origin/main` state of `Turboquant-CUDA` and rewrote the
README so it reflects the newly merged Triality weight-contract work without
overstating runtime completeness.

## Background / requirements

- The repository advanced after the previous local checkout.
- The new upstream change added `triality_contract` tightening, a new
  `tests/test_triality_contract.py`, and updated `_docs` around a weight-v1 PR.
- The user requested pulling the current PR-updated repository state and
  rewriting `README.md`.

## Assumptions / decisions

- Treated `origin/main` as the requested source of truth because the local
  branch was tracking `origin/main` and the latest PR changes had already been
  fast-forwarded there.
- Kept the README rewrite focused on the newly merged contract work.
- Explicitly documented that weight metadata / fixture export are implemented,
  while native weight codec and CUDA matmul are still not implemented in this
  repo.

## Changed files

- `README.md`
- `_docs/2026-04-19_readme-rewrite-after-pr_hub_Qwen3.5-9B-SOT.md`

## Implementation details

- Pulled `origin/main` with `git pull --ff-only origin main`.
- Updated the README overview and TL;DR to mention:
  - Triality fixture export
  - `hypura.turboquant.weight.v1`
  - `tq4_1s` contract status
  - the distinction between contract completion and runtime completion
- Added a dedicated weight-status section to avoid implying that native
  weight-side TurboQuant already exists end-to-end in this repo.
- Added fixture export and verification examples for:
  - Qwen 3.5 text-only
  - Gemma 4 paired multimodal (`mmproj`)
- Updated runtime-eval examples to use the current `zapabob/llama.cpp` path
  rather than the old `vendor/llama.cpp` path.

## Commands run

- `git fetch --all --prune`
- `git pull --ff-only origin main`
- `git rev-parse HEAD`
- `uv sync --extra cpu --extra dev`
- `uv run python scripts\validate_repo_contract.py`
- `uv run python -m pytest tests\test_triality_contract.py -q`

## Test / verification results

- Passed: `uv run python scripts\validate_repo_contract.py`
- Passed: `uv run python -m pytest tests\test_triality_contract.py -q` (`9 passed`)
- Confirmed: local checkout fast-forwarded to `origin/main` at `50b386b57c4f3ad8f289ccae35a4aeb2cf0fdbd5`

## Residual risks

- The README now reflects the merged contract work, but it still depends on the
  current runtime code remaining aligned with the documented non-claims around
  weight-side support.
- No large-model capture or runtime benchmark was rerun as part of this docs
  update.

## Recommended next actions

- Keep the README and `triality_contract` in sync whenever the weight payload
  schema changes.
- If upstream adds true `tq4_1s` runtime execution, revisit the README's
  non-claims section and split stream-time dequant from native compressed
  resident matmul support.
