# RTX 3060 Plan Refresh Implementation Log

## Overview

Rewrote `implementation_plan.md` so the repository's execution plan matches a
realistic RTX 3060-first scope: Qwen-first, text-only, offline correctness
before runtime expansion, and explicit separation between the default 3060
runtime path and the Triality research lane.

## Background / requirements

- User request: cut the plan down for RTX 3060 reality and update the repo plan.
- Repository constraints:
  - correctness and reproducibility first
  - Windows-native / PowerShell-safe workflow
  - no hidden scope expansion
- Quantization constraints:
  - keep Stage 1 / Stage 2 visible
  - keep paper baseline and research extension distinct
  - preserve the repo-wide K-side Triality reference while still narrowing the
    first runtime slice

## Assumptions / decisions

- Treated `implementation_plan.md` as the user-selected canonical plan location
  for this refresh rather than creating a second plan file.
- Narrowed the milestone to Qwen3.5-9B 12 GB primary lane plus Qwen3.5-4B 6 GB
  reduced lane.
- Deferred Gemma runtime work, weight-side `TQ4_1S`, and vision integration out
  of the first 3060 milestone.
- Preserved the existing Triality reference as research / production K-side
  guidance, but kept it out of the default 3060 runtime sequence for the first
  slice.

## Changed files

- Modified:
  - `implementation_plan.md`
- Added:
  - `_docs/2026-04-16_RTX3060-plan-refresh_hub_Qwen3.5-9B-SOT.md`

## Implementation details

- Replaced the older high-level first-pass plan with a tasked plan that:
  - locks the 12 GB and 6 GB lanes
  - centers the first milestone on Qwen offline replay
  - adds a dedicated `head_dim = 256` regression checkpoint
  - restricts `llama.cpp` runtime work to Qwen full-attention layers only
  - inserts a hard review gate before any Gemma work starts
- Added a file map so future implementation work has exact repo entrypoints.
- Added explicit non-goals to prevent drift into 27B/31B, `TQ4_1S`, early
  vision, and premature Triality runtime defaulting.

## Commands run

```powershell
Get-Content -Raw implementation_plan.md
Get-Content -Raw README.md
Get-Content -Raw scripts\capture_qwen_kv.py
Get-Content -Raw scripts\paper_validate_captured_qwen.py
Get-Content -Raw turboquant\runtime.py
Get-Content -Raw turboquant\research_extension\k_triality.py
Get-Content -Raw vendor\llama.cpp\src\models\qwen35.cpp
Get-Content -Raw vendor\llama.cpp\src\models\gemma4-iswa.cpp
```

## Test / verification results

- No automated test suite was run.
- Reason:
  - this change is documentation / planning only
  - no runtime, quantization, or script behavior changed
- Verification performed:
  - checked the current repo structure and relevant source files before
    rewriting the plan
  - confirmed the vendored Qwen and Gemma model source file paths used by the
    new plan

## Residual risks

- `implementation_plan.md` now intentionally leads runtime work toward a narrow
  3060 slice, but the actual codebase still contains broader research paths and
  existing artifacts that could tempt scope drift during implementation.
- The plan references a future Qwen3.5-4B lane, but the current workspace still
  appears centered on 9B assets and scripts.
- Because this was a docs-only change, no automated guard currently enforces the
  new scope boundaries.

## Recommended next actions

- Implement Task 1 from `implementation_plan.md`: add explicit 12 GB / 6 GB
  lane metadata and presets in `turboquant/runtime.py` and related capture
  metadata paths.
- Add the `head_dim = 256` asymmetric regression slice before touching
  `vendor/llama.cpp`.
- Keep Gemma and Triality runtime-default work behind the Qwen offline closure
  checkpoint.
