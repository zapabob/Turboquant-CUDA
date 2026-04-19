# 2026-04-19 README rewrite for latest Turboquant-CUDA mainline (Codex)

## Overview

Rewrote the repository README so the current branch reads like a strong project
landing page instead of only a research notebook.

## Background / requirements

- The README needed to reflect the latest branch state, not just the older
  Qwen 3060 research lane.
- Recent branch work added:
  - paired Gemma multimodal fixture manifests
  - strict weight payload v1 export
  - explicit `tq4_1s` codec metadata
  - tighter GGUF export behavior
- Those changes were not prominent enough in the previous README.

## Assumptions / decisions

- Used a `balanced mainline` structure with a product-first opening and the
  research evidence retained later in the document.
- Kept the repo honest about scope: Qwen 3060 remains the strongest measured
  lane, while Gemma 4 is primarily an export-contract lane here.
- Compressed the long research sections instead of trying to preserve every
  previous table and narrative detail in the landing page.

## Changed files

- `README.md`
- `_docs/2026-04-19_readme-rewrite-latest-mainline_Codex.md`

## Implementation details

- Reframed the opening around offline research plus runtime-contract export.
- Added an explicit `Latest Mainline Additions` section for the dual-family
  export and weight-v1 work.
- Added a short contract snapshot showing the current `weight.v1` structure.
- Preserved the measured 3060 snapshot and figure references, but moved them
  behind the higher-level product story.
- Tightened the quick-start and smoke sections around the current export and
  verification scripts.

## Commands run

- `Get-Content README.md`
- `Get-Content scripts/export_triality_fixture.py`
- `Get-Content scripts/verify_triality_export.py`
- `Get-Content tests/test_triality_contract.py`

## Test / verification results

- No code behavior changed; this was a documentation rewrite only.
- README claims were aligned with the current branch features and the already
  passing contract/export tests on this branch.

## Residual risks

- The README is intentionally shorter than the prior research-heavy version, so
  some deep contextual detail now lives in scripts, tests, and artifacts rather
  than the landing page.
- If the branch lands with further export-surface changes, the `Latest Mainline
  Additions` and contract snapshot sections may need one more refresh.

## Recommended next actions

- Push this README update to the existing branch so PR #6 shows the new
  positioning.
- If desired later, split deep result discussion into a dedicated docs page and
  keep the README even tighter.
