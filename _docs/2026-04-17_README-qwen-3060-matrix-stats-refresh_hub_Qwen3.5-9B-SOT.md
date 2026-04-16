# README Qwen 3060 Matrix Stats Refresh

## Overview

Rewrote the RTX 3060 12 GB README results section so the reduced real matrix is
documented with explicit mean +/- SD values, tracked error-bar figures, and
compact summary-statistics guidance instead of a few isolated bullet points.

## Why this change was needed

- The prior README only listed a handful of cherry-picked observations.
- The actual artifact bundle already contained richer statistical outputs:
  - `qwen_3060_matrix_mean_pm_sd.csv`
  - `qwen_3060_matrix_summary.csv`
  - `qwen_3060_matrix_summary.md`
  - exported PNG plots with error bars
- The user asked for the README to surface mean +/- SD, error-bar charts, and
  summary statistics directly.

## What changed

- Reworked `README.md` section `3b. Observed reduced real outcome` into a
  proper results subsection with:
  - explicit source-file pointers
  - evaluation shape (`4 prompts x 2 layers x 3 trials = 24` raw rows)
  - plot-convention notes explaining that the figure error bars come from `sem`
  - embedded quality/runtime PNG figures
  - a 4-bit headline mean +/- SD table covering all 7 matrix modes
  - a compact selected-summary-statistics table (`n`, SEM, 95% CI, memory)
  - statistical-reading bullets for the reduced slice
- Added tracked README image assets:
  - `_docs/assets/qwen_3060_matrix_attention.png`
  - `_docs/assets/qwen_3060_matrix_runtime.png`
- Extended the export section to state that the README copies of the exported
  PNGs live under `_docs\assets\`.

## Data sources used

- Mean +/- SD headline table:
  - `artifacts/qwen_3060_matrix/metrics/qwen_3060_matrix_mean_pm_sd.csv`
- Error-bar explanation and per-prompt summary-stat column names:
  - `artifacts/qwen_3060_matrix/metrics/qwen_3060_matrix_summary.csv`
- Pairwise significance and Friedman caveat:
  - `artifacts/qwen_3060_matrix/reports/qwen_3060_matrix_summary.md`
- README-tracked figures were copied from:
  - `artifacts/qwen_3060_matrix/plots/qwen_3060_matrix_attention.png`
  - `artifacts/qwen_3060_matrix/plots/qwen_3060_matrix_runtime.png`

## Commands run

```powershell
Get-Content README.md
Get-Content artifacts\qwen_3060_matrix\reports\qwen_3060_matrix_summary.md
Get-Content artifacts\qwen_3060_matrix\metrics\qwen_3060_matrix_mean_pm_sd.csv
Get-Content artifacts\qwen_3060_matrix\metrics\qwen_3060_matrix_summary.csv
Copy-Item artifacts\qwen_3060_matrix\plots\qwen_3060_matrix_attention.png _docs\assets\qwen_3060_matrix_attention.png
Copy-Item artifacts\qwen_3060_matrix\plots\qwen_3060_matrix_runtime.png _docs\assets\qwen_3060_matrix_runtime.png
git diff -- README.md _docs\2026-04-17_README-qwen-3060-matrix-stats-refresh_hub_Qwen3.5-9B-SOT.md
git diff --check -- README.md _docs\2026-04-17_README-qwen-3060-matrix-stats-refresh_hub_Qwen3.5-9B-SOT.md
```

## Verification

- `git diff --check -- README.md _docs\2026-04-17_README-qwen-3060-matrix-stats-refresh_hub_Qwen3.5-9B-SOT.md`
  - expected doc-only whitespace check
- `Get-ChildItem _docs\assets`
  - confirmed the tracked README PNG copies exist with non-zero sizes

## Notes for future edits

- The README now mixes two statistical views on purpose:
  - compact mean +/- SD from the exported `mean_pm_sd` table
  - selected SEM / 95% CI context derived from the raw trial bundle and the
    summary report
- If the matrix is rerun with more layers or more trials, refresh both the
  tracked `_docs/assets` PNGs and the numeric tables together so the README does
  not drift from the exported artifact bundle.
