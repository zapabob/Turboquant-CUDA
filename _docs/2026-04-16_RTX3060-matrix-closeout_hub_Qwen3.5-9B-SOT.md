# RTX 3060 Matrix Closeout Implementation Log

## Overview

Closed the reduced real RTX 3060 12 GB matrix slice for Qwen3.5-9B, exported the
report bundle, verified the vendored runtime build path, ran the remaining
follow-up pytest slice, and refreshed the repo docs so the 12 GB matrix is
documented as the current mainline artifact.

## Background / requirements

- User handoff identified the remaining closeout work:
  - complete the reduced real matrix run
  - export the matrix report
  - finish `README.md` and `implementation_plan.md`
  - verify the vendored runtime build
  - add the missing `_docs` implementation log
- Repository constraints:
  - correctness and reproducibility first
  - Windows-native / PowerShell-safe workflow
  - explicit artifact metadata and reviewable changes
- Quantization / experiment constraints:
  - keep the 7-mode 12 GB matrix explicit
  - preserve paper baseline vs research-extension separation
  - treat `key_only_block_so8_triality_vector` as the production K-side reference

## Assumptions / decisions

- Kept the user-provided reduced real scope unchanged:
  - `--bits 3,3.5,4`
  - `--trials 3`
  - `--max-layers 2`
- Re-ran the matrix in a detached PowerShell wrapper that writes an explicit
  exit-code file, because the prior background process had already died and an
  in-shell long run previously timed out without a durable completion signal.
- Limited repo edits to the closeout docs only. No quantization logic or runtime
  source files were changed in this slice.

## Changed files

- Modified:
  - `README.md`
  - `implementation_plan.md`
- Added:
  - `_docs/2026-04-16_RTX3060-matrix-closeout_hub_Qwen3.5-9B-SOT.md`
- Generated artifacts:
  - `artifacts/qwen_3060_matrix/metrics/*`
  - `artifacts/qwen_3060_matrix/plots/*`
  - `artifacts/qwen_3060_matrix/reports/qwen_3060_matrix_summary.md`
  - `artifacts/qwen_3060_matrix/qwen_3060_matrix_report.md`

## Implementation details

- Confirmed the original handoff PIDs were already gone and the prior run had
  terminated before writing the final matrix outputs.
- Inspected the existing `artifacts/qwen_3060_matrix_debug` bundle to verify the
  reporting/export path while treating it as a debug-only checkpoint, not the
  final artifact root.
- Re-ran `scripts/validate_qwen_3060_matrix.py` against
  `artifacts/kv_rtx3060_qwen9b` and
  `artifacts/research_extension/triality_full_train_prod_bf16/rotations` until
  `artifacts/qwen_3060_matrix/logs/validate_qwen_3060_matrix.exitcode.txt`
  recorded exit code `0`.
- Exported the final matrix bundle with `scripts/export_report.py`, producing
  the expected summary Markdown and plot outputs under
  `artifacts/qwen_3060_matrix`.
- Refreshed `README.md` with the observed reduced real outcomes so the 12 GB
  flow documents both the command path and the current artifact-level result
  highlights.
- Updated `implementation_plan.md` so its remaining-work and completion-evidence
  sections match the actual completed state of the closeout slice.

## Commands run

```powershell
Get-Process -Id 4200,21984,46892
Get-Content -Raw artifacts\qwen_3060_matrix\logs\validate_qwen_3060_matrix.stderr.log
Get-Content -Raw artifacts\qwen_3060_matrix_debug\metrics\qwen_3060_matrix_run_meta.json
uv run python scripts\validate_qwen_3060_matrix.py --kv-dir artifacts\kv_rtx3060_qwen9b --rotation-dir artifacts\research_extension\triality_full_train_prod_bf16\rotations --eval-device cuda --bits 3,3.5,4 --trials 3 --max-layers 2 --skip-plots --output-dir artifacts\qwen_3060_matrix
powershell -ExecutionPolicy Bypass -File .\scripts\build_rust_workspace.ps1 -Package hypura -NoCuda
uv run python -m pytest tests\test_capture.py tests\test_multiscreen_eval_analysis.py tests\test_triality_eval_recovery.py -q
uv run python scripts\export_report.py --matrix-dir artifacts\qwen_3060_matrix
git diff -- README.md implementation_plan.md
```

## Test / verification results

- `scripts\validate_qwen_3060_matrix.py`
  - passed on the detached rerun
  - `artifacts/qwen_3060_matrix/logs/validate_qwen_3060_matrix.exitcode.txt` = `0`
  - expected outputs now exist:
    - `metrics/qwen_3060_matrix_trials.csv`
    - `metrics/qwen_3060_matrix_summary.csv`
    - `metrics/qwen_3060_matrix_mean_pm_sd.csv`
    - `reports/qwen_3060_matrix_summary.md`
    - `plots/qwen_3060_matrix_attention.png`
    - `plots/qwen_3060_matrix_runtime.png`
- `uv run python scripts\export_report.py --matrix-dir artifacts\qwen_3060_matrix`
  - passed
  - exported the summary Markdown and plots from the final matrix CSVs
- `powershell -ExecutionPolicy Bypass -File .\scripts\build_rust_workspace.ps1 -Package hypura -NoCuda`
  - passed
  - repo contract validated as `OK`
  - `cargo build -p hypura -j 1` finished successfully
- `uv run python -m pytest tests\test_capture.py tests\test_multiscreen_eval_analysis.py tests\test_triality_eval_recovery.py -q`
  - passed
  - `20 passed in 11.09s`

## Residual risks

- The matrix artifact is still the intentionally reduced real run, not a
  full-layer or wider-bit sweep. Any README interpretation must stay scoped to
  `bits 3/3.5/4`, `trials 3`, and `max-layers 2`.
- The Friedman rows in the reduced matrix output remain `nan` / `1.0` because
  the reduced pooled blocks are too tied / degenerate for a meaningful omnibus
  statistic in this slice; the pairwise Wilcoxon-Holm rows carry the useful
  signal here.
- Existing unrelated dirty state remains in the workspace, including
  `artifacts/paper_baseline/*`, `artifacts/reports/env_check.txt`, and the
  vendored `vendor/llama.cpp` subtree.

## Recommended next actions

- If this matrix becomes the stable closeout baseline, consider adding a small
  checkpoint/resume facility to `scripts/validate_qwen_3060_matrix.py` similar
  to the triality eval flow, because the run is long enough that durable
  progress snapshots would reduce recovery risk.
- If broader evidence is needed, scale this same matrix outward incrementally:
  first more layers, then more trials, then wider bit sweeps.
- Keep future runtime or online-generation work gated on these offline artifacts
  rather than bypassing them.
