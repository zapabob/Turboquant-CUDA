# RTX 3060 12GB Qwen Matrix Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the current RTX 3060 desktop 12 GB milestone by finishing the Qwen3.5-9B offline comparison matrix, exporting the report, refreshing the repo docs, and verifying the vendored runtime build.

**Architecture:** Keep the repository split explicit. `turboquant.paper_baseline` remains the paper-faithful correctness anchor, `turboquant.research_extension` remains the K/V research lane, and the RTX 3060 completion slice is a fixed offline 12 GB Qwen matrix that feeds reporting and a narrow runtime-consumption verification. Do not broaden scope before the offline matrix artifacts are complete and documented.

**Tech Stack:** Windows PowerShell, `uv`, Python 3.12.9, CUDA 12.8 PyTorch, pandas/matplotlib reporting, vendored `zapabob/llama.cpp`, Rust/Hypura workspace build scripts.

---

## Scope Lock

### Primary target

- RTX 3060 desktop 12 GB only.
- Qwen3.5-9B text-only captured KV only.
- Offline validation and report export first, runtime build verification second.

### Canonical comparison matrix

- `exact`
- `key_only_random`
- `full_kv`
- `asym_q8_turbo4`
- `asym_q8_turbo3`
- `multiscreen_relevance`
- `key_only_block_so8_triality_vector`

### Required artifact roots

- Capture root: `artifacts\kv_rtx3060_qwen9b`
- Matrix root: `artifacts\qwen_3060_matrix`
- Rotation root: `artifacts\research_extension\triality_full_train_prod_bf16\rotations`

### Explicitly deferred

- RTX 3060 6 GB lane
- Qwen3.5-4B lane
- Gemma runtime work
- Vision / `mmproj`
- Qwen3.5-27B / 31B
- Weight-side TurboQuant `TQ4_1S`
- Online generation integration before offline closure

### Quantization stance

- The repository-wide production K-side research reference remains `key_only_block_so8_triality_vector` from `turboquant/research_extension/k_triality.py`.
- The 12 GB matrix must still report the asymmetric Q8/Turbo4 and Q8/Turbo3 V-first baselines separately from that production K-side research reference.
- Paper-faithful `exact` and `full_kv` rows remain in the matrix as correctness anchors and ablations, not as the recommended 12 GB operating path.

---

## Current State

### Completed foundation

- [x] `turboquant/analysis.py`
  Added `QWEN_3060_MATRIX_MODES`, `evaluate_asymmetric_q8_value_attention_row()`, `evaluate_qwen_3060_matrix_rows()`, and `compute_qwen_3060_multigroup_statistics()`.
- [x] `scripts/validate_qwen_3060_matrix.py`
  Added the 7-mode 12 GB-only matrix orchestrator.
- [x] `turboquant/reporting.py` and `scripts/export_report.py`
  Added `qwen_3060_matrix_*` summary/export paths.
- [x] Vendored runtime and contract surface
  Added runtime mode selection, asymmetric default, triality-vector selection, K/V gating helpers, and non-recurrent Qwen layer gating across the vendored `llama.cpp` files and repo contract validation.
- [x] Focused verification
  `uv run python -m pytest tests\test_reporting.py tests\test_repo_contract.py tests\test_qwen_3060_matrix.py tests\test_attention_metrics.py tests\test_turboquant_prod.py -q`
  and `uv run python scripts\validate_repo_contract.py` were reported green in the handoff.

### Remaining milestone work

- [x] Finish or recover the reduced real matrix run.
- [x] Export the matrix report from completed artifacts.
- [x] Rewrite `README.md` to present the 12 GB Qwen matrix as the mainline flow.
- [x] Verify the vendored runtime build with the repo-standard build path.
- [x] Add a new `_docs` implementation log for this completion slice.
- [x] Run the remaining practical verification for capture / multiscreen / triality replay if time permits.

### Completion evidence

- [x] Reduced real matrix run completed with exit code `0` and wrote the expected bundle under `artifacts\qwen_3060_matrix`.
- [x] `uv run python scripts\export_report.py --matrix-dir artifacts\qwen_3060_matrix` generated `plots\qwen_3060_matrix_attention.png`, `plots\qwen_3060_matrix_runtime.png`, and `reports\qwen_3060_matrix_summary.md`.
- [x] `.\scripts\build_rust_workspace.ps1 -Package hypura -NoCuda` passed after repo-contract validation.
- [x] `uv run python -m pytest tests\test_capture.py tests\test_multiscreen_eval_analysis.py tests\test_triality_eval_recovery.py -q` passed (`20 passed`).

---

## Acceptance Contract

The RTX 3060 12 GB milestone is complete only if all of the following are true:

- `artifacts\kv_rtx3060_qwen9b` is the documented capture root for the 12 GB lane.
- `scripts\validate_qwen_3060_matrix.py` has produced the matrix CSV/Markdown artifacts under `artifacts\qwen_3060_matrix\metrics`.
- `uv run python scripts\export_report.py --matrix-dir artifacts\qwen_3060_matrix` has been run successfully against those artifacts.
- `README.md` documents the 12 GB-only mainline flow and the 7-mode matrix outputs.
- `implementation_plan.md` no longer treats 6 GB / 4B as part of the mainline milestone.
- The vendored runtime build has been exercised via the repo-standard build script.
- A new `_docs` implementation log records commands run, verification evidence, and residual risks.

---

## Task List

### Task 1: Close the active reduced real run

**Description:** Let the currently running `validate_qwen_3060_matrix.py` reduced case finish if it is still making progress; only kill and rerun if there is evidence it has stalled.

**Files / surfaces:**

- Read: `artifacts\qwen_3060_matrix\metrics\*`
- Read: running process state for `uv.exe` / Python worker PIDs

**Acceptance criteria:**

- [ ] The active run is either confirmed progressing or explicitly terminated with a documented reason.
- [ ] On success, the following files exist:
  - `artifacts\qwen_3060_matrix\metrics\qwen_3060_matrix_trials.csv`
  - `artifacts\qwen_3060_matrix\metrics\qwen_3060_matrix_summary.csv`
  - `artifacts\qwen_3060_matrix\metrics\qwen_3060_matrix_mean_pm_sd.csv`
  - `artifacts\qwen_3060_matrix\qwen_3060_matrix_report.md`
- [ ] The run metadata records the 12 GB lane and 7 matrix modes.

**Verification:**

- [ ] `Get-Process -Id 4200,21984,46892`
- [ ] `Get-ChildItem artifacts\qwen_3060_matrix\metrics`
- [ ] If recovery is required, rerun:

```powershell
uv run python scripts\validate_qwen_3060_matrix.py `
  --kv-dir artifacts\kv_rtx3060_qwen9b `
  --rotation-dir artifacts\research_extension\triality_full_train_prod_bf16\rotations `
  --eval-device cuda `
  --bits 3,3.5,4 `
  --trials 3 `
  --max-layers 2 `
  --skip-plots `
  --output-dir artifacts\qwen_3060_matrix
```

### Task 2: Export the 12 GB matrix report bundle

**Description:** Turn the matrix CSV outputs into the final plots and Markdown summary consumed by the repo docs.

**Dependencies:** Task 1

**Files likely touched:**

- Read: `artifacts\qwen_3060_matrix\metrics\*.csv`
- Generate: `artifacts\qwen_3060_matrix\plots\*`
- Generate: `artifacts\qwen_3060_matrix\reports\qwen_3060_matrix_summary.md`

**Acceptance criteria:**

- [ ] `export_report.py --matrix-dir` runs successfully against the matrix artifact root.
- [ ] Plot exports exist for attention and runtime trade-offs.
- [ ] The Markdown summary is generated under `artifacts\qwen_3060_matrix\reports`.

**Verification:**

- [ ] `uv run python scripts\export_report.py --matrix-dir artifacts\qwen_3060_matrix`

### Task 3: Refresh `README.md` for the 12 GB mainline

**Description:** Make the README describe the 12 GB Qwen matrix flow first, not the older generic / mixed-lane framing.

**Dependencies:** Task 2

**Files likely touched:**

- Modify: `README.md`

**Acceptance criteria:**

- [ ] The mainline flow uses `artifacts\kv_rtx3060_qwen9b` and `scripts\validate_qwen_3060_matrix.py`.
- [ ] The README explicitly names the 7-mode matrix and the `asym_q8_turbo4` / `asym_q8_turbo3` comparison.
- [ ] The exported `qwen_3060_matrix` report and plots are linked from the README.
- [ ] 6 GB / 4B content does not appear as a co-equal current milestone.

**Verification:**

- [ ] Manual content check against the generated matrix summary and plots
- [ ] `git diff -- README.md`

### Task 4: Verify the vendored runtime build path

**Description:** Run the repo-standard build flow that validates the repo contract and builds the Rust workspace against the vendored `llama.cpp`.

**Dependencies:** Task 3

**Files / surfaces:**

- Read / execute: `scripts\build_rust_workspace.ps1`
- Read / execute: `scripts\validate_repo_contract.py`
- Build: `rust\`

**Acceptance criteria:**

- [ ] Repo contract validation passes in the same execution path as the build.
- [ ] The build command exits successfully.
- [ ] Any skipped CUDA or target-dir deviations are documented.

**Verification:**

- [ ] `.\scripts\build_rust_workspace.ps1 -Package hypura -NoCuda`

### Task 5: Record the completion slice in `_docs`

**Description:** Add a new implementation log that captures the matrix completion, docs refresh, build evidence, and residual risks.

**Dependencies:** Task 4

**Files likely touched:**

- Add: `_docs\YYYY-MM-DD_<topic>_hub_Qwen3.5-9B-SOT.md`

**Acceptance criteria:**

- [ ] The log includes Overview, Background / requirements, Assumptions / decisions, Changed files, Implementation details, Commands run, Test / verification results, Residual risks, and Recommended next actions.
- [ ] The log reflects the actual commands and outcomes from this completion slice.

**Verification:**

- [ ] Manual review of the saved Markdown log

### Task 6: Optional follow-on verification

**Description:** Run the remaining practical tests that were explicitly called out in the handoff after the mainline closure items are done.

**Dependencies:** Task 5

**Files / surfaces:**

- Test: `tests\test_capture.py`
- Test: `tests\test_multiscreen_eval_analysis.py`
- Test: `tests\test_triality_eval_recovery.py`

**Acceptance criteria:**

- [ ] These tests are either run successfully or explicitly documented as skipped with a reason.

**Verification:**

- [ ] `uv run python -m pytest tests\test_capture.py tests\test_multiscreen_eval_analysis.py tests\test_triality_eval_recovery.py -q`

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| The active reduced real run takes too long or stalls without partial output | High | Confirm progress via process CPU and only rerun if clearly stalled |
| README drifts from the actual exported matrix numbers | High | Export reports first, then update README directly from generated artifacts |
| Vendored runtime build fails despite repo-contract green | Medium | Use the repo-standard build script so contract validation and build happen in one path |
| 6 GB / 4B scope leaks back into the milestone narrative | Medium | Keep all 6 GB / 4B work explicitly deferred in this plan and the README |
| Existing dirty state in `artifacts/`, `vendor/llama.cpp`, or reports is accidentally reverted | High | Make narrow edits only and do not reset unrelated files |

---

## Completion Checklist

- [x] Matrix run artifacts present under `artifacts\qwen_3060_matrix\metrics`
- [x] `export_report.py --matrix-dir` executed successfully
- [x] `README.md` updated to 12 GB-only mainline wording
- [x] `implementation_plan.md` reflects the 12 GB-only milestone
- [x] Vendored runtime build verified
- [x] New `_docs` implementation log added
- [x] Remaining verification either run or explicitly deferred with evidence
