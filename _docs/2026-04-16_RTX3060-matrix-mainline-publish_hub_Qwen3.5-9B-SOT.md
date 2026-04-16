# RTX 3060 Matrix Mainline Publish Implementation Log

## Overview

Recorded the mainline publication path for the RTX 3060 12 GB Qwen3.5-9B matrix
closeout after the reduced real run had already been validated and documented.
This follow-up log captures how the closeout was landed onto the zapabob
mainline repos without disturbing existing dirty artifacts, how the vendored
runtime pointer was advanced, and how Windows-native build verification was
completed when the default worktree-local build path failed.

## Why a second log was needed

- The original closeout log focused on experiment completion, report export, and
  doc refresh.
- The remaining operationally important details were still missing:
  - how the superproject and vendored runtime were propagated to zapabob
    mainline
  - why isolated worktrees were used instead of the already-dirty primary
    workspace
  - what failed during build verification and how the final build proof was
    recovered
- Future agents need this publish trail because the repo currently mixes
  long-lived artifacts, a vendored git submodule, and Windows-native build
  constraints.

## Constraints and repository state

- The working tree already contained unrelated dirty or generated state that
  could not be reverted:
  - `artifacts/paper_baseline/*`
  - `artifacts/reports/env_check.txt`
  - `_tmp/`
  - `artifacts/kv_rtx3060_qwen9b/`
  - `artifacts/qwen_3060_matrix/`
  - `artifacts/qwen_3060_matrix_debug/`
  - `rust/target_smoke_ninja/`
- The superproject default branch is `main`, but the vendored
  `vendor/llama.cpp` upstream default branch is `master`.
- The previously pushed feature branches already existed:
  - superproject: `codex/rtx3060-qwen-matrix-closeout`
  - vendored runtime: `codex/rtx3060-qwen-runtime-matrix`
- The requirement was to land the work on the respective zapabob mainlines
  without sweeping unrelated local state into the commit set.

## Mainline publication strategy

- Added `.worktrees/` to the superproject `.gitignore` first so project-local
  worktrees stay untracked and safe.
- Reset local refs to the latest remotes before integration:
  - superproject `main` -> `origin/main`
  - vendored runtime `master` -> `origin/master`
- Created isolated worktrees:
  - superproject mainline worktree:
    `H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\.worktrees\root-main`
  - vendored runtime mainline worktree:
    `H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\.worktrees\vendor-llama-master`
- Cherry-picked the already-reviewed feature commits onto the mainline
  worktrees instead of merging the dirty primary workspace.

## Commit mapping

- Superproject feature-branch commits that carried the closeout:
  - `a9ed231` `feat: add qwen 3060 matrix closeout workflow`
  - `c018a26` `chore: ignore local worktree directories`
- Superproject mainline commits after cherry-pick + gitlink update:
  - `f1ac267` `chore: ignore local worktree directories`
  - `638a5a7` `feat: add qwen 3060 matrix closeout workflow`
  - `92f1992` `chore: update vendored llama.cpp runtime commit`
- Vendored runtime feature-branch commit:
  - `bee2109cf` `feat: add qwen turboquant runtime mode gating`
- Vendored runtime mainline commit after cherry-pick:
  - `745e34731` `feat: add qwen turboquant runtime mode gating`

## What was actually done

- In `vendor-llama-master`, cherry-picked `bee2109cf` onto `master` and pushed
  it to `origin/master`.
- In `root-main`, cherry-picked `c018a26` and `a9ed231` onto `main`.
- Initialized the submodule inside `root-main`, fetched the updated vendored
  runtime, checked out `745e34731`, then committed the resulting gitlink update
  as `92f1992`.
- Pushed `root-main` to `origin/main` only after fresh verification in that
  isolated worktree.

## Build verification failure and recovery

- The repo-standard build script succeeded earlier in the primary workspace, but
  re-running it inside `root-main` exposed a worktree-local environment issue:
  `uv` created a fresh `.venv` in the worktree and then failed contract
  validation because that minimal environment did not contain `torch`.
- After removing the temporary worktree-local `.venv`, Python-based validation
  and pytest were re-run using the parent repo interpreter:
  `H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\.venv\Scripts\python.exe`
- A direct `cargo build -p hypura -j 1` inside the worktree initially failed
  with Windows `os error 5` when Cargo tried to execute the build script from
  the worktree-local target directory.
- The final successful workaround was to keep the `root-main` source tree but
  move `CARGO_TARGET_DIR` onto the C drive:
  `C:\_codex_targets\qwen9b_root_main`
- That C-drive target build finished successfully and became the build proof
  used for mainline publication.

## Commands run

```powershell
git remote show origin
git -C vendor/llama.cpp remote show origin
git branch -f main origin/main
git -C vendor/llama.cpp branch -f master origin/master
git worktree add '.worktrees/root-main' main
git -C vendor/llama.cpp worktree add H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\.worktrees\vendor-llama-master master
git cherry-pick bee2109cf95444d57799b81ccbe7982eaf83d72d
git push origin master
git cherry-pick c018a26
git cherry-pick a9ed231
git submodule update --init vendor/llama.cpp
git -C vendor/llama.cpp fetch origin
git -C vendor/llama.cpp checkout 745e34731
git add vendor/llama.cpp
git commit -m "chore: update vendored llama.cpp runtime commit"
Remove-Item -LiteralPath '.venv' -Recurse -Force
H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\.venv\Scripts\python.exe scripts\validate_repo_contract.py
H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\.venv\Scripts\python.exe -m pytest tests\test_reporting.py tests\test_repo_contract.py tests\test_qwen_3060_matrix.py tests\test_attention_metrics.py tests\test_turboquant_prod.py tests\test_capture.py tests\test_multiscreen_eval_analysis.py tests\test_triality_eval_recovery.py -q
$env:LLAMA_CPP_DIR = (Resolve-Path '.\vendor\llama.cpp').Path
$env:HYPURA_LLAMA_CPP_DIR = $env:LLAMA_CPP_DIR
$env:HYPURA_NO_CUDA = '1'
$env:CARGO_TARGET_DIR = 'C:\_codex_targets\qwen9b_root_main'
cargo build -p hypura -j 1
git push origin main
```

## Verification evidence

- `scripts\validate_repo_contract.py`
  - passed in `root-main`
  - repo contract reported `OK`
- Pytest in `root-main`
  - command covered:
    - `tests/test_reporting.py`
    - `tests/test_repo_contract.py`
    - `tests/test_qwen_3060_matrix.py`
    - `tests/test_attention_metrics.py`
    - `tests/test_turboquant_prod.py`
    - `tests/test_capture.py`
    - `tests/test_multiscreen_eval_analysis.py`
    - `tests/test_triality_eval_recovery.py`
  - result: `33 passed, 1 warning`
- Rust build in `root-main`
  - command: `cargo build -p hypura -j 1`
  - environment:
    - `HYPURA_NO_CUDA=1`
    - `LLAMA_CPP_DIR` / `HYPURA_LLAMA_CPP_DIR` pointed at the `root-main`
      vendored checkout
    - `CARGO_TARGET_DIR=C:\_codex_targets\qwen9b_root_main`
  - result: success, `Finished 'dev' profile [unoptimized + debuginfo]`
- Final published refs:
  - superproject `origin/main` -> `92f1992`
  - vendored runtime `origin/master` -> `745e34731`

## Residual notes for future agents

- If a future verification run is required from a worktree on this machine,
  prefer the parent repo Python interpreter or a fully synchronized `uv` env.
  A minimal worktree-local `.venv` is not enough for this repo because the
  validation path imports `torch`.
- If Cargo build scripts fail under a worktree-local target directory on this
  Windows host with `os error 5`, retry with a short C-drive `CARGO_TARGET_DIR`
  before assuming the source tree is broken.
- The publication path intentionally avoided touching experiment artifacts.
  Reproducibility evidence lives in the artifact directories and `_docs`; git
  history only carries the code, docs, and submodule pointer updates.
