# Overview

Updated the nested `zapabob/llama.cpp` submodule pointer so Turboquant-CUDA can
be used as a clone-reproducible release pin inside the Triality/Hypura stack.

# Background / Requirements

- Keep the release line on the zapabob three-repo stack:
  `Turboquant-CUDA`, `llama.cpp`, and `hypura`.
- Avoid a dirty or implicit nested submodule pointer inside
  `vendor/turboquant-cuda` when Hypura vendors Turboquant-CUDA.
- Align the nested runtime foundation with the parent Triality stack's verified
  `llama.cpp` commit.

# Assumptions / Decisions

- The correct nested target is
  `a020899959e5ad83ace83fca3042b6a47b7153c4`, the same
  `zapabob/llama.cpp` commit used by the parent Triality stack for TurboQuant
  weight payload validation.
- This change only updates the nested submodule pointer and does not alter
  Turboquant-CUDA source behavior.

# Changed Files

- `zapabob/llama.cpp`
- `_docs/2026-05-21_nested-llama-runtime-pin_Codex.md`

# Implementation Details

- Initialized the nested `zapabob/llama.cpp` submodule.
- Fetched `origin/codex/weight-payload-v1-validation`.
- Checked out `a020899959e5ad83ace83fca3042b6a47b7153c4`.
- Staged the nested submodule pointer update for this Turboquant-CUDA branch.

# Commands Run

```powershell
git submodule update --init zapabob/llama.cpp
git -C zapabob/llama.cpp fetch origin codex/weight-payload-v1-validation
git -C zapabob/llama.cpp checkout a020899959e5ad83ace83fca3042b6a47b7153c4
git status --short --branch
git diff --submodule=log
```

# Test / Verification Results

- `git -C zapabob/llama.cpp rev-parse HEAD`:
  `a020899959e5ad83ace83fca3042b6a47b7153c4`.
- `git -C zapabob/llama.cpp branch -r --contains HEAD` includes
  `origin/codex/weight-payload-v1-validation`.

# Residual Risks

- Full parent stack verification is expected to run after Hypura and the
  Triality parent pins are updated to this Turboquant-CUDA commit.

# Recommended Next Actions

- Commit and push this Turboquant-CUDA branch.
- Update Hypura's `vendor/turboquant-cuda` pointer to the new commit.
