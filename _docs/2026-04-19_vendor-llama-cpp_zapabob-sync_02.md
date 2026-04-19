# 2026-04-19 vendor/llama.cpp zapabob sync 02

## Overview

- Updated the `vendor/llama.cpp` submodule from `745e347319bf82c3202a9d3dd0feaf33f7c437d8`
  to `9276d42f45b152acfeead86d624cb43e062b5a5a`.
- Aligned the vendored tree to the current `zapabob/llama.cpp` `origin/master`
  after confirming that `master` is the remote default branch and is ahead of the
  previously inspected local clone.

## Why this sync was taken

- `origin/master` now includes the Triality/TurboQuant integration path already
  merged through `origin/main`, plus newer fork-specific fixes.
- The latest fork delta beyond the previously known `fe66971d2` was:
  - `0c0d6ae69329ff5000ac0896ea193cbd45940c7a`
  - Message: `fix: ignore auxiliary triality tensors in loader accounting`
- That loader-accounting fix is directly relevant to Triality metadata handling in
  this repository, so the submodule was advanced all the way to the current
  `origin/master` merge commit instead of stopping at an older local snapshot.

## Effective submodule delta

- `feat(turboquant): default TrialitySO8 rotation`
- `merge: upstream ggerganov/llama.cpp master; keep Triality/TurboQuant API-aligned`
- `docs: align main with fork runtime integrations`
- `docs: redact usernames in main docs`
- `merge: sync zapabob main and master`
- `Merge pull request #7 from zapabob/codex/triality-platform-sync`
- `feat: add real-model triality metadata coverage`
- `fix: ignore auxiliary triality tensors in loader accounting`
- `Merge pull request #8 from zapabob/codex/ignore-triality-aux-tensors`

## Verification

- Build:
  - `cmake --build vendor/llama.cpp/build --config Release --target test-quantize-fns test-turboquant-artifact test-turboquant-gguf-metadata llama-turboquant`
- Tests:
  - `ctest --test-dir vendor/llama.cpp/build -C Release -R "^(test-quantize-fns|test-turboquant-artifact|test-turboquant-gguf-metadata)$" --output-on-failure`
  - Result: `100% tests passed, 0 tests failed out of 3`
- TurboQuant smoke:
  - `vendor/llama.cpp/build/bin/Release/llama-turboquant.exe train --out ... --head-dim 128 --vecs 64 --seed 7 --so8-learned 1 --triality 1`
  - `vendor/llama.cpp/build/bin/Release/llama-turboquant.exe eval --artifact ... --vecs 64 --seed 7`
  - Output:
    - `triality_exact_mse=0.99802417`
    - `triality_proxy_mse=0.54189771`
    - `relative_mse_reduction=0.45702946`

## Verification gap

- `uv run python scripts\validate_repo_contract.py` did not complete in the current
  environment because the active `uv` environment is missing PyTorch:
  - `ModuleNotFoundError: No module named 'torch.nn'`
- This is an environment issue rather than a source-level regression observed from
  the submodule sync itself.

## Notes

- A direct foreground MSBuild invocation from the shell sometimes returned early in
  this desktop session without preserving a useful failure line. Retrying with
  smaller target batches and log capture completed successfully.
