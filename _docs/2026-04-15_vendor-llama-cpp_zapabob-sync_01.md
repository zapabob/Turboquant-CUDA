# Vendor llama.cpp zapabob Sync Implementation Log

## Overview

Compared the current `vendor/llama.cpp` worktree against the latest fetched `zapabob/llama.cpp` (`origin/master`) and pulled in the largest buildable subset around TurboQuant, quantization, GGUF metadata tests, and the dependency closure required for Gemma4-related upstream model changes.

## Background / requirements

- User request: compare current `zapabob/llama.cpp` and `Vendor/llama.cpp`, and take the `zapabob/llama.cpp` implementation as canonical.
- Project rules: correctness and reproducibility first, small explicit patches, Windows-native workflow, and implementation logging under `_docs`.
- Repo-local build contract: `vendor/llama.cpp` must remain aligned with `zapabob/llama.cpp`, and `scripts\\validate_repo_contract.py` should pass before claiming build-state health.

## Assumptions / decisions

- Did not attempt a full 98-commit wholesale vendor upgrade across all 284 upstream-changed files in one shot because this repo has downstream Rust/Python integration expectations and the task asked to take in "as much as possible," not to force a blind full-tree rewrite.
- Focused on the TurboQuant / quantize / GGUF-metadata path first, then expanded only as far as necessary to restore a buildable upstream dependency closure.
- Treated upstream `tests/test-quant-type-selection.cpp` as part of the desired sync, but could not execute it in this environment because OpenSSL is missing and the corresponding CMake target is conditionally disabled.
- Removed the local `test-turboquant-*` CMake hook after verification showed those local-only tests no longer matched the upstream `llama-turboquant` API shape. The files remain present and untracked; they are simply no longer part of the build graph.
- Used a temporary build directory on `C:` because drive `H:` had `0 GB` free and the existing in-repo build failed with an MSBuild out-of-disk error.

## Changed files

- Synced to upstream `origin/master`:
  - `vendor/llama.cpp/src/CMakeLists.txt`
  - `vendor/llama.cpp/src/llama-arch.cpp`
  - `vendor/llama.cpp/src/llama-arch.h`
  - `vendor/llama.cpp/src/llama-hparams.h`
  - `vendor/llama.cpp/src/llama-model.cpp`
  - `vendor/llama.cpp/src/llama-model.h`
  - `vendor/llama.cpp/src/llama-quant.cpp`
  - `vendor/llama.cpp/src/llama-turboquant.cpp`
  - `vendor/llama.cpp/src/llama-turboquant.h`
  - `vendor/llama.cpp/src/models/gemma-embedding.cpp`
  - `vendor/llama.cpp/src/models/gemma3.cpp`
  - `vendor/llama.cpp/src/models/gemma3n-iswa.cpp`
  - `vendor/llama.cpp/src/models/models.h`
  - `vendor/llama.cpp/src/models/gemma4-iswa.cpp`
  - `vendor/llama.cpp/tests/CMakeLists.txt`
  - `vendor/llama.cpp/tests/gguf-model-data.cpp`
  - `vendor/llama.cpp/tests/gguf-model-data.h`
  - `vendor/llama.cpp/tests/test-gguf-model-data.cpp`
  - `vendor/llama.cpp/tests/test-quant-type-selection.cpp`
  - `vendor/llama.cpp/tools/quantize/quantize.cpp`
  - `vendor/llama.cpp/tools/turboquant/turboquant.cpp`
- Verified but left as pre-existing local upstream-aligned changes:
  - `vendor/llama.cpp/include/llama.h`
  - `vendor/llama.cpp/src/llama-ext.h`
  - `vendor/llama.cpp/src/llama-model-loader.cpp`

## Implementation details

- Fetched `vendor/llama.cpp` upstream state from `origin/master`.
- Classified overlap between current local vendor edits and upstream changes.
- Identified that several files were already effectively upstream-aligned even though the submodule base commit is older.
- Replaced the selected source/test/tool files with their `origin/master` contents.
- Expanded the sync to Gemma4-related arch/model dependency files after the first build showed unresolved upstream symbols from `src/llama-model.cpp`.
- Restored `tests/CMakeLists.txt` to the upstream version after a local merge variant incorrectly kept incompatible local TurboQuant tests in the build graph.
- Left untracked local files `tests/test-turboquant-artifact.cpp` and `tests/test-turboquant-gguf-metadata.cpp` on disk but unreferenced by CMake.

## Commands run

```powershell
git -C .\vendor\llama.cpp fetch origin
git -C .\vendor\llama.cpp diff --name-only HEAD..origin/master
git -C .\vendor\llama.cpp diff --stat HEAD..origin/master
cmake -S .\vendor\llama.cpp -B C:\Users\downl\AppData\Local\Temp\llama-codex-build-zapabob-sync -G "Visual Studio 18 2026" -A x64 -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF
cmake --build C:\Users\downl\AppData\Local\Temp\llama-codex-build-zapabob-sync --config Release --target test-quantize-fns llama-quantize llama-turboquant
ctest --test-dir C:\Users\downl\AppData\Local\Temp\llama-codex-build-zapabob-sync -C Release -R '^test-quantize-fns$' --output-on-failure
C:\Users\downl\AppData\Local\Temp\llama-codex-build-zapabob-sync\bin\Release\llama-turboquant.exe train --out C:\Users\downl\AppData\Local\Temp\codex-llama-turboquant-artifact.tq --head-dim 128 --vecs 64 --seed 7 --so8-learned 1 --triality 1
C:\Users\downl\AppData\Local\Temp\llama-codex-build-zapabob-sync\bin\Release\llama-turboquant.exe eval --artifact C:\Users\downl\AppData\Local\Temp\codex-llama-turboquant-artifact.tq --vecs 64 --seed 7
uv run python scripts\validate_repo_contract.py
```

## Test / verification results

- `cmake --build ... --target test-quantize-fns llama-quantize llama-turboquant`: passed.
- `ctest --test-dir ... -R '^test-quantize-fns$' --output-on-failure`: passed.
  - Result: `1/1` tests passed, `0` failed.
- `llama-turboquant.exe train ...`: passed.
  - Output included `saved artifact: ...codex-llama-turboquant-artifact.tq`.
- `llama-turboquant.exe eval ...`: passed.
  - Output included:
    - `triality_exact_mse=0.99802417`
    - `triality_proxy_mse=0.54189771`
    - `relative_mse_reduction=0.45702946`
- `uv run python scripts\validate_repo_contract.py`: passed.
  - Output: `Repository contract OK.`

## Residual risks

- `vendor/llama.cpp` is still not a full-tree sync to current `origin/master`; many unrelated upstream changes remain unapplied by design.
- `tests/test-quant-type-selection.cpp` is now synced from upstream but was not executable here because the build did not find OpenSSL, so the CMake block that defines `gguf-model-data`-dependent tests remained disabled.
- The vendor submodule base commit itself was not advanced; the worktree now contains a selective upstream sync on top of the older checked-out submodule commit.
- Local untracked TurboQuant tests still exist and may need rewrite or removal later if the repo fully standardizes on the upstream API.

## Recommended next actions

- If the goal is full vendor convergence, continue syncing the remaining upstream file set in staged slices rather than all at once.
- Install or point CMake to OpenSSL and then build/run `test-quant-type-selection`.
- Decide whether to rewrite the local `test-turboquant-*` files for the upstream API or remove them entirely.
- If this sync should be preserved as a durable vendor state, advance the submodule with an explicit commit strategy rather than leaving it as a dirty selective sync.
