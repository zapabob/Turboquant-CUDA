# 2026-04-19 zapabob llama.cpp path rename

## Overview

- Renamed the repository-local zapabob llama.cpp submodule path from
  `vendor/llama.cpp` to `zapabob/llama.cpp`.
- Updated current code, configuration, README, and CLAUDE guidance so the new
  canonical runtime path is `zapabob/llama.cpp`.
- Kept `vendor/llama.cpp` path suffixes as compatibility fallbacks in a small
  number of resolver locations to avoid breaking older mirrors and copied
  workspaces immediately.

## Background / requirement

- The user asked to make `vendor/llama.cpp` become `zapabob/llama.cpp`.
- The workspace already treats `https://github.com/zapabob/llama.cpp.git` as the
  upstream runtime source of truth; this change makes the on-disk path match that
  ownership/name explicitly.

## Assumptions / decisions

- Historical `_docs` logs were left untouched so they continue to describe the
  repo state that existed when those logs were written.
- Current operational files were updated to the new canonical path.
- Fallback references to `../vendor/llama.cpp` and `../../vendor/llama.cpp` were
  intentionally retained in:
  - `repo_contract.toml`
  - `rust/hypura-sys/build.rs`
  This preserves compatibility with older mirrors while making
  `zapabob/llama.cpp` the primary lookup target.

## Changed files

- `.gitmodules`
- `CLAUDE.md`
- `README.md`
- `repo_contract.toml`
- `rust/hypura-sys/build.rs`
- `rust/kobold_gguf_gui/Cargo.toml`
- `rust/kobold_gguf_gui/src/main.rs`
- `studio-web/src/tabs/RuntimeTab.tsx`
- `studio-web/src/tabs/ServeTab.tsx`
- `tests/test_eval_scripts.py`
- `tests/test_runtime_eval.py`
- `tests/test_studio_api.py`
- `turboquant/gguf_profiles.py`
- `turboquant/repo_contract.py`

## Implementation details

- Moved the git submodule worktree to `zapabob/llama.cpp`.
- Updated `.gitmodules` section naming and path so the declared submodule path is
  now `zapabob/llama.cpp`.
- Updated current docs and UI defaults to point at
  `zapabob/llama.cpp/build/bin/Release/...`.
- Updated Rust resolver logic so it now searches:
  1. `../zapabob/llama.cpp`
  2. `../../zapabob/llama.cpp`
  3. older `vendor/...` fallbacks
- Updated repo contract validation to treat `zapabob/llama.cpp` as the canonical
  vendored path while still allowing legacy fallback suffixes for compatibility
  resolution.
- Updated GGUF helper discovery to search the new `zapabob/llama.cpp/gguf-py`
  location first.

## Commands run

```powershell
git mv vendor/llama.cpp zapabob/llama.cpp
git submodule sync -- "zapabob/llama.cpp"
git grep -n "vendor/llama\.cpp"
git -C "zapabob/llama.cpp" status --short --branch
git -C "zapabob/llama.cpp" rev-parse --short HEAD
```

Additional direct validation:

```powershell
@'
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("repo_contract_mod", Path("turboquant/repo_contract.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
contract = mod.load_repository_contract(Path("."))
print(contract.paths.vendored_llama_cpp.as_posix())
print(contract.rust.llama_cpp_candidate_suffixes)
print(mod.validate_gitmodules(contract))
print(mod.validate_vendor_remote(contract))
print(mod.resolve_llama_cpp_checkout(contract).as_posix())
'@ | .venv\Scripts\python.exe -
```

## Test / verification results

- `git -C zapabob/llama.cpp rev-parse --short HEAD` returned `9276d42f4`
- direct repo-contract validation reported:
  - canonical path: `zapabob/llama.cpp`
  - no `.gitmodules` errors
  - no vendored-remote errors
  - resolved checkout: `.../zapabob/llama.cpp`
- `git grep -n "vendor/llama\.cpp"` now shows current non-historical references
  only in the intentional compatibility fallbacks:
  - `repo_contract.toml`
  - `rust/hypura-sys/build.rs`

## Residual risks

- Full `pytest` / `uv run python scripts\validate_repo_contract.py` verification
  is still blocked by the current local Python environment missing `torch.nn`.
- Historical `_docs` files still mention `vendor/llama.cpp` by design; this is
  expected archival context, not a live config mismatch.
- The submodule gitdir still lives under `.git/modules/vendor/llama.cpp`, which
  is acceptable for Git internals after a path move but may look surprising.

## Recommended next actions

1. Repair the local Python environment so `torch.nn` is importable again.
2. Re-run `uv run python scripts\validate_repo_contract.py`.
3. If desired, normalize the submodule internals further by fresh-cloning or
   re-initializing the renamed submodule after the current work is committed.
