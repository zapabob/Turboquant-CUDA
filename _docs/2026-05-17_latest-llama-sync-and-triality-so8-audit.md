## Overview

Reflected the latest `zapabob/llama.cpp` `master` line into the vendored
submodule and added a reproducible Triality learned SO(8) rotation audit for
the public vector, spinor-plus proxy, and spinor-minus proxy views.

## Background / Requirements

- The user asked to reflect the latest `zapabob/llama.cpp` features.
- The user also asked to measure per-bit Triality SO(8) rotations for vector
  and spinor +/- views, verify orthogonality and determinant stability, prevent
  outliers, and publish the result in the README with figures.
- Current upstream checks on 2026-05-17 showed:
  - `zapabob/llama.cpp` `main`: `29f446a68`
  - `zapabob/llama.cpp` `master`: `31b900be6`
  - `master` includes the May 17 upstream sync and TheTom TurboQuant KV-cache
    sync line, so it is the latest feature line for this request.

## Assumptions / Decisions

- Use `origin/master` for the vendored `llama.cpp` reflection because it is
  newer than `origin/main` and contains the latest TurboQuant merge series.
- Audit existing learned rotation artifacts instead of retraining them.
- Treat the saved `bfloat16` rotation tensors with a fail-closed validity
  budget of `1e-2` for both effective orthogonality drift and determinant drift.
- Audit the effective view-composed 8x8 blocks, not only the learned rotation
  tensor, so the vector/spinor proxy adapters are included in the validation.

## Changed Files

- `README.md`
- `scripts/audit_triality_so8_rotations.py`
- `tests/test_triality_so8_audit.py`
- `_docs/assets/2026-05-17-triality-so8-audit/*`
- `zapabob/llama.cpp` submodule pointer

## Implementation Details

- Added `scripts/audit_triality_so8_rotations.py`.
- The script reads
  `artifacts/research_extension/triality_full_train_prod_bf16/metrics/triality_rotation_manifest.csv`.
- For every layer/bit/view/block, it computes:
  - learned block orthogonality error
  - learned block determinant drift
  - effective view-composed block orthogonality error
  - effective view-composed block determinant drift
  - z-score based outlier indicators
- The script writes:
  - detail CSV
  - summary CSV
  - README-ready markdown
  - status JSON
  - orthogonality PNG
  - determinant PNG
- Added a README concept visual at
  `_docs/assets/2026-05-17-triality-so8-audit/triality_so8_gptimage_style_concept.png`.
- The README now includes:
  - latest `zapabob/llama.cpp` sync note
  - compact worst-case Triality SO(8) audit table
  - links to generated artifacts
  - two generated figures

## Commands Run

```powershell
git -C zapabob/llama.cpp fetch origin main master --prune
git -C zapabob/llama.cpp checkout --detach origin/master
```

```powershell
uv run python -m pytest tests\test_triality_so8_audit.py -q
uv run python scripts\audit_triality_so8_rotations.py
```

## Test / Verification Results

- `tests\test_triality_so8_audit.py`: `2 passed`
- Triality SO(8) audit:
  - status: `pass`
  - rows audited: `4608`
  - outliers: `0`
  - max effective orthogonality error: `0.00626921933144331`
  - max effective determinant error: `0.008674837823614001`
  - orthogonality threshold: `0.01`
  - determinant threshold: `0.01`

## Residual Risks

- The audit validates existing saved `bfloat16` learned SO(8) artifacts; it does
  not retrain rotations or prove downstream generation quality by itself.
- `master` is intentionally newer than `main` in `zapabob/llama.cpp` for this
  request. Future release work should re-check branch roles before publishing.
- The README figures are generated locally from tracked artifacts. If the
  training manifest changes, the audit script should be rerun before updating
  README claims.

## Recommended Next Actions

- Run the broader focused suite before commit or PR packaging:
  `uv run python -m pytest tests\test_triality_contract.py tests\test_turboquant_gguf_profiles.py tests\test_triality_so8_audit.py -q`
- Rebuild local CUDA `llama-server` only if this submodule pin needs to be used
  as the installed runtime on this PC.
