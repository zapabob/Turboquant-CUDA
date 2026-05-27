# 2026-05-27 Dependabot lock refresh - Codex

## Goal

Resolve the open Dependabot alert for the locked `idna` package without changing TurboQuant runtime code.

## Changes

- Refreshed `uv.lock`.
- Updated `idna` from `3.11` to `3.16`.

## Verification

- `uv lock --check`: passed.
- `uv lock --dry-run`: no lockfile changes detected.
- `uv tree --package idna --invert`: resolved `idna v3.16`.
