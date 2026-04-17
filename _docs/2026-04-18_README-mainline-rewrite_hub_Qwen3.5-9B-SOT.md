# 2026-04-18 README mainline rewrite

## Scope

- rewrote `README.md` to present the repo as the Windows-native TurboQuant workbench
- moved the document to a shorter public-facing structure:
  - TL;DR
  - why the repo exists
  - current mainline
  - workflow overview
  - quick start
  - current 12 GB matrix snapshot
  - validation policy
  - build contract
  - key scripts
- kept the practical K-side reference explicit: `key_only_block_so8_triality_vector`
- kept the offline-first research policy explicit: Stage 1/Stage 2 separation, exact vs estimated separation, hidden/logit/runtime separation
- kept `TurboQuant Studio` documented as a local operator shell layered over the existing CLI flows

## Notes

- this was a docs-only rewrite; no code paths or artifact contracts were changed
- the tracked README figures continue to point at `_docs/assets/qwen_3060_matrix_attention.png` and `_docs/assets/qwen_3060_matrix_runtime.png`
- follow-up work, if needed, should refine examples and screenshots rather than re-expand the README into a full internal audit log
