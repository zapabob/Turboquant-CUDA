# 2026-03-27 Captured Results Publication Refresh (Codex)

## Summary

Refreshed the public-facing report surfaces after the captured CUDA replay
completed successfully. The goal of this pass was not to change the core
algorithm, but to make the repo's public materials reflect the latest
decision-ready captured results.

## What Changed

- Updated `scripts/export_report.py` so generated `artifacts/reports/summary.md`
  includes a stronger `Captured Replay` section with:
  - explicit runtime recommendation
  - short captured headline bullets
  - representative 2-bit and 4-bit comparisons for:
    - `key_only_block_so8_learned`
    - `protected_v_lowrank`
    - `full_kv`
  - a note that `peak_vram_mb` is replay-side additional CUDA memory, not total
    end-to-end model inference VRAM
- Updated `README.md` to present the repo as a fixed-weight Qwen3.5-9B KV codec
  Pareto comparison rather than a generic synthetic-only TurboQuant prototype
- Added a captured representative runtime table to the README
- Documented the runtime decision rule:
  - keep runtime default as `key-only`
  - do not move to a value-aware branch unless it clearly beats `full_kv` on
    hidden-state metrics and approaches the `key_only_block_so8_learned`
    baseline closely enough to justify complexity

## Captured Results Referenced

The refresh uses the latest captured replay outputs from:

- `artifacts/metrics/attention_summary_captured.csv`
- `artifacts/metrics/attention_summary_captured.md`

Representative values used in the public summary:

- 2-bit `key_only_block_so8_learned`
  - memory ratio `0.5664`
  - hidden cosine `1.0010`
- 2-bit `protected_v_lowrank`
  - memory ratio `0.2122`
  - hidden cosine `0.9648`
- 2-bit `full_kv`
  - memory ratio `0.1309`
  - hidden cosine `0.9404`
- 4-bit `key_only_block_so8_learned`
  - memory ratio `0.6289`
  - hidden cosine `0.9980`
- 4-bit `protected_v_lowrank`
  - memory ratio `0.3308`
  - hidden cosine `0.9980`
- 4-bit `full_kv`
  - memory ratio `0.2559`
  - hidden cosine `0.9951`

## Rationale

Before this refresh, `artifacts/reports/summary.md` and `README.md` lagged
behind the latest captured run. That created a public mismatch:

- the metrics files already showed a clear decision
- the top-level report and repo landing page were still partially anchored to an
  older summary state

This pass resolves that mismatch and makes the public narrative consistent with
the current evidence:

- `key-only` remains the runtime default
- `protected_v` and `protected_v_lowrank` are real middle Pareto points
- value-side compression is still the dominant mathematical bottleneck
