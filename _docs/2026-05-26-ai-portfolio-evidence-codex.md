# 2026-05-26 AI Portfolio Evidence Refresh - Codex

## Goal

Review the README from an AI-engineering portfolio perspective and make the proof surface explicit for readers evaluating model, dataset, metrics, repro, and limitations.

## Review Finding

- The README already had strong experimental evidence, but the top block did not expose a compact reviewer-facing evidence card.
- No quantization or runtime code was changed.

## Files Changed

- `README.md`
- `_docs/2026-05-26-ai-portfolio-evidence-codex.md`

## Verification

- Documentation-only change.
- Confirmed the new evidence card references existing README evidence families and quickstart commands.

## Remaining Risk

- The README still depends on artifact paths under `artifacts/`; future work should ensure any externally referenced figures and CSV summaries stay present in public clones.
