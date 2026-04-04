# AGENTS.md

## Repository purpose

This repository implements a research-grade TurboQuant prototype for KV-cache
compression experiments on Qwen3.5-9B in a Windows-native Codex workflow.

The priority is correctness and reproducibility first.
Do not optimize for cleverness.
Do not jump to online generation integration before offline validation is
correct.

## Working style

- Plan first for nontrivial changes.
- Keep edits local and explicit.
- Prefer small, reviewable patches.
- Preserve existing file structure unless a structural change is clearly
  justified.
- Do not introduce hidden framework magic.

## Shell and platform rules

- This project targets Windows native execution.
- Use PowerShell-safe commands and assumptions.
- Use `uv` as the canonical environment manager.
- Use Python 3.12.x only for the main development path.
- Do not rely on bash, sed, awk, grep, or GNU-only tools in scripts.
- Use Python for portable logic.
- Use `pathlib` for paths.
- Assume file paths may contain spaces.

## Core technical rules

- All public functions must have type hints.
- Add docstrings to nontrivial public APIs.
- Add shape comments for tensor-heavy code.
- Fail loudly on invalid shapes or unsupported dtypes.
- Never silently move tensors across devices.
- Never silently cast dtype in quantization code.
- Keep math correspondence to the paper visible in comments.

## Quantization-specific rules

- Do not collapse TurboQuant into a plain reconstruction-only quantizer.
- Keep Stage 1 and Stage 2 codepaths distinct.
- Keep exact-score and estimated-score evaluation separate.
- Validate inner-product bias explicitly.
- Report both reconstruction metrics and attention/logit metrics.
- **Production K-side reference (実用正系):** Triality SO(8) proxy with the **vector** view and mode `key_only_block_so8_triality_vector` (`PRODUCTION_K_TURBOQUANT_MODE` in `k_triality.py`); paper random/static modes are ablations unless explicitly chosen.

## Reproducibility rules

Every experiment artifact must include enough metadata to reproduce the run:

- model name
- tokenizer name
- seed
- dtype
- device
- quantization config
- prompt hash
- timestamp
- package versions where practical

## Testing rules

When you change quantization logic, run at minimum:

- unit tests for the touched modules
- synthetic validation script if estimator logic changed

When you change model capture or offline validation logic, run at minimum:

- environment check
- KV capture script
- offline attention validation script on a reduced case if necessary

## Preferred setup

```powershell
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev
```
