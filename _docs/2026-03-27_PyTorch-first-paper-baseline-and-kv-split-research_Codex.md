# 2026-03-27 PyTorch-first paper baseline + K/V-split research implementation

## Summary

- Added `turboquant.paper_baseline` as the PyTorch-only paper-faithful entrypoint.
- Added `turboquant.research_extension` for K/V-separated research APIs and wrappers.
- Added `turboquant.adapters.hf_qwen` as an explicit optional adapter namespace.
- Moved HF/Qwen dependencies in `pyproject.toml` to the optional `hf_qwen` extra.
- Added new CLI entrypoints:
  - `scripts/paper_validate_synthetic.py`
  - `scripts/paper_validate_attention.py`
  - `scripts/research_validate_v_codecs.py`
  - `scripts/research_value_sensitivity.py`
- Updated `README.md` to make the repo PyTorch-first and to demote Qwen replay to a secondary adapter path.

## Verification

- `uv run python -m pytest -q`
- `uv run python scripts/paper_validate_synthetic.py --trials 1 --dim 32 --num-vectors 64 --num-pairs 64`
- `uv run python scripts/paper_validate_attention.py --trials 1 --synthetic-layers 1 --batch 1 --heads 1 --seq-len 16 --head-dim 32 --bits 2,3`
- `uv run python scripts/research_validate_v_codecs.py --query-source synthetic --trials 1 --synthetic-layers 1 --batch 1 --heads 1 --seq-len 16 --head-dim 32 --bits 2,3 --output-dir artifacts/research_extension/metrics_smoke`
- `uv run python scripts/research_value_sensitivity.py --trials 1 --synthetic-layers 1 --batch 1 --heads 1 --seq-len 16 --head-dim 32 --bits 2 --protection-grid-layer-limit 1`
