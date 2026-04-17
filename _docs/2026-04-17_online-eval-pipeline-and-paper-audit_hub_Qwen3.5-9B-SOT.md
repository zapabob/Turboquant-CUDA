# 2026-04-17 Online Eval Pipeline And Paper Audit

## Scope

- implement the first-pass **HF online** TurboQuant evaluation shim for Qwen3
- implement **runtime-side** command builders / parsers for `perplexity`, `llama-bench`, and `lm-eval`
- add a new `eval` optional dependency extra
- rewrite README sections around:
  - paper claim audit
  - cherry-picking risk
  - Pareto framing
  - online benchmark pipeline / artifact schema

## Main code changes

### Shared statistics

- added [`turboquant/eval_stats.py`](../turboquant/eval_stats.py)
- new helpers:
  - continuous summary with `n`, `mean`, `std`, `sem`, `ci95_low`, `ci95_high`
  - benchmark accuracy summary with Wilson interval
  - `Wilcoxon signed-rank + Holm`
  - exact `McNemar` pairwise test
- [`turboquant/reporting.py`](../turboquant/reporting.py) now delegates its continuous summary path to the shared helper so offline and online reports use the same column family

### HF online adapter

- added [`turboquant/adapters/hf_qwen/online_eval.py`](../turboquant/adapters/hf_qwen/online_eval.py)
- design choice:
  - keep this **Qwen3-only** and **evaluation-only**
  - patch only the attention/cache path needed for online evaluation
  - do not introduce a generic Transformers monkeypatch API
- supported modes in this first pass:
  - `exact`
  - `key_only_random`
  - `full_kv`
  - `asym_q8_turbo4`
  - `key_only_block_so8_triality_vector`
- implementation notes:
  - uses a custom `TurboQuantQwenCache`
  - keeps K-side estimation distinct from plain decode-only reconstruction for TurboQuant-backed modes
  - supports GQA by regrouping query heads against KV heads before score estimation
  - fails loudly on unsupported paths (currently batch size > 1, missing custom cache, beam-search cache ops)

### Runtime-side helpers

- added [`turboquant/runtime_eval.py`](../turboquant/runtime_eval.py)
- includes:
  - `build_llama_perplexity_command`
  - `build_llama_bench_command`
  - `build_lm_eval_command`
  - `parse_perplexity_output`
  - `parse_llama_bench_json`
  - `load_lm_eval_results`
- `lm-eval` summaries normalize stderr-bearing metrics into `mean` + `sem` + `ci95_*` rows when available

### Scripts

- added [`scripts/eval_hf_online_qwen.py`](../scripts/eval_hf_online_qwen.py)
  - dry-run support
  - plain-text chunked PPL evaluation
  - generic multiple-choice JSONL manifest scoring
- added [`scripts/eval_runtime_qwen.py`](../scripts/eval_runtime_qwen.py)
  - dry-run support
  - durable command manifest / stdout / stderr / exit-code logging
  - parser hookup for `perplexity`, `llama-bench`, and `lm-eval`
- added [`scripts/export_online_eval_report.py`](../scripts/export_online_eval_report.py)
  - combines replay summary, HF online summary, and runtime summary
  - emits error-bar figures and memory-join Pareto figures
  - writes `artifacts/online_eval_report/reports/online_eval_summary.md`

### Packaging

- [`pyproject.toml`](../pyproject.toml) gained a new `eval` extra
- `uv.lock` updated accordingly

## README decisions

- use the **paper** and **Google Research blog** only for the original claim envelope
- keep README claims about this repo limited to:
  - local replay evidence already present in `google_blog_audit`, paper-baseline captured replay, and the 3060 matrix
  - new online-eval scripts and artifact schema that are now implemented
- explicitly say that:
  - the reduced 3060 `Friedman` rows are degenerate and not used as omnibus evidence
  - fresh HF/runtime online benchmark artifacts are not yet committed, so README does not overclaim new PPL or task outcomes

## Verification

```powershell
uv run python -m pytest tests\test_reporting.py tests\test_repo_contract.py tests\test_qwen_3060_matrix.py tests\test_eval_stats.py tests\test_hf_online_eval.py tests\test_runtime_eval.py tests\test_eval_scripts.py -q
uv run python scripts\validate_repo_contract.py
git diff --check
```

## Result

- pytest: `24 passed`
- repo contract: `OK`
- `git diff --check`: no whitespace errors; only CRLF normalization warnings on Windows worktree files

## Known follow-up

- no fresh `artifacts/hf_online_eval` or `artifacts/runtime_eval` benchmark run was executed in this change set
- runtime item-level task logging still depends on the exact `lm-eval` sample export shape at execution time
- README online-benchmark sections intentionally describe the pipeline and publication policy, not unrun task results
