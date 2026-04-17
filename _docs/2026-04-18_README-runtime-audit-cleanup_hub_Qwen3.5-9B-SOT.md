# 2026-04-18 README Runtime Audit Cleanup

## Scope

- rewrite `README.md` so the online-benchmark section reflects the **actual** current-main runtime audit outcome
- keep runtime/eval pipeline code in scope
- keep unrelated `studio` prototype files out of the commit scope
- return the primary worktree to a clean `git status`

## README change

The previous README wording was still pipeline-oriented:

- online eval code existed
- future HF/runtime artifacts were expected
- publication was deferred in broad terms

That was no longer the best summary after the current-main audit closed.

The rewritten README now makes four distinctions explicit:

1. paper/blog claim envelope
2. replay / HF diagnostic evidence
3. authoritative `zapabob/llama.cpp` current-main runtime audit evidence
4. unclaimable areas that should **not** be overstated yet

## Current-main runtime stance now recorded in README

- `zapabob/llama.cpp` current `main` is treated as the only runtime authority for this phase
- reduced exact-baseline runtime evaluation completed
- `LLAMA_TURBOQUANT=1` smoke completed
- fresh `Q8_0 -> .turboquant.gguf` repack + readback completed as secondary validation
- README still does **not** claim runtime TurboQuant superiority

Reasons kept explicit:

- current-main env knobs are accepted/logged, but SO8 / triality runtime helper math is not yet exposed as a claimable mode-selectable compute branch
- current-main `llama-server` does not provide prompt `token_logprobs` needed for stock `lm-eval` multiple-choice loglikelihood tasks
- pairwise p-values therefore stay unavailable rather than being synthesized from non-claimable runs

## Scope cleanup

The worktree contained unrelated `studio` prototype files alongside the runtime/eval work.

Cleanup policy:

- keep runtime/eval code, tests, docs, and README changes in the tracked commit scope
- keep generated runtime/HF artifacts local-only
- keep `studio` prototype files local-only
- keep the main worktree clean without deleting local prototype work

This was done by:

- removing `studio`-only dependency leakage from `pyproject.toml`
- regenerating `uv.lock`
- adding local excludes for generated artifacts and unrelated prototype directories under `.git/info/exclude`

## Verification

```powershell
uv run python -m pytest tests\test_eval_stats.py tests\test_runtime_eval.py tests\test_eval_scripts.py tests\test_reporting.py tests\test_repo_contract.py -q
uv run python scripts\validate_repo_contract.py
git diff --check
git status --short
```

## Result

- runtime/eval tracked scope remains intact
- README now matches the current-main audit state
- unrelated local prototype work stays preserved but hidden from the primary worktree status
