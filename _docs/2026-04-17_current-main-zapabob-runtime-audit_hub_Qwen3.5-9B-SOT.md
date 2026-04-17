# 2026-04-17 Current-main zapabob Runtime Audit

## Scope

- treat `C:\Users\downl\Desktop\llama.cpp-zapabob` `main` as the only runtime authority for this phase
- build and verify:
  - `llama-perplexity.exe`
  - `llama-bench.exe`
  - `llama-server.exe`
- run a reduced runtime baseline on:
  - `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`
- keep README publication deferred
- write authoritative findings to artifacts and `_docs`, not to README

## Why README stayed deferred

- current external `main` does accept `LLAMA_TURBOQUANT*` env knobs and logs them
- however the audited source still does **not** wire SO8 / triality helper functions into a selectable runtime compute branch
- `llama-server` current-main OpenAI completions also does **not** expose prompt `token_logprobs`, so `lm-eval` multiple-choice loglikelihood tasks are not claimable through the stock API surface
- because of that, this phase closes as:
  - build + harness + baseline validation
  - current-main limitation audit
  - not a publishable TurboQuant runtime superiority claim

## Runtime authority and build evidence

- authoritative checkout:
  - `C:\Users\downl\Desktop\llama.cpp-zapabob`
- audited commit:
  - `fa5aa4d3788a640d83e7c35869d9feb623cbfbd5`
- binary paths recorded in:
  - `artifacts/runtime_eval/current_main/runtime_code_audit.json`
- CUDA build succeeded with:
  - preset `x64-windows-msvc-release`
  - `GGML_CUDA=ON`
  - explicit `CMAKE_ASM_COMPILER=...\\ml64.exe`
- device visibility confirmed with `llama-bench --list-devices`
  - RTX 3060 detected by GGML CUDA

## Code / script changes in this repo

- `turboquant/runtime_eval.py`
  - added current-main runtime profile resolution
  - added `llama-server` readiness wait
  - added runtime code audit helpers
  - added local-completions prompt-logprob probe
  - made `lm-eval` command building use the active interpreter instead of bare `python`
  - made perplexity parsing robust to current-main inline `[1]x,[2]y,` output
- `scripts/eval_runtime_qwen.py`
  - added local `llama-server` lifecycle management
  - added `--server-bin`, `--server-host`, `--server-port`, `--server-ready-timeout-sec`, `--runtime-profile`, `--runtime-env-json`, `--tokenizer-path`
  - added `--allow-mcq-unavailable` for audit-mode runs
  - writes `llama_server_loglikelihood_probe.json`
  - skips MCQ execution only when the probe proves prompt logprobs are unavailable and audit-mode explicitly allows it
  - passes `--fewshot_as_multiturn --apply_chat_template` for chat-completions runs
- `scripts/audit_zapabob_runtime.py`
  - writes:
    - `artifacts/runtime_eval/current_main/runtime_code_audit.json`
    - `artifacts/runtime_eval/current_main/runtime_code_audit.md`
- `scripts/export_online_eval_report.py`
  - now uses headless `Agg`
  - supports nested runtime mode directories
  - writes empty / not-available pairwise outputs instead of inventing p-values when fewer than 2 claimable modes exist
- `pyproject.toml`
  - `eval` extra updated to `lm-eval[api]`

## Reduced exact baseline

- output root:
  - `artifacts/runtime_eval/current_main/exact`
- corpus:
  - `artifacts/online_eval_inputs/data/wikitext2_test.txt`
- runtime report:
  - `artifacts/online_eval_report/current_main_runtime_audit/reports/online_eval_summary.md`

### What completed

- `perplexity`
  - exit `0`
  - chunk rows written to `runtime_ppl_chunks.csv`
  - summary written to `runtime_ppl_tool_summary.csv`
- `llama-bench`
  - exit `0`
  - raw samples written to `runtime_bench_samples.csv`
- `lm-eval` chat path
  - `gsm8k` reduced run completed
  - summary written to `lm_eval_chat_summary.csv`
  - item rows written to `lm_eval_chat_items.csv`

### What did not remain claimable

- `lm-eval` MCQ path is marked unavailable in:
  - `artifacts/runtime_eval/current_main/exact/metrics/lm_eval_mcq_unavailable.json`
- reason:
  - current-main `/v1/completions` responses provide `logprobs.content[*].logprob`
  - they do **not** provide OpenAI-style prompt `token_logprobs`
  - therefore `lm-eval` loglikelihood tasks cannot be claimed without an adapter/proxy
- proof artifact:
  - `artifacts/runtime_eval/current_main/exact/logs/llama_server_loglikelihood_probe.json`

## TurboQuant-enabled smoke

- output root:
  - `artifacts/runtime_eval/current_main/turboquant_enabled_audit`
- runtime env:
  - `LLAMA_TURBOQUANT=1`
  - `LLAMA_TURBOQUANT_SO8=1`
  - `LLAMA_TURBOQUANT_SO8_LEARNED=0`
  - `LLAMA_TURBOQUANT_TRIALITY=1`
  - `LLAMA_TURBOQUANT_TRIALITY_MIX=0.5`
  - `LLAMA_TURBOQUANT_ROTATION_SEED=0`
- server smoke completed and `llama_server.stderr.log` contains:
  - TurboQuant enabled banner
  - `cpy_k: TurboQuant K-path active ...`
  - `cpy_v: TurboQuant V-path active ...`

### Interpretation

- this proves env acceptance and log-path activation in current-main
- it does **not** prove that SO8/triality helper math is wired into a selectable runtime path
- the source audit still shows helper symbols are only defined in `src/llama-turboquant.cpp` and not called from the runtime hot path

## Secondary GGUF validation

- goal:
  - fresh `Q8_0 -> .turboquant.gguf` repack as secondary distribution/readback validation
- first attempt on `H:` failed and was captured in:
  - `artifacts/turboquant_gguf_validation/current_main/fresh_package_blocker.json`
- completion path:
  - reran the repack on `C:\_codex_targets\qwen9b_runtime_validation`
  - fresh package completed successfully there
- success artifact:
  - `artifacts/turboquant_gguf_validation/current_main/fresh_package_success.json`
- fresh package output:
  - `C:\_codex_targets\qwen9b_runtime_validation\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.current-main-validation.turboquant.gguf`
- fresh readback outputs:
  - `artifacts/turboquant_gguf_validation/current_main/fresh_turboquant_manifest_readback.json`
  - `artifacts/turboquant_gguf_validation/current_main/fresh_hypura_bridge_readback.json`

### Reference readback kept for comparison

- reference file:
  - `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.turboquant.gguf`
- readback outputs:
  - `artifacts/turboquant_gguf_validation/current_main/reference_turboquant_manifest_readback.json`
  - `artifacts/turboquant_gguf_validation/current_main/reference_hypura_bridge_readback.json`

## Verification

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\test_eval_stats.py tests\test_runtime_eval.py tests\test_eval_scripts.py tests\test_reporting.py tests\test_repo_contract.py -q
uv run python scripts\validate_repo_contract.py
git diff --check
```

## Expected follow-up

- if README should claim runtime TurboQuant outcomes, first land a mode-selectable runtime path in the authoritative external fork
- if MCQ benchmarks must run through `lm-eval` on current-main, add a response-schema adapter/proxy for prompt logprobs
- if the fresh package needs to move back under repo-managed storage, free > package-size space on a non-system workspace drive first
