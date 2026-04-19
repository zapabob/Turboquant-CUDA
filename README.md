# TurboQuant CUDA

**Offline-first TurboQuant research, GGUF packaging, and runtime-contract export
for Qwen 3.5 and Gemma 4.**

TurboQuant CUDA is the Windows-native workbench behind the current
`Triality/TurboQuant -> GGUF -> llama.cpp/Hypura` handoff.

It is built for people who care about more than "the artifact was created."
This repo keeps offline metrics, export contracts, runtime-facing metadata, and
practical operator flows in one place.

## Why This Repo Exists

Most quantization repos stop at one of these points:

- reconstruction error looks good
- the benchmark table looks good
- a model file was exported

This repo exists to keep the whole path honest:

- reproduce TurboQuant Stage 1 and Stage 2 offline
- judge hidden-state transport separately from logit similarity
- compare memory, quality, and runtime trade-offs on a practical GPU class
- export GGUF artifacts that current runtime loaders can actually validate

The rule here is simple:

**offline correctness first, runtime claims second**

## What Works Now

- Qwen3.5-9B captured KV validation on RTX 3060 12 GB
- fixed 7-mode reduced comparison matrix with hidden, attention, logit, memory,
  and runtime views
- Triality-proxy SO(8) K-path evaluation with
  `key_only_block_so8_triality_vector`
- deterministic Triality fixture export for both:
  - Qwen 3.5 text-only bundles
  - Gemma 4 paired `text GGUF + mmproj GGUF` bundles
- strict `hypura.turboquant.*` metadata export
- strict weight payload v1 export with explicit `tq4_1s` codec metadata
- local FastAPI + React `TurboQuant Studio`

## Latest Mainline Additions

The current branch adds the pieces that make the export path more useful to
runtime consumers:

- paired multimodal manifests for Gemma 4
- strict `hypura.turboquant.weight.v1` payload validation
- explicit `hypura.turboquant.weight.codec = tq4_1s`
- Qwen 3.5 `config-i` style default weight plans
- Gemma 4 `kv-first-multimodal-safe` default weight plans
- GGUF writer changes that preserve source tensor metadata more cleanly
- embedded rotation tensor naming that stays within GGUF length limits

## Mainline Lanes

### 1. Measured Research Lane

This is the lane with the strongest real evidence today:

- model family: Qwen3.5-9B
- modality scope: text-only
- hardware class: RTX 3060 12 GB
- workflow: captured KV -> reduced 7-mode matrix -> report export

### 2. Export Contract Lane

This is the lane that makes the runtime integration story practical:

- deterministic fixture generation
- Qwen text-only manifest export
- Gemma paired multimodal manifest export
- payload and metadata hashing
- strict rebuild-time contract verification

### 3. Operator Lane

This is the lane for iterating locally without reinventing orchestration:

- `TurboQuant Studio`
- FastAPI backend plus React frontend
- capture, validate, compare, package, and serve workflows in one shell

## Quick Start

Run everything from the repository root.

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev --extra hf_qwen --extra eval
uv run python scripts\env_check.py
uv run python scripts\validate_repo_contract.py
```

### Extras

- `--extra cu128`: CUDA PyTorch
- `--extra dev`: tests and local verification helpers
- `--extra hf_qwen`: Hugging Face / Qwen capture path
- `--extra eval`: runtime eval and report export dependencies

## Five-Minute Smoke

### Contract smoke for Qwen

```powershell
uv run python scripts\export_triality_fixture.py `
  --output-dir artifacts\fixture_smoke_qwen `
  --mode triality-proxy-so8-pareto `
  --model-family "Qwen/Qwen3.5-27B"

uv run python scripts\verify_triality_export.py `
  --manifest artifacts\fixture_smoke_qwen\triality-proxy-so8-pareto\triality-fixture-manifest.json
```

### Contract smoke for Gemma 4 paired export

```powershell
uv run python scripts\export_triality_fixture.py `
  --output-dir artifacts\fixture_smoke_gemma `
  --mode triality-proxy-so8-pareto `
  --model-family "google/gemma-4-e4b-it"

uv run python scripts\verify_triality_export.py `
  --manifest artifacts\fixture_smoke_gemma\triality-proxy-so8-pareto\triality-fixture-manifest.json
```

### Launch TurboQuant Studio

Backend:

```powershell
uv run python scripts\run_turboquant_studio.py
```

Frontend:

```powershell
Set-Location .\studio-web
npm install
npm run dev
```

Then open:

- `http://127.0.0.1:8000/studio`

## Contract Snapshot

This repo exports GGUF-facing metadata under `hypura.turboquant.*`.

The current weight payload path is intentionally explicit:

```json
{
  "schema": "hypura.turboquant.weight.v1",
  "codec": "tq4_1s",
  "policy": "qwen35-config-i",
  "source_ftype": "q8_0",
  "tensor_plan": {
    "blk.*.attn_q.weight": "tq4_1s",
    "blk.*.attn_k.weight": "tq4_1s",
    "blk.*.attn_v.weight": "tq4_1s",
    "blk.*.attn_output.weight": "tq4_1s",
    "blk.*.ffn_gate.weight": "tq4_1s",
    "blk.*.ffn_up.weight": "tq4_1s",
    "blk.*.ffn_down.weight": "q4_k"
  }
}
```

Current family defaults:

- Qwen 3.5:
  - policy: `qwen35-config-i`
  - modality scope: `text-only`
- Gemma 4:
  - policy: `gemma4-kv-first-multimodal-safe`
  - modality scope: `full-multimodal` for multimodal families

## What This Repo Does Not Claim

This repo is careful about scope.

It does claim:

- reproducible offline research
- strong contract export discipline
- runtime-facing metadata that loaders can validate

It does not automatically claim:

- native runtime `tq4_1s` matmul is complete everywhere
- replay-only evidence is enough to prove runtime superiority
- Gemma 4 already has the same measured depth here as the Qwen 3060 lane

## Current 12 GB Snapshot

The measured headline remains the Qwen3.5-9B RTX 3060 reduced matrix.

### 4-bit snapshot

| Mode | Logit cosine | Hidden cosine | Memory ratio vs exact |
| --- | --- | --- | --- |
| `exact` | `1.000000 +/- 0.000000` | `1.000000 +/- 0.000000` | `1.000000 +/- 0.000000` |
| `multiscreen_relevance` | `1.002604 +/- 0.006379` | `1.000000 +/- 0.000000` | `0.660156 +/- 0.000000` |
| `key_only_block_so8_triality_vector` | `1.000000 +/- 0.000000` | `0.999349 +/- 0.001595` | `0.628906 +/- 0.000000` |
| `asym_q8_turbo4` | `1.001302 +/- 0.005881` | `0.994141 +/- 0.004784` | `0.378906 +/- 0.000000` |
| `asym_q8_turbo3` | `0.996745 +/- 0.002941` | `0.981771 +/- 0.004731` | `0.347656 +/- 0.000000` |

Practical read:

- `key_only_block_so8_triality_vector` is the safest K-side production
  reference here
- `multiscreen_relevance` is also strong when hidden stability matters
- `asym_q8_turbo4` stays useful as the aggressive memory-saving baseline

Tracked README figure copies:

![Qwen 3060 Matrix Quality Summary](_docs/assets/qwen_3060_matrix_attention.png)

![Qwen 3060 Matrix Runtime Summary](_docs/assets/qwen_3060_matrix_runtime.png)

## Key Workflows

| Goal | Script |
| --- | --- |
| Environment check | `scripts\env_check.py` |
| Repo contract validation | `scripts\validate_repo_contract.py` |
| Capture Qwen KV | `scripts\capture_qwen_kv.py` |
| Run reduced 3060 matrix | `scripts\validate_qwen_3060_matrix.py` |
| Export matrix report | `scripts\export_report.py` |
| Export deterministic fixture bundle | `scripts\export_triality_fixture.py` |
| Verify exported fixture bundle | `scripts\verify_triality_export.py` |
| Package GGUF metadata | `scripts\pack_turboquant_gguf.py` |
| Runtime eval audit | `scripts\eval_runtime_qwen.py` |
| Launch Studio | `scripts\run_turboquant_studio.py` |

## Repository Layout

- `turboquant/`
  - paper baseline logic
  - research extensions
  - runtime-facing contract helpers
- `scripts/`
  - capture, validation, export, plotting, packaging, and studio entrypoints
- `tests/`
  - contract and packaging tests
- `studio-web/`
  - local React frontend for `TurboQuant Studio`
- `artifacts/`
  - captured runs, reports, plots, and fixture outputs
- `vendor/llama.cpp/`
  - vendored GGUF/runtime reference path

## Validation Policy

This repo is intentionally strict:

- Stage 1 and Stage 2 are kept conceptually separate
- exact-score and estimated-score paths are not merged into one label
- hidden-state transport is treated as first-class, not optional
- runtime claims are made only when the runtime path itself is measured

Recommended validation:

```powershell
uv run python scripts\validate_repo_contract.py
uv run python -m pytest tests/test_turboquant_gguf_profiles.py tests/test_triality_contract.py -q
```

## Related Repositories

| Repository | Role |
| --- | --- |
| [zapabob/Turboquant-CUDA](https://github.com/zapabob/Turboquant-CUDA) | Upstream offline TurboQuant work |
| [zapabob/llama.cpp](https://github.com/zapabob/llama.cpp) | Runtime GGUF loader and serving path |
| [zapabob/hypura](https://github.com/zapabob/hypura) | Tiered inference and operational integration target |
| [zapabob/multiscreen-pytorch](https://github.com/zapabob/multiscreen-pytorch) | Multiscreen reference path |

## License

Apache-2.0
