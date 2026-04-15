# TurboQuant CUDA — Paper-Faithful TurboQuant + Triality-Proxy SO(8) Research

Paper-faithful PyTorch implementation of the TurboQuant KV-cache compression algorithm,
Qwen3.5-9B captured KV replay, and K/V research extensions including triality-proxy SO(8) learned rotations.
**Windows + `uv` + Python 3.12.x**; CUDA via `uv sync --extra cu128`.

- **What**: Reproduces TurboQuant Stage 1 (sphere Lloyd-Max) + Stage 2 (inner-product estimator + QJL sketch) faithful to the paper, then layers research extensions on top.
- **Production K path**: Multiscreen relevance → per-channel mixed-bit allocation → triality-proxy SO(8) (vector proxy view) + TurboQuant Stage 1+2 on captured KV.
- **Validation order**: synthetic → attention → captured (lock offline correctness first).

## Scope Layers

- **Paper-faithful**: TurboQuant Stage 1 / Stage 2 math, paper mixed-bit presets, synthetic + captured attention replay.
- **Production canonical**: `key_only_block_so8_triality_vector`, which is the current **triality-proxy** vector-view K-only path. The runtime mode name is legacy; the math label is proxy, not true Spin(8) triality.
- **Research / ablation**: random/static SO(8), learned block-SO(8), proxy view comparisons, value-codec experiments, and future true `triality_spin8` work.

---

## Triality-Proxy SO(8): Current Research Differentiator

The distinguishing current research contribution of this repo is **per-layer learned SO(8) block rotations** applied to the K side before TurboQuant quantization via a triality-proxy view family.

### What it is

- The key head dimension is split into **8-dimensional blocks**.
- Each block is rotated by an element of **SO(8)** fit to minimize reconstruction loss on captured KV via gradient descent through the matrix exponential of a skew-symmetric generator (`block_so8_from_skew`).
- **Triality proxy** (`triality_proxy.py`) provides 3 empirical proxy views of the learned rotation: `vector`, `spinor_plus_proxy`, `spinor_minus_proxy`.
- These proxy views are **not** strict Spin(8) vector / half-spinor representations. True `triality_spin8` work is future research mode and is not the production default yet.
- The production canonical K path uses the **`vector` proxy view** with mode `key_only_block_so8_triality_vector` (see `k_triality.PRODUCTION_K_TURBOQUANT_MODE` / `PRODUCTION_K_TURBOQUANT_VIEW`).
- Trained rotations are saved to `rotations/*.pt` and loaded with `--rotation-dir`.

### Why it matters

At the same K-only memory budget as `key_only_random`:
- Triality-vector hidden cosine tracks or exceeds random at 2-bit
- `full_kv` uses less memory but hidden cosine degrades sharply at low bits — a different Pareto point

*Table: mean hidden cosine by K mode at selected bits (from `triality_summary_mean_pm_sd.md`)*

| Bits | `triality_vector` | `key_only_random` | `full_kv` | Δ (Tri − rand) | Δ (Tri − full-KV) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.000326 ± 0.004596 | 0.999837 ± 0.003729 | 0.941569 ± 0.004383 | +0.000488 | +0.058757 |
| 4 | 1.000651 ± 0.003762 | 1.000000 ± 0.003048 | 0.995605 ± 0.003115 | +0.000651 | +0.005046 |
| 8 | 0.999837 ± 0.003150 | 0.999512 ± 0.003321 | 0.999674 ± 0.003241 | +0.000326 | +0.000163 |

Ablation modes (`key_only_random`, static/learned SO(8), `full_kv`) remain available for reproducibility and paper replay.

---

## Architecture

| Layer | Role |
| --- | --- |
| `turboquant.paper_baseline` | Paper-faithful Stage 1 (sphere Lloyd-Max → `QuantizedMSEBatch`) + Stage 2 (QJL sketch residual correction → `QuantizedProdBatch`) |
| `turboquant.research_extension` | K/V codecs, V sensitivity, `ProtectedValueCodec`, Multiscreen KV relevance gate, **triality-proxy SO(8)**, production K modes |
| `turboquant.adapters.hf_qwen` | Optional: HF/Qwen KV capture from bf16/4bit/8bit weight loads |

### Rotation policies

| Policy | Description |
| --- | --- |
| `random_haar` | Dense Haar-like orthogonal matrix via QR decomposition (cached) |
| `block_so8_static` | Block-diagonal SO(8) with static seed-derived 8×8 blocks |
| `block_so8_learned` | Block-diagonal SO(8) fitted via gradient descent on quantization loss |
| `fast_hadamard` | D1·H·D2 structured rotation (O(d log d) via FWHT; materialised as dense matrix in `rotation_from_policy`). Use `apply_fast_rotation` for the O(d log d) path directly. |

---

## Related Repositories

| Repository | Role |
| --- | --- |
| [zapabob/multiscreen-pytorch](https://github.com/zapabob/multiscreen-pytorch) | PyTorch reference implementation of the Multiscreen architecture. `trim_and_square` + MiPE KV relevance scoring feeds into `research_extension` |
| [zapabob/Hypura](https://github.com/zapabob/Hypura) | GPU/RAM/NVMe tiered inference scheduler; combined with Multiscreen KV window cap for VRAM reduction |
| [zapabob/llama.cpp](https://github.com/zapabob/llama.cpp) | llama.cpp fork for consuming TurboQuant / triality-proxy artifacts on the GGUF side |

---

## Build Contract

This workspace is expected to stay aligned with two upstream sources of truth:

- **PyTorch / offline quantization semantics:** [zapabob/Turboquant-CUDA](https://github.com/zapabob/Turboquant-CUDA)
- **GGUF / runtime consumption path:** vendored [zapabob/llama.cpp](https://github.com/zapabob/llama.cpp) at `vendor/llama.cpp`

Rules:

- `.gitmodules` must keep `vendor/llama.cpp` pinned to `https://github.com/zapabob/llama.cpp.git`.
- Rust / Hypura builds must use the vendored `vendor/llama.cpp`, or `LLAMA_CPP_DIR` / `HYPURA_LLAMA_CPP_DIR` must point to a **zapabob/llama.cpp-compatible** checkout.
- A zapabob-compatible checkout is validated by the presence of the TurboQuant runtime files `src/llama-turboquant.h` and `src/llama-turboquant.cpp` plus the expected TurboQuant / triality symbols.
- Repo-level consistency is checked by `repo_contract.toml` and `scripts\validate_repo_contract.py`.

Validation:

```powershell
uv run python scripts\validate_repo_contract.py
```

This validation is also part of `.\scripts\run_production_tests.ps1`.

For Rust workspace builds, prefer:

```powershell
.\scripts\build_rust_workspace.ps1 -Package hypura -NoCuda
```

This runs the same contract validation before invoking `cargo build`.
If `rust\target` is pressuring disk space, add `-TargetDir "C:\path\to\cargo-target"` to move Cargo artifacts to a roomier drive.

---

## Contributing GPU Benchmarks

We are building a community result table. If you have hardware not yet represented, please run the scripts and open an issue.

**Target hardware:**
- NVIDIA RTX 3080 Ti / 3090 / 4090 / 4090 Ti / 5090 (CUDA)
- AMD RDNA4 (HIP / ROCm)
- Apple M-series (Metal / MPS)

**Requested metrics:**
- PPL (WikiText-103) at bits 2 / 2.5 / 3 / 3.5 / 4 for Stage 1 and Stage 1+2
- NIAH (Needle-in-a-Haystack) retrieval accuracy at each bit level
- Decode speed (tokens/sec) vs `q8_0` and `q4_0` baselines
- VRAM footprint at each bit level (use `research_vram_multigroup_qwen.py`)

**How to contribute:**
1. Fork [zapabob/Turboquant-CUDA](https://github.com/zapabob/Turboquant-CUDA)
2. Capture KV: `uv run python scripts\capture_qwen_kv.py --weight-load 4bit --dtype bfloat16 --trust-remote-code --model-id <your-model-path> --output-dir artifacts\kv_4bit --max-length 64`
3. Run `uv run python scripts\paper_validate_synthetic.py --trials 8`
4. Run `uv run python scripts\research_validate_multiscreen_kv.py --captured-dir artifacts\kv_4bit --eval-device cuda`
5. Run `uv run python scripts\research_vram_multigroup_qwen.py --kv-dir artifacts\kv_4bit --eval-device cuda --trials 8` for VRAM data
6. Open an issue titled **"GPU Benchmark: \<GPU model\>"** and attach the CSV outputs from `artifacts/`

---

## Environment Setup

**Required**: Run all commands from `hub_Qwen3.5-9B-SOT/` (the directory containing `pyproject.toml`).

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev --extra hf_qwen
uv run python scripts\env_check.py
```

- `--extra cu128`: required for CUDA PyTorch (omitting gives CPU-only torch)
- `--extra hf_qwen`: required for `capture_qwen_kv.py` (transformers, accelerate, bitsandbytes)
- Do **not** use bare `py -3` for project scripts — use `uv run python ...`
- From a parent directory: `uv run --project hub_Qwen3.5-9B-SOT python scripts\...`

Production bundle (env check + pytest):

```powershell
.\scripts\run_production_tests.ps1
```

---

## Quick Start (Offline Validation)

```powershell
Set-Location H:\path\to\hub_Qwen3.5-9B-SOT
uv run python scripts\env_check.py
uv run python -m pytest -q
uv run python scripts\paper_validate_synthetic.py --trials 8
uv run python scripts\paper_validate_attention.py --trials 8 --synthetic-layers 4
uv run python scripts\research_validate_v_codecs.py --query-source synthetic --trials 3
uv run python scripts\research_value_sensitivity.py --trials 3 --synthetic-layers 4
```

---

## Production Flow

### 1. Qwen KV Capture

- `--weight-load none`: full bf16 weights, no BitsAndBytes; `from_pretrained` uses `device_map="auto"` (~18 GB VRAM for 9B)
- `--weight-load 4bit` / `8bit`: BitsAndBytesConfig + `device_map="auto"` (12 GB default path)

```powershell
# 12 GB GPU (4-bit weights, recommended default)
uv run python scripts\capture_qwen_kv.py `
  --weight-load 4bit --dtype bfloat16 --trust-remote-code `
  --model-id "H:\Qwen3.5-9B-official-hf" `
  --output-dir artifacts\kv_4bit --max-length 64

# Full bf16 (~18 GB+ VRAM)
uv run python scripts\capture_qwen_kv.py `
  --weight-load none --dtype bfloat16 --trust-remote-code `
  --model-id "H:\Qwen3.5-9B-official-hf" `
  --output-dir artifacts\kv_full_bf16 --max-length 96
```

Outputs: `artifacts\<kv-dir>\<capture_id>\capture_manifest.json` and per-layer `layer_*_{key,value}.pt`.

### 2. Paper Baseline on Captured KV

```powershell
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv_4bit
```

### 3. Triality Full Pipeline (train → eval)

[`scripts/run_triality_full_pipeline.py`](scripts/run_triality_full_pipeline.py) always runs
`research_train_k_triality.py` then `research_validate_k_triality.py`.

```powershell
$env:PYTHONUNBUFFERED = "1"
uv run python scripts\run_triality_full_pipeline.py `
  --kv-dir artifacts\kv_4bit `
  --train-output-dir artifacts\research_extension\triality_full_train `
  --eval-output-dir artifacts\research_extension\triality_full_eval
```

Default `--bits`: `2,2.5,3,3.5,4,8` (8 is a repo extension; paper-only: `--bits 2,2.5,3,3.5,4`).
Everything after `--` is forwarded to the eval script:

```powershell
uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv_4bit -- --resume
```

**Note**: training still runs every time above. To evaluate only from saved rotations, use:

```powershell
$env:PYTHONUNBUFFERED = "1"
uv run python scripts\research_validate_k_triality.py `
  --kv-dir artifacts\kv_4bit `
  --rotation-dir artifacts\research_extension\triality_full_train\rotations `
  --bits 2,2.5,3,3.5,4,8 `
  --eval-device cuda `
  --output-dir artifacts\research_extension\triality_full_eval `
  --resume
```

**Important**: Never run multiple evals/pipelines on the **same `--output-dir`** concurrently.

### 4. Mixed-bit Multiscreen + Triality Proxy

```powershell
# Production K path (triality-proxy SO(8), vector proxy view, default --mode)
uv run python scripts\research_validate_multiscreen_kv.py `
  --captured-dir artifacts\kv_4bit --eval-device cuda

# Ablation: multiscreen relevance + explicit bits
uv run python scripts\research_validate_multiscreen_kv.py `
  --captured-dir artifacts\kv_4bit --mode multiscreen_relevance --bits 3 --max-layers 4
```

### 5. VRAM / KV Footprint Multi-group Study

```powershell
uv run python scripts\research_vram_multigroup_qwen.py `
  --kv-dir artifacts\kv_4bit `
  --model-id "H:\Qwen3.5-9B-official-hf" `
  --eval-device cuda --trials 8

# With learned rotations and triality vector mode
uv run python scripts\research_vram_multigroup_qwen.py `
  --kv-dir artifacts\kv_4bit --eval-device cuda --trials 8 `
  --modes exact,multiscreen_relevance,multiscreen_triality_vector `
  --rotation-dir artifacts\research_extension\triality_full_train\rotations
```

---

## Eval Output Layout

| Path | Contents |
| --- | --- |
| `metrics/triality_trials_captured.csv` | Raw per-trial rows |
| `metrics/triality_summary_captured.csv` / `.md` | Pooled summary (mean, std, sem, CI) |
| `metrics/triality_summary_mean_pm_sd.csv` / `.md` | Mode × bit mean ± SD table |
| `metrics/triality_statistics.csv` / `.md` | Mode-wise statistical tests |
| `metrics/triality_friedman_rotation_modes.csv` / `.md` | Friedman test across K modes at fixed bit |
| `metrics/triality_pairwise_wilcoxon_rotation_modes.csv` / `.md` | Pairwise Wilcoxon vs baseline (Holm) |
| `metrics/eval_resume_state.json` | Resume cursor + config fingerprint |
| `plots/triality_*_captured.png` | Trade-off / mean±SD plots |
| `plots/triality_advantage_*.png` | Triality advantage figures |
| `eval_status.json` | Stage machine (done → `finish`) |

Extra eval flags: `research_validate_k_triality.py -h` (`--skip-statistics`, `--skip-plots`, `--evaluate-only`, `--from-existing-trials`, `--force-fresh`).

---

## Qwen3.5-9B Captured Baseline Results (2–8 bit)

*Source: `artifacts/paper_baseline/qwen_captured_full_bf16/metrics/attention_trials_captured.csv`; n = 96 per (mode, bit) — 4 prompts × 8 layers × 3 trials.*

*KO = key_only_random, FV = full_kv. Mixed-bit: 2.5 = 25% @3bit + 75% @2bit; 3.5 = 25% @4bit + 75% @3bit.*

| Bits | Logit cosine | Hidden cosine (KO) | Hidden cosine (FV) | Memory/exact (KO) | Memory/exact (FV) | Attn err (KO) | Attn err (FV) |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 2 | 0.997314 ± 0.003554 | 0.999756 ± 0.003794 | 0.940959 ± 0.003962 | 0.566406 | 0.130859 | 0.027470 ± 0.021959 | 0.339172 ± 0.006291 |
| 2.5 | 0.998494 ± 0.003982 | 0.998779 ± 0.004099 | 0.958700 ± 0.003791 | 0.574219 | 0.146484 | 0.027599 ± 0.029244 | 0.285421 ± 0.006102 |
| 3 | 0.999349 ± 0.003872 | 0.999715 ± 0.003168 | 0.982625 ± 0.003395 | 0.597656 | 0.193359 | 0.014811 ± 0.013992 | 0.184458 ± 0.004565 |
| 3.5 | 0.999308 ± 0.003353 | 1.000285 ± 0.003896 | 0.988403 ± 0.003179 | 0.605469 | 0.208984 | 0.015692 ± 0.016130 | 0.151886 ± 0.004057 |
| 4 | 0.999552 ± 0.004312 | 0.999959 ± 0.003823 | 0.995524 ± 0.003352 | 0.628906 | 0.255859 | 0.006952 ± 0.005955 | 0.096430 ± 0.002715 |
| 8 | 1.000244 ± 0.004563 | 1.000000 ± 0.003802 | 0.999715 ± 0.003854 | 0.753906 | 0.505859 | 0.002421 ± 0.001509 | 0.029344 ± 0.000504 |

**Qualitative takeaway**: paper-faithful `full_kv` tends to break **hidden / transport metrics** (V-dependent) before logit-style scores vs `key_only_random`. KV savings are real; the K-only path is more stable at low bits.

Interactive plots:
- [attention_tradeoffs_captured.html](artifacts/paper_baseline/qwen_captured_full_bf16/plots/attention_tradeoffs_captured.html)
- [attention_runtime_tradeoffs_captured.html](artifacts/paper_baseline/qwen_captured_full_bf16/plots/attention_runtime_tradeoffs_captured.html)

![Paper baseline trade-offs](artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.png)

![Paper baseline mean ± SD](artifacts/paper_baseline/qwen_captured_reported/plots/attention_mean_pm_sd_captured.png)

---

## Triality Advantage Figures

![Triality Pareto](artifacts/research_extension/triality_full_eval_prod_bf16/plots/triality_advantage_pareto_hidden_memory.png)

![Triality grouped hidden](artifacts/research_extension/triality_full_eval_prod_bf16/plots/triality_advantage_grouped_hidden.png)

![Triality delta hidden](artifacts/research_extension/triality_full_eval_prod_bf16/plots/triality_advantage_delta_hidden.png)

Regenerate: `uv run python scripts\plot_triality_advantage.py` (use `--input-csv` / `--plots-dir` to override paths).

---

## Scripts Reference

| Category | Scripts |
| --- | --- |
| Paper baseline | `paper_validate_synthetic.py`, `paper_validate_attention.py`, `paper_validate_captured_qwen.py` |
| Triality train/eval | `research_train_k_triality.py`, `research_validate_k_triality.py`, `run_triality_full_pipeline.py`, `plot_triality_advantage.py` |
| Multiscreen + VRAM | `research_validate_multiscreen_kv.py`, `research_vram_multigroup_qwen.py` |
| V codecs | `research_validate_v_codecs.py`, `research_value_sensitivity.py` |
| HF / Qwen | `capture_qwen_kv.py`, `validate_attention_scores.py`, `export_report.py` |
| Environment | `env_check.py`, `benchmark_encode_decode.py` |

---

## Config Families

- `turboquant_config.paper.json` — paper baseline and HF replay (`PAPER_SCHEMA_KIND`)
- `turboquant_config.research.json` — research extensions (`RESEARCH_SCHEMA_KIND`)

Schema helpers: `turboquant.schema` — `build_*_config`, `read_turboquant_config`, `write_turboquant_config`, `validate_*_config`.

---

## License

Apache License 2.0 — [LICENSE](LICENSE)
