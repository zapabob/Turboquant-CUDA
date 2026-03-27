# Turboquant-CUDA

PyTorch-first TurboQuant paper reproduction, Qwen3.5-9B captured replay, and K/V-separated research extensions.  
PyTorch を正系にした TurboQuant 論文再現、Qwen3.5-9B captured replay、そして K/V 分離の研究拡張をまとめたリポジトリです。

## Overview / 概要

This repository is organized into three layers.  
このリポジトリは 3 層構成です。

- `turboquant.paper_baseline`
  Paper-faithful Stage 1 / Stage 2 in PyTorch only.  
  論文忠実な Stage 1 / Stage 2 の PyTorch 実装です。
- `turboquant.research_extension`
  K/V-separated codecs, V sensitivity analysis, and protected-V research branches.  
  K/V 分離 codec、V 感度解析、protected-V 系の研究実装です。
- `turboquant.adapters.hf_qwen`
  Optional Hugging Face / Qwen capture and replay adapter.  
  任意の Hugging Face / Qwen capture / replay adapter です。

The baseline acceptance target is PyTorch-only reproduction, not runtime integration.  
基準となる合格条件は runtime 統合ではなく、PyTorch-only の論文再現です。

## Dual Schema / デュアルスキーマ

This repo emits two config families.  
このリポジトリは 2 系統の config を出力します。

- `turboquant_config.paper.json`
  Paper-faithful baseline for HF/Qwen replay.  
  HF/Qwen replay 用の論文忠実 baseline です。
- `turboquant_config.research.json`
  Research / future Hypura-GGUF sidecar schema.  
  研究用、および将来の Hypura / GGUF sidecar 用 schema です。

## Quick Start / クイックスタート

```powershell
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev
uv run python scripts\env_check.py
uv run python -m pytest -q
uv run python scripts\paper_validate_synthetic.py --trials 8
uv run python scripts\paper_validate_attention.py --trials 8 --synthetic-layers 4
uv run python scripts\research_validate_v_codecs.py --query-source synthetic --trials 3
uv run python scripts\research_value_sensitivity.py --trials 3 --synthetic-layers 4
```

To enable the Qwen adapter as well:  
Qwen adapter も使う場合は次です。

```powershell
uv sync --extra cu128 --extra dev --extra hf_qwen
uv run python scripts\capture_qwen_kv.py --weight-load 4bit --max-length 96
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv --bits 2,2.5,3,3.5,4 --write-config
uv run python scripts\research_validate_v_codecs.py --query-source captured --kv-dir artifacts\kv --trials 1 --max-layers 1 --bits 2,2.5,3.5,4 --write-config
```

## Main Entry Points / 主なエントリポイント

### Paper Baseline / 論文忠実 baseline

- `scripts\paper_validate_synthetic.py`
- `scripts\paper_validate_attention.py`
- `scripts\paper_validate_captured_qwen.py`

### Research Extension / 研究拡張

- `scripts\research_validate_v_codecs.py`
- `scripts\research_value_sensitivity.py`

### HF/Qwen Adapter / HF・Qwen adapter

- `scripts\capture_qwen_kv.py`
- `scripts\validate_attention_scores.py`
- `scripts\export_report.py`
- `scripts\env_check.py`
- `scripts\benchmark_encode_decode.py`

## Design Position / 設計上の立場

`paper_baseline` is the canonical implementation target.  
`paper_baseline` を正系の実装対象とします。

`research_extension` is where we test the hypothesis that `K` and `V` should not share the same codec family.  
`research_extension` では、`K` と `V` は同じ codec family を共有すべきではない、という仮説を検証します。

`adapters.hf_qwen` is secondary validation only.  
`adapters.hf_qwen` は二次的な検証レイヤーです。

`bitsandbytes`, `transformers`, `llama.cpp`, and `Unsloth` are not part of the paper-baseline acceptance criteria.  
`bitsandbytes`、`transformers`、`llama.cpp`、`Unsloth` は論文 baseline の合格条件には含めません。

## Current Baseline Result / 現在の baseline 結果

The current mathematical bottleneck is the value path.  
現在の数理的ボトルネックは value path です。

Paper-faithful `full_kv` reduces KV cache much more aggressively than `key_only_random`, but hidden-state drift and attention-output transport error are consistently worse.  
論文忠実な `full_kv` は `key_only_random` より KV cache を強く削減しますが、hidden-state drift と attention-output transport error は一貫して悪化します。

### Paper Baseline Qwen3.5-9B Captured Replay / Qwen3.5-9B captured replay

This extracted artifact set uses only the paper modes.  
この抽出 artifact は論文 baseline の mode だけを使っています。

- `exact`
- `key_only_random`
- `full_kv`

Mixed-bit points `2.5` and `3.5` follow the paper policy.  
mixed-bit の `2.5` と `3.5` は論文ポリシーに従います。

- `2.5 bit = 32 channels @ 3 bit + 96 channels @ 2 bit`
- `3.5 bit = 32 channels @ 4 bit + 96 channels @ 3 bit`

Canonical artifacts / 正式な artifact:

- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured_mean_pm_sd.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_summary_captured_mean_pm_sd.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_memory_by_bit_mean_pm_sd.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_memory_by_bit_mean_pm_sd.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_omnibus.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_omnibus.md`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_pairwise.csv`
- `artifacts/paper_baseline/qwen_captured_reported/metrics/attention_statistics_pairwise.md`
- `artifacts/paper_baseline/qwen_captured_reported/turboquant_config.paper.json`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_runtime_tradeoffs_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_mean_pm_sd_captured.png`
- `artifacts/paper_baseline/qwen_captured_reported/plots/attention_v_breakage_by_bit_sd.png`

Representative captured values / 代表値:

| Mode | Bits | Logit Cosine | Hidden Cosine | Memory / Exact |
| --- | ---: | ---: | ---: | ---: |
| key-only random | 2.0 | 0.997070 | 0.997070 | 0.566406 |
| key-only random | 4.0 | 0.998047 | 0.999023 | 0.628906 |
| full-KV | 2.0 | 0.997070 | 0.940430 | 0.130859 |
| full-KV | 4.0 | 0.998047 | 0.995117 | 0.255859 |

Mean +/- SD summary from raw replay aggregation / raw replay 集計に基づく mean +/- SD:

| Mode | Bits | Logit Cosine (mean +/- SD) | Hidden Cosine (mean +/- SD) | Memory Ratio (mean +/- SD) |
| --- | ---: | --- | --- | --- |
| key-only random | 2.0 | 0.995117 +/- 0.001953 | 0.997070 +/- 0.003740 | 0.566406 +/- 0.000000 |
| key-only random | 2.5 | 0.998047 +/- 0.002255 | 0.999023 +/- 0.001953 | 0.574219 +/- 0.000000 |
| key-only random | 3.0 | 1.000000 +/- 0.000000 | 0.998047 +/- 0.002255 | 0.597656 +/- 0.000000 |
| key-only random | 3.5 | 0.999023 +/- 0.001953 | 0.999023 +/- 0.001953 | 0.605469 +/- 0.000000 |
| key-only random | 4.0 | 0.998047 +/- 0.003906 | 0.999023 +/- 0.001953 | 0.628906 +/- 0.000000 |
| full-KV | 2.0 | 0.995117 +/- 0.001953 | 0.939453 +/- 0.006766 | 0.130859 +/- 0.000000 |
| full-KV | 2.5 | 0.998047 +/- 0.002255 | 0.957031 +/- 0.003189 | 0.146484 +/- 0.000000 |
| full-KV | 3.0 | 1.000000 +/- 0.000000 | 0.980469 +/- 0.005524 | 0.193359 +/- 0.000000 |
| full-KV | 3.5 | 0.999023 +/- 0.001953 | 0.988281 +/- 0.003189 | 0.208984 +/- 0.000000 |
| full-KV | 4.0 | 0.998047 +/- 0.003906 | 0.995117 +/- 0.001953 | 0.255859 +/- 0.000000 |

KV cache / memory usage by bit / bit ごとの KV cache・メモリ使用量:

| Bits | key-only random memory ratio | full-KV memory ratio | full-KV memory bits (mean +/- SD) |
| --- | --- | --- | --- |
| 2.0 | 0.566406 +/- 0.000000 | 0.130859 +/- 0.000000 | 78256.0 +/- 22246.7 |
| 2.5 | 0.574219 +/- 0.000000 | 0.146484 +/- 0.000000 | 87600.0 +/- 24903.0 |
| 3.0 | 0.597656 +/- 0.000000 | 0.193359 +/- 0.000000 | 115632.0 +/- 32872.0 |
| 3.5 | 0.605469 +/- 0.000000 | 0.208984 +/- 0.000000 | 124976.0 +/- 35528.3 |
| 4.0 | 0.628906 +/- 0.000000 | 0.255859 +/- 0.000000 | 153008.0 +/- 43497.3 |

The practical trade-off is consistent across all bits: `full_kv` saves much more KV cache, but the additional V compression mainly appears as hidden-state degradation and transport error, not as a large logit-cosine drop.  
実務上のトレードオフは全 bit で一貫しています。`full_kv` は KV cache を大きく削減しますが、その追加の V 圧縮は主に hidden-state 劣化と transport error として現れ、logit cosine の大きな低下としては現れません。

Figures / 図:

![Paper baseline trade-offs](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\qwen_captured_reported\plots\attention_tradeoffs_captured.png)

![Paper baseline mean plus SD](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\qwen_captured_reported\plots\attention_mean_pm_sd_captured.png)

![V breakage by bit](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\qwen_captured_reported\plots\attention_v_breakage_by_bit_sd.png)

### Statistical Treatment / 統計処理

The statistical summary uses `attention_trials_captured.csv` with `n = 4` per `(mode, bit)` group.  
統計処理は `attention_trials_captured.csv` を使い、各 `(mode, bit)` 群で `n = 4` です。

Omnibus multi-group comparison across the 10 quantized groups / 10 群の多群比較:

| Metric | Test | Statistic | p-value |
| --- | --- | ---: | ---: |
| hidden cosine similarity | Kruskal-Wallis | 33.8627 | 9.44e-05 |
| attention output relative error | Kruskal-Wallis | 37.5473 | 2.10e-05 |
| logit cosine similarity | Kruskal-Wallis | 17.2946 | 4.43e-02 |

This cleanly separates the V-sensitive metrics from the score metric.  
これにより、V に敏感な指標と score 系の指標がきれいに分かれます。

Bit-wise one-sided exact Mann-Whitney comparisons / bit ごとの exact Mann-Whitney:

| Bits | Hidden delta (`key_only - full_kv`) | Hidden p | Attention error delta (`key_only - full_kv`) | Attention error p | Logit delta | Logit p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 0.057617 | 0.0143 | -0.291382 | 0.0143 | 0.000000 | 1.0000 |
| 2.5 | 0.041992 | 0.0143 | -0.254044 | 0.0143 | 0.000000 | 1.0000 |
| 3.0 | 0.017578 | 0.0143 | -0.160980 | 0.0143 | 0.000000 | 1.0000 |
| 3.5 | 0.010742 | 0.0143 | -0.136307 | 0.0143 | 0.000000 | 1.0000 |
| 4.0 | 0.003906 | 0.0571 | -0.087006 | 0.0143 | 0.000000 | 1.0000 |

Interpretation / 解釈:

- `full_kv` always lowers hidden cosine relative to `key_only_random`.  
  `full_kv` は常に `key_only_random` より hidden cosine を下げます。
- `full_kv` always increases attention-output relative error.  
  `full_kv` は常に attention-output relative error を上げます。
- `logit_cosine_similarity` does not separate the two modes in this baseline.  
  この baseline では `logit_cosine_similarity` は両 mode をほとんど分離しません。
- The failure mode is therefore value-path transport, not the score path.  
  したがって、壊れている本体は score path ではなく value-path transport です。

Because `n = 4` per group, Holm-corrected exact pairwise tests are conservative. The omnibus Kruskal results plus the consistent directional deltas are the more informative summary.  
各群 `n = 4` と小さいため、Holm 補正済みの exact pairwise test はかなり保守的です。そのため、Kruskal の omnibus 結果と一貫した差の向きのほうが情報量が高いと解釈しています。

### Audit: Google Blog vs Paper vs Qwen3.5-9B / 監査: Google ブログ vs 論文 vs Qwen3.5-9B

We also provide a three-column audit table that compares the public Google blog messaging, the actual scope of the paper, and our Qwen3.5-9B captured replay.  
Google の広報ブログ、論文が実際に保証している範囲、そして Qwen3.5-9B の captured replay 実測を 3 列で並べた監査表も用意しています。

Audit artifacts / 監査 artifact:

- `artifacts/paper_baseline/google_blog_audit/metrics/google_blog_paper_qwen_audit.csv`
- `artifacts/paper_baseline/google_blog_audit/metrics/google_blog_paper_qwen_audit.md`
- `artifacts/paper_baseline/google_blog_audit/metrics/qwen_summary_stats.csv`
- `artifacts/paper_baseline/google_blog_audit/metrics/qwen_summary_stats.md`
- `artifacts/paper_baseline/google_blog_audit/plots/google_blog_paper_qwen_audit.png`

Audit conclusion / 監査の結論:

- The Google blog is directionally correct about KV-cache reduction.  
  Google ブログは KV cache 削減については方向として正しいです。
- It is too broad if read as "no accuracy loss in general runtimes."  
  ただし「一般的な runtime でも性能劣化なし」と読むと広すぎます。
- On Qwen3.5-9B captured replay, `full_kv` preserves score-like metrics much better than it preserves value transport.  
  Qwen3.5-9B captured replay では、`full_kv` は score 系指標の保持に比べて value transport の保持がかなり弱いです。

![Google blog audit plot](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\google_blog_audit\plots\google_blog_paper_qwen_audit.png)

## Final Takeaway / 最終結論

Paper baseline conclusion: `key_only_random` preserves hidden geometry better than `full_kv` on Qwen3.5-9B captured replay.  
論文 baseline の結論: Qwen3.5-9B captured replay では `key_only_random` のほうが `full_kv` より hidden geometry を保ちます。

Mixed-bit `2.5 / 3.5` remain useful intermediate Pareto points.  
mixed-bit の `2.5 / 3.5` は中間 Pareto 点として依然有用です。

The statistical evidence and the audit table both support the same qualitative claim: TurboQuant reduces KV cache size as intended, but the paper-faithful `full_kv` path damages the V-dependent token-output transport signal before it meaningfully damages the logit score signal.  
統計処理と監査表の両方が、同じ質的結論を支持しています。つまり、TurboQuant は意図どおり KV cache を削減しますが、論文忠実な `full_kv` 経路は、logit の score 信号を大きく壊す前に、V 依存の token-output transport 信号を壊します。

Licensed under the MIT License. See `LICENSE`.  
ライセンスは MIT License です。`LICENSE` を参照してください。
