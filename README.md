# Turboquant-CUDA

PyTorch-first TurboQuant paper reproduction, rebuttal-oriented Qwen3.5-9B evaluation, and K/V-separated research extensions.  
PyTorch を正系にした TurboQuant 論文再現、Qwen3.5-9B による反論志向の評価、そして K/V 分離の研究拡張をまとめたリポジトリです。

## Position / 立場

This repository is intentionally positioned as a rebuttal-oriented reproduction.  
このリポジトリは、意図的に「反論志向の再現実装」として位置づけています。

Our current position is:  
現時点の立場は次です。

- TurboQuant does reduce KV-cache memory strongly.  
  TurboQuant は KV cache メモリを大きく削減します。
- However, Google's public messaging appears selectively benchmarked, or effectively cherry-picked, relative to broader runtime conditions.  
  ただし Google の対外的な説明は、より広い runtime 条件に対しては選択的なベンチマーク提示、つまり実質的にチェリーピッキング寄りに見えます。
- In our Qwen3.5-9B captured replay, paper-faithful `full_kv` preserves score-like metrics much better than it preserves V-dependent hidden-state transport.  
  私たちの Qwen3.5-9B captured replay では、論文忠実な `full_kv` は score 系指標の保持に比べて、V 依存の hidden-state transport の保持がかなり弱いです。

In short: KV reduction is real, but the broad "no quality loss" reading is not supported by our runtime evidence.  
要するに、KV 削減自体は本物ですが、「広い条件で性能劣化なし」という読みは私たちの runtime 実測では支持されません。

## Repository Layout / リポジトリ構成

This repository has three explicit layers.  
このリポジトリは 3 層構成です。

- `turboquant.paper_baseline`  
  Paper-faithful Stage 1 / Stage 2 in PyTorch only.  
  論文忠実な Stage 1 / Stage 2 の PyTorch 実装です。
- `turboquant.research_extension`  
  K/V-separated codecs, V sensitivity analysis, protected-V branches, and low-rank residual experiments.  
  K/V 分離 codec、V 感度解析、protected-V 系、low-rank residual 実験です。
- `turboquant.adapters.hf_qwen`  
  Optional Hugging Face / Qwen capture and replay adapter.  
  任意の Hugging Face / Qwen capture / replay adapter です。

The canonical target is still the PyTorch baseline.  
正系の対象はあくまで PyTorch baseline です。

## License / ライセンス

This repository is licensed under Apache License 2.0.  
このリポジトリは Apache License 2.0 です。

See [LICENSE](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\LICENSE).  
詳細は [LICENSE](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\LICENSE) を参照してください。

## Dual Schema / デュアルスキーマ

This repository emits two config families.  
このリポジトリは 2 系統の config を出力します。

- `turboquant_config.paper.json`  
  Paper-faithful baseline for HF/Qwen replay.  
  HF/Qwen replay 用の論文忠実 baseline です。
- `turboquant_config.research.json`  
  Research / future Hypura-GGUF sidecar schema.  
  研究用、および将来の Hypura / GGUF sidecar 用 schema です。

## Environment (Python, CUDA, uv) / 実行環境

- **Python**: This project supports **3.12.x only** (`requires-python = ">=3.12,<3.13"` in [`pyproject.toml`](pyproject.toml)).  
  **Python**: **3.12.x のみ**（[`pyproject.toml`](pyproject.toml) の `requires-python`）。

- **CUDA PyTorch**: Use **`uv sync --extra cu128`** so `torch` resolves from the `pytorch-cu128` index under `[tool.uv.sources]` / `[[tool.uv.index]]` in [`pyproject.toml`](pyproject.toml). Without that extra, you often get a CPU wheel and `torch.cuda.is_available()` stays `False`.  
  **CUDA 版 PyTorch**: **`uv sync --extra cu128`**。`pyproject.toml` の `tool.uv.sources` が `pytorch-cu128` を指す。付けないと CPU 版になりやすい。

- **Do not run project scripts with a global `py -3`** (for example Python 3.14): that interpreter is outside the supported range and usually carries a mismatched `torch`. Always **`cd` into `hub_Qwen3.5-9B-SOT` and use `uv run python ...`**.  
  **グローバルな `py -3` で直接実行しない**（例: 3.14 は範囲外）。**`hub_Qwen3.5-9B-SOT` で `uv run python ...`** を使う。

- **Working directory**: `pyproject.toml` and `scripts\` live **inside `hub_Qwen3.5-9B-SOT`**. Running `uv sync` or `uv run python scripts\...` from a parent folder (e.g. `Qwen3.5-9B-SOT-Deployment`) fails with “No pyproject.toml” or “can't open file …\scripts\…”. Either **`Set-Location hub_Qwen3.5-9B-SOT`** first, or from the parent use **`uv sync --project hub_Qwen3.5-9B-SOT`** and **`uv run --project hub_Qwen3.5-9B-SOT python scripts\...`**.  
  **作業ディレクトリ**: `pyproject.toml` と `scripts\` は **`hub_Qwen3.5-9B-SOT` 配下**のみ。親フォルダで `uv` するとエラーになる。**`cd` するか `uv --project hub_Qwen3.5-9B-SOT`** を使う。

- **GPU driver**: Use an NVIDIA driver compatible with the CUDA 12.8 stack shipped with PyTorch cu128 wheels; verify with `nvidia-smi`.  
  **GPU ドライバ**: cu128 ホイールに合う NVIDIA ドライバ。`nvidia-smi` で確認。

If `uv` is not installed yet: / `uv` 未導入なら:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

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

### Verification / 動作確認

```powershell
uv run python scripts\env_check.py
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

When CUDA is set up correctly, `env_check` should report `cuda_available: True` and `target_cuda_match: True`.  
正常時は `env_check` に `cuda_available: True` と `target_cuda_match: True` が出る。

### Alternative bootstrap / 代替セットアップ

```powershell
.\scripts\bootstrap_uv.ps1 -PythonVersion 3.12.9 -TorchExtra cu128
```

If you also need the Qwen adapter, run `uv sync --extra cu128 --extra dev --extra hf_qwen` afterward.  
Qwen 用なら続けて `uv sync --extra cu128 --extra dev --extra hf_qwen`。

To enable the Qwen adapter as well:  
Qwen adapter も使う場合は次です。

```powershell
uv sync --extra cu128 --extra dev --extra hf_qwen
uv run python scripts\capture_qwen_kv.py --weight-load 4bit --max-length 96
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv --bits 2,2.5,3,3.5,4 --write-config
uv run python scripts\research_validate_v_codecs.py --query-source captured --kv-dir artifacts\kv --trials 1 --max-layers 1 --bits 2,2.5,3.5,4 --write-config
```

### Triality pipeline (captured KV) / Triality（キャプチャ KV）

[`scripts/run_triality_full_pipeline.py`](scripts/run_triality_full_pipeline.py) runs **train-then-eval** on the same `--kv-dir` (defaults under `artifacts/research_extension/`). Arguments after `--` are forwarded to `research_validate_k_triality.py` (for example `--resume`).

```powershell
$env:PYTHONUNBUFFERED = "1"
uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv
```

Example with eval-only extras: / 評価側にだけオプションを渡す例:

```powershell
uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv -- --resume
```

**Production / 本番（学習済み回転だけ評価）**: For real captured KV and trained rotations, **only replace `--kv-dir` and `--rotation-dir`** with your capture root (where `capture_manifest.json` lives, or the parent of multiple prompt captures) and your training output `rotations/` (from `research_train_k_triality.py`). **`--output-dir` may stay as below or any other directory name** you prefer. Always use `uv run` from this repo root.

本番データでは **`--kv-dir` と `--rotation-dir` だけ**を、あなたのキャプチャルートと学習出力の `rotations` に差し替えればよい。**`--output-dir` は例のままでも別名でも可**。

```powershell
$env:PYTHONUNBUFFERED = "1"
uv run python scripts\research_validate_k_triality.py `
  --kv-dir "D:\path\to\captured\kv_root" `
  --rotation-dir "D:\path\to\train_output\rotations" `
  --eval-device cuda `
  --output-dir artifacts\research_extension\triality_full_eval
```

## Main Entry Points / 主なエントリポイント

### Paper Baseline / 論文忠実 baseline

- `scripts\paper_validate_synthetic.py`
- `scripts\paper_validate_attention.py`
- `scripts\paper_validate_captured_qwen.py`

### Research Extension / 研究拡張

- `scripts\research_validate_v_codecs.py`
- `scripts\research_value_sensitivity.py`
- `scripts\run_triality_full_pipeline.py` (triality proxy: train rotations + validate on captured KV / 同一 KV で学習→評価)
- `scripts\research_train_k_triality.py`
- `scripts\research_validate_k_triality.py`

### HF/Qwen Adapter / HF・Qwen adapter

- `scripts\capture_qwen_kv.py`
- `scripts\validate_attention_scores.py`
- `scripts\export_report.py`
- `scripts\env_check.py`
- `scripts\benchmark_encode_decode.py`

## Paper Baseline Result on Qwen3.5-9B / Qwen3.5-9B での論文 baseline 結果

The extracted paper-baseline artifact set lives under:  
論文 baseline の抽出 artifact は次にあります。

- `artifacts/paper_baseline/qwen_captured_reported/`

This section uses only the paper modes.  
この節では論文 baseline の mode のみを使います。

- `exact`
- `key_only_random`
- `full_kv`

Mixed-bit points `2.5` and `3.5` follow the paper policy.  
mixed-bit の `2.5` と `3.5` は論文ポリシーに従います。

- `2.5 bit = 32 channels @ 3 bit + 96 channels @ 2 bit`
- `3.5 bit = 32 channels @ 4 bit + 96 channels @ 3 bit`

### Representative Values / 代表値

| Mode | Bits | Logit Cosine | Hidden Cosine | Memory / Exact |
| --- | ---: | ---: | ---: | ---: |
| key-only random | 2.0 | 0.997070 | 0.997070 | 0.566406 |
| key-only random | 4.0 | 0.998047 | 0.999023 | 0.628906 |
| full-KV | 2.0 | 0.997070 | 0.940430 | 0.130859 |
| full-KV | 4.0 | 0.998047 | 0.995117 | 0.255859 |

### Mean +/- SD / mean +/- SD

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

### KV Cache Reduction / KV cache 削減

| Bits | key-only random memory ratio | full-KV memory ratio | full-KV memory bits (mean +/- SD) |
| --- | --- | --- | --- |
| 2.0 | 0.566406 +/- 0.000000 | 0.130859 +/- 0.000000 | 78256.0 +/- 22246.7 |
| 2.5 | 0.574219 +/- 0.000000 | 0.146484 +/- 0.000000 | 87600.0 +/- 24903.0 |
| 3.0 | 0.597656 +/- 0.000000 | 0.193359 +/- 0.000000 | 115632.0 +/- 32872.0 |
| 3.5 | 0.605469 +/- 0.000000 | 0.208984 +/- 0.000000 | 124976.0 +/- 35528.3 |
| 4.0 | 0.628906 +/- 0.000000 | 0.255859 +/- 0.000000 | 153008.0 +/- 43497.3 |

The practical trade-off is stable across all bits: `full_kv` saves much more KV cache, but the extra V compression mainly appears as hidden-state degradation and transport error rather than as a large logit-cosine drop.  
実務上のトレードオフは全 bit で安定しています。`full_kv` は KV cache を大きく削減しますが、その追加の V 圧縮は主に hidden-state 劣化と transport error として現れ、logit cosine の大きな低下としては現れません。

### Plots / 図

![Paper baseline trade-offs](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\qwen_captured_reported\plots\attention_tradeoffs_captured.png)

![Paper baseline mean plus SD](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\qwen_captured_reported\plots\attention_mean_pm_sd_captured.png)

![V breakage by bit](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\qwen_captured_reported\plots\attention_v_breakage_by_bit_sd.png)

## Statistical Treatment / 統計処理

The statistical summary uses `attention_trials_captured.csv` with `n = 4` per `(mode, bit)` group.  
統計処理は `attention_trials_captured.csv` を使い、各 `(mode, bit)` 群で `n = 4` です。

### Omnibus Comparison / 多群比較

| Metric | Test | Statistic | p-value |
| --- | --- | ---: | ---: |
| hidden cosine similarity | Kruskal-Wallis | 33.8627 | 9.44e-05 |
| attention output relative error | Kruskal-Wallis | 37.5473 | 2.10e-05 |
| logit cosine similarity | Kruskal-Wallis | 17.2946 | 4.43e-02 |

This cleanly separates the V-sensitive metrics from the score metric.  
これにより、V に敏感な指標と score 系の指標がきれいに分かれます。

### Bit-wise Pairwise Tests / bit ごとの対比較

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

## Audit: Google Blog vs Paper vs Qwen3.5-9B / 監査: Google ブログ vs 論文 vs Qwen3.5-9B

We provide a three-column audit table comparing the public Google blog messaging, the actual scope of the paper, and our Qwen3.5-9B captured replay.  
Google の広報ブログ、論文が実際に保証している範囲、そして Qwen3.5-9B の captured replay 実測を 3 列で並べた監査表を用意しています。

Audit artifacts / 監査 artifact:

- `artifacts/paper_baseline/google_blog_audit/metrics/google_blog_paper_qwen_audit.csv`
- `artifacts/paper_baseline/google_blog_audit/metrics/google_blog_paper_qwen_audit.md`
- `artifacts/paper_baseline/google_blog_audit/metrics/qwen_summary_stats.csv`
- `artifacts/paper_baseline/google_blog_audit/metrics/qwen_summary_stats.md`
- `artifacts/paper_baseline/google_blog_audit/plots/google_blog_paper_qwen_audit.png`

Audit conclusion / 監査の結論:

- The Google blog is directionally correct about KV-cache reduction.  
  Google ブログは KV cache 削減については方向として正しいです。
- It becomes too broad if read as "no accuracy loss in general runtimes."  
  ただし「一般的な runtime でも性能劣化なし」と読むと広すぎます。
- On Qwen3.5-9B captured replay, `full_kv` preserves score-like metrics much better than it preserves value transport.  
  Qwen3.5-9B captured replay では、`full_kv` は score 系指標の保持に比べて value transport の保持がかなり弱いです。
- This repository therefore argues that Google's public result framing is selectively benchmarked, or effectively cherry-picked, relative to broader runtime behavior.  
  したがって本リポジトリは、Google の対外的な結果提示は、より広い runtime 挙動に対しては選択的、すなわち実質的にチェリーピッキング的であると主張します。

![Google blog audit plot](H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\paper_baseline\google_blog_audit\plots\google_blog_paper_qwen_audit.png)

## Final Takeaway / 最終結論

Paper baseline conclusion: `key_only_random` preserves hidden geometry better than `full_kv` on Qwen3.5-9B captured replay.  
論文 baseline の結論: Qwen3.5-9B captured replay では `key_only_random` のほうが `full_kv` より hidden geometry を保ちます。

Mixed-bit `2.5 / 3.5` remain useful intermediate Pareto points.  
mixed-bit の `2.5 / 3.5` は中間 Pareto 点として依然有用です。

The statistical evidence and the audit table support the same qualitative claim: TurboQuant reduces KV cache size as intended, but the paper-faithful `full_kv` path damages the V-dependent token-output transport signal before it meaningfully damages the logit score signal.  
統計処理と監査表の両方が、同じ質的結論を支持しています。つまり、TurboQuant は意図どおり KV cache を削減しますが、論文忠実な `full_kv` 経路は、logit の score 信号を大きく壊す前に、V 依存の token-output transport 信号を壊します。
