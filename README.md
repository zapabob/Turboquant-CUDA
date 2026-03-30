# TurboQuant CUDA（Qwen3.5-9B）

PyTorch 正系の TurboQuant 論文再現、Qwen3.5-9B の captured replay、K/V 分離の研究拡張（Triality 等）をまとめたリポジトリです。  
Windows + `uv` + Python **3.12.x** を前提にしています。

## 立場（要約）

- TurboQuant の KV 削減効果は実在する。  
- 一方、私たちの Qwen3.5-9B captured replay では、論文忠実な `full_kv` は **logit 系より V 依存の hidden / transport 指標の方が先に崩れやすい**。  
- 詳細な数値・統計・Google ブログ監査は本文後半の表と `artifacts/paper_baseline/` を参照。

## リポジトリ構成

| 層 | 役割 |
| --- | --- |
| `turboquant.paper_baseline` | 論文忠実 Stage 1 / Stage 2（PyTorch のみ） |
| `turboquant.research_extension` | K/V codec、V 感度、protected-V、low-rank、Triality proxy |
| `turboquant.adapters.hf_qwen` | 任意: HF / Qwen の KV キャプチャと replay |

正系の検証順序は **synthetic → attention → captured**（オフラインが正しいことを先に固める）。

## ライセンス

Apache License 2.0 — [LICENSE](LICENSE)

## 設定ファイルの二系統

- `turboquant_config.paper.json` — 論文 baseline / HF replay 向け  
- `turboquant_config.research.json` — 研究用・将来の sidecar 向け

---

## 環境セットアップ

**必須**: リポジトリルートは **`hub_Qwen3.5-9B-SOT`**（`pyproject.toml` があるディレクトリ）。親フォルダで `uv run` すると失敗します。

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.12.9
uv venv --python 3.12.9
uv sync --extra cu128 --extra dev --extra hf_qwen
uv run python scripts\env_check.py
```

- CUDA 版 PyTorch は **`--extra cu128`** が必要。付けないと CPU 版になり `torch.cuda.is_available()` が `False` になりやすい。  
- **グローバルな `py -3` だけ**でプロジェクトスクリプトを走らせない（未対応バージョン・別 torch と混ざる）。**`uv run python ...`** を使う。  
- 親ディレクトリから実行する場合: `uv run --project hub_Qwen3.5-9B-SOT python scripts\...`

代替: `.\scripts\bootstrap_uv.ps1 -PythonVersion 3.12.9 -TorchExtra cu128`  
CUDA が既に合っていれば `.\scripts\bootstrap_uv.ps1 -SkipSyncIfCudaReady`

本番向け一括: `.\scripts\run_production_tests.ps1`（`env_check` + `pytest`）

---

## クイックスタート（オフライン検証）

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

## 本番フロー: KV キャプチャ → 論文 baseline → Triality

### 1. Qwen KV キャプチャ

- **`--weight-load none`**: BitsAndBytes は使わず、bf16 等でフル重みロード。`from_pretrained` には **`device_map="auto"`**（accelerate）が付与される。  
- **`--weight-load 4bit` / `8bit`**: `BitsAndBytesConfig` + `device_map="auto"`。

ローカル重み（`config.json` + safetensors）の例:

```powershell
uv run python scripts\capture_qwen_kv.py `
  --weight-load none --dtype bfloat16 --trust-remote-code `
  --model-id "H:\Qwen3.5-9B-official-hf" `
  --output-dir artifacts\kv_full_bf16 --max-length 96
```

Hub から読む場合は `--model-id Qwen/Qwen3.5-9B` 等に差し替え。  
既定 `--model-id` は `turboquant.runtime` の `LOCAL_CAPTURE_MODEL_PATH`（環境に合わせて確認）。

**VRAM**: 9B を bf16 でフルロードする場合、おおむね **18GB 級**。10GB 級 GPU では OOM しやすい。

出力は `artifacts\kv_full_bf16\<capture_id>\capture_manifest.json` と各層 `layer_*_{key,value}.pt`（複数プロンプトなら親ディレクトリがルート）。

VRAM が厳しい場合は 4bit ロード（論文の「非量子化フル重み」条件とは別）:

```powershell
uv run python scripts\capture_qwen_kv.py --weight-load 4bit --max-length 96
```

### 2. Captured 上の論文 baseline

```powershell
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv_full_bf16
```

オプションは `uv run python scripts\paper_validate_captured_qwen.py -h` で確認。

### 3. Triality フルパイプライン（学習 → 評価）

[`scripts/run_triality_full_pipeline.py`](scripts/run_triality_full_pipeline.py) は **毎回** `research_train_k_triality.py` を実行してから `research_validate_k_triality.py` を実行する。

```powershell
$env:PYTHONUNBUFFERED = "1"
uv run python scripts\run_triality_full_pipeline.py `
  --kv-dir artifacts\kv_full_bf16 `
  --train-output-dir artifacts\research_extension\triality_full_train_prod_bf16 `
  --eval-output-dir artifacts\research_extension\triality_full_eval_prod_bf16
```

**既定の「フル」の意味**

- `--max-layers` 省略（0）→ **全キャプチャ層**を対象（`max_layers <= 0` はフィルタなし）。  
- `--bits` 省略 → `2,2.5,3,3.5,4,8`（**8 はリポジトリ拡張**。論文表だけなら `--bits 2,2.5,3,3.5,4`）。  
- `--train-steps` 既定 60、`--trials` 既定 3。

`--` 以降は評価スクリプトへ転送（例: パイプライン経由で評価にだけオプションを付ける）:

```powershell
uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv_full_bf16 -- --resume
```

**注意**: 上記でも **学習フェーズは毎回走る**。学習済みの **`rotations/`** のみから評価を続ける場合は、パイプラインではなく **`research_validate_k_triality.py` を直接**呼ぶ（下記）。

### 4. Triality 評価の再開（`--resume`）

長時間の評価が途中で切れた場合、`metrics/triality_trials_partial.csv` と `metrics/eval_resume_state.json` から再開できる。

- **初回と同一**の `--kv-dir` / `--rotation-dir` / `--bits` / `--max-layers` / `--trials` / `--eval-device` / `--output-dir` に **`--resume`** を付ける。  
- **`bundle_keys` はキャプチャディレクトリの解決済みパスで正規化**されているため、相対・絶対の `--kv-dir` 混在でも fingerprint は一致しやすい（実装: `k_triality._bundle_keys_filtered`）。

```powershell
$env:PYTHONUNBUFFERED = "1"
uv run python scripts\research_validate_k_triality.py `
  --kv-dir artifacts\kv_full_bf16 `
  --rotation-dir artifacts\research_extension\triality_full_train_prod_bf16\rotations `
  --bits 2,2.5,3,3.5,4,8 `
  --max-layers 0 `
  --trials 3 `
  --eval-device cuda `
  --output-dir artifacts\research_extension\triality_full_eval_prod_bf16 `
  --resume
```

- 頭からやり直す: `--force-fresh`（`--resume` と併用で state を無視）。  
- 途中経過のコピー: `eval_output/checkpoints/cp_*`、既定のローリング間隔は `--checkpoint-interval-seconds`（既定 300）。短くすると停止時の損失が減る。

**重要**: **同じ `--output-dir` に対して**、`--resume` なしの評価とパイプラインを**同時に複数起動しない**（部分 CSV と state が競合し、再開が壊れる）。

**完了の目印**: `metrics/triality_summary_captured.md` / `.csv`、`triality_trials_captured.csv`、統計系 CSV/MD、`eval_status.json` の `last_completed_stage` が `finish`。

評価ループ中は `eval_status.json` の `last_completed_stage` が進まない（長い `replay_or_load_trials` のあとにまとめて `write_csv_md` 等が走る）。途中終了時は `current_stage` が `replay_or_load_trials` のまま残りうる。

### 学習済み回転だけ評価（本番ワンショット例）

```powershell
uv run python scripts\research_validate_k_triality.py `
  --kv-dir "D:\path\to\kv_root" `
  --rotation-dir "D:\path\to\train_out\rotations" `
  --eval-device cuda `
  --output-dir artifacts\research_extension\triality_full_eval
```

---

## K 側ビットグリッド

多くのスクリプトの既定 `--bits` は **`2,2.5,3,3.5,4,8`**。整数 **8** は拡張。論文寄りの速いスイープは `--bits 2,2.5,3,3.5,4`。`bits=8` の Lloyd–Max 初回ビルドは head dim 128 で冷キャッシュ時に遅くなりうる。

---

## 主なスクリプト

| 区分 | パス |
| --- | --- |
| 論文 baseline | `scripts/paper_validate_synthetic.py`, `paper_validate_attention.py`, `paper_validate_captured_qwen.py` |
| 研究拡張 | `scripts/research_validate_v_codecs.py`, `research_value_sensitivity.py`, `run_triality_full_pipeline.py`, `research_train_k_triality.py`, `research_validate_k_triality.py` |
| HF / Qwen | `scripts/capture_qwen_kv.py`, `validate_attention_scores.py`, `export_report.py`, `env_check.py`, `benchmark_encode_decode.py` |

---

## Qwen3.5-9B captured baseline（2–8 bit、Mean ± SD と統計）

### データソース

- **拡張グリッド（2, 2.5, 3, 3.5, 4, 8）** の集約指標と統計は、  
  `artifacts/paper_baseline/qwen_captured_full_bf16/metrics/attention_trials_captured.csv` の **trial 行**を対象に算出した。  
  各 **(mode, bit_setting)** について **n = 96**（4 プロンプト × 8 層 × 3 trials；キャプチャ構成が変われば n も変わる）。
- **論文レンジのみのサマリ**（別 run）: `artifacts/paper_baseline/qwen_captured_reported/`（表は過去の `n = 4` 集計向け）。

モード: `key_only_random`, `full_kv`（ほか `exact` は集約表外）。  
Mixed-bit: `2.5` = 32ch@3bit + 96ch@2bit、`3.5` = 32ch@4bit + 96ch@3bit。  
**8** はリポジトリ既定グリッドの拡張ビット。

### Mean ± SD（trial 集約、`qwen_captured_full_bf16`）

Logit cosine はこのキャプチャ集合では **key_only と full_kv で層別 trial 値が一致**するため、下表では **1 列**にまとめた。

| Bits | Logit cosine | Hidden cosine (key_only) | Hidden cosine (full_kv) | Memory / exact (KO) | Memory / exact (FV) | Attn out rel err (KO) | Attn out rel err (FV) | Memory bits (KO) | Memory bits (FV) |
| ---: | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2 | 0.997314 ± 0.003554 | 0.999756 ± 0.003794 | 0.940959 ± 0.003962 | 0.566406 | 0.130859 | 0.027470 ± 0.021959 | 0.339172 ± 0.006291 | 338720 ± 83829 | 78256 ± 19367 |
| 2.5 | 0.998494 ± 0.003982 | 0.998779 ± 0.004099 | 0.958700 ± 0.003791 | 0.574219 | 0.146484 | 0.027599 ± 0.029244 | 0.285421 ± 0.006102 | 343392 ± 84985 | 87600 ± 21680 |
| 3 | 0.999349 ± 0.003872 | 0.999715 ± 0.003168 | 0.982625 ± 0.003395 | 0.597656 | 0.193359 | 0.014811 ± 0.013992 | 0.184458 ± 0.004565 | 357408 ± 88454 | 115632 ± 28617 |
| 3.5 | 0.999308 ± 0.003353 | 1.000285 ± 0.003896 | 0.988403 ± 0.003179 | 0.605469 | 0.208984 | 0.015692 ± 0.016130 | 0.151886 ± 0.004057 | 362080 ± 89610 | 124976 ± 30930 |
| 4 | 0.999552 ± 0.004312 | 0.999959 ± 0.003823 | 0.995524 ± 0.003352 | 0.628906 | 0.255859 | 0.006952 ± 0.005955 | 0.096430 ± 0.002715 | 376096 ± 93079 | 153008 ± 37867 |
| 8 | 1.000244 ± 0.004563 | 1.000000 ± 0.003802 | 0.999715 ± 0.003854 | 0.753906 | 0.505859 | 0.002421 ± 0.001509 | 0.029344 ± 0.000504 | 450848 ± 111579 | 302512 ± 74868 |

Memory ratio 列は設定上定数のため SD = 0。Memory bits は層・シーケンス長のばらつきで SD > 0。

### エラーバー付きグラフ（HTML）

静的 PNG が無い場合でも、次をブラウザで開くと **トレードオフ曲線＋誤差表示（実装依存で error bar / band）** が見られる。

- [attention_tradeoffs_captured.html](artifacts/paper_baseline/qwen_captured_full_bf16/plots/attention_tradeoffs_captured.html)
- [attention_runtime_tradeoffs_captured.html](artifacts/paper_baseline/qwen_captured_full_bf16/plots/attention_runtime_tradeoffs_captured.html)

PNG を別途吐き出す場合は `paper_validate_captured_qwen.py` の出力先（例: `qwen_captured_reported/plots/`）を参照。

![Paper baseline trade-offs（reported run、PNG がある場合）](artifacts/paper_baseline/qwen_captured_reported/plots/attention_tradeoffs_captured.png)

![Paper baseline mean ± SD（reported run）](artifacts/paper_baseline/qwen_captured_reported/plots/attention_mean_pm_sd_captured.png)

![V breakage by bit（reported run）](artifacts/paper_baseline/qwen_captured_reported/plots/attention_v_breakage_by_bit_sd.png)

---

## 統計処理（`qwen_captured_full_bf16`、6 bit × 2 mode）

### 多群比較: Kruskal–Wallis（trial 単位、12 群）

群 = `(key_only_random | full_kv) × (2, 2.5, 3, 3.5, 4, 8)`。Python `scipy.stats.kruskal`、各群 n = 96。

| Metric | H statistic | p-value |
| --- | ---: | ---: |
| hidden_cosine_similarity | 896.32 | ≈ 3.84 × 10⁻¹⁸⁵ |
| attention_output_relative_error | 1029.70 | ≈ 7.78 × 10⁻²¹⁴ |
| logit_cosine_similarity | 55.40 | ≈ 6.55 × 10⁻⁸ |

Hidden / attention error は **bit と mode の組み合わせ**で分布が強く分かれる。Logit も 12 群間で有意だが、上表のとおり **同一 bit 内では KO/FV の trial 分布が一致**する。

### 二元比較: Mann–Whitney U（key_only_random vs full_kv、各 bit）

同一 bit 内で両 mode の trial を二群比較（two-sided）。

**hidden_cosine_similarity**

| Bits | p-value | mean (key_only) | mean (full_kv) |
| ---: | --- | ---: | ---: |
| 2 | ≈ 2.23 × 10⁻³⁴ | 0.999756 | 0.940959 |
| 2.5 | ≈ 2.88 × 10⁻³⁴ | 0.998779 | 0.958700 |
| 3 | ≈ 4.71 × 10⁻³⁵ | 0.999715 | 0.982625 |
| 3.5 | ≈ 1.13 × 10⁻³³ | 1.000285 | 0.988403 |
| 4 | ≈ 2.55 × 10⁻¹⁵ | 0.999959 | 0.995524 |
| 8 | ≈ 0.540 | 1.000000 | 0.999715 |

**attention_output_relative_error**

| Bits | p-value | mean (key_only) | mean (full_kv) |
| ---: | --- | ---: | ---: |
| 2 | ≈ 4.79 × 10⁻³³ | 0.027470 | 0.339172 |
| 2.5 | ≈ 4.47 × 10⁻³³ | 0.027599 | 0.285421 |
| 3 | ≈ 4.95 × 10⁻³³ | 0.014811 | 0.184458 |
| 3.5 | ≈ 4.50 × 10⁻³³ | 0.015692 | 0.151886 |
| 4 | ≈ 5.05 × 10⁻³³ | 0.006952 | 0.096430 |
| 8 | ≈ 4.56 × 10⁻³³ | 0.002421 | 0.029344 |

### 解釈（8 bit を含めて）

- **低〜中 bit**: hidden cosine は一貫して `full_kv` が `key_only_random` より低く、Mann–Whitney も極めて小さい p。  
- **8 bit**: hidden は両 mode がほぼ天井付近で **差が統計的に検出されない**（p ≈ 0.54）。一方 **attention_output_relative_error** は 8 bit でも full_kv の方が大きく、p は依然として有意（≪ 0.001）。  
- **logit cosine** はこのデータでは mode 間で同一分布のため、上記二元比較は省略。

### 参考: 旧サマリ（`qwen_captured_reported`、各 (mode, bit) で n = 4）

層平均サマリのみを並べた旧表向けの Kruskal（例）:

| Metric | H | p-value |
| --- | ---: | ---: |
| hidden cosine similarity | 33.86 | 9.44e-05 |
| attention output relative error | 37.55 | 2.10e-05 |
| logit cosine similarity | 17.29 | 4.43e-02 |

Holm 補正後の bit ごと pairwise は n が小さく保守的になりやすい — **omnibus と上記 trial 集約**を併用する。

---

## 監査: Google ブログ vs 論文 vs Qwen3.5-9B

- `artifacts/paper_baseline/google_blog_audit/metrics/google_blog_paper_qwen_audit.csv`（ほか `.md`, `qwen_summary_stats.*`）  
- `artifacts/paper_baseline/google_blog_audit/plots/google_blog_paper_qwen_audit.png`

![Google blog audit plot](artifacts/paper_baseline/google_blog_audit/plots/google_blog_paper_qwen_audit.png)

---

## 最終メッセージ

Captured replay では **`key_only_random` の方が `full_kv` より hidden geometry を保ちやすい**（低〜中 bit）。**8 bit 拡張点**では hidden は mode 差が統計的に潰れうる一方、attention 出力相対誤差はなお full_kv 側が大きい。mixed 2.5 / 3.5 は中間 Pareto として有用。  
TurboQuant の KV 削減は実在する一方、論文忠実 `full_kv` は **score 系より V 依存の transport 側が先に劣化しやすい** — trial 集約・多群比較・監査表が同じ質的結論を支持する。

---

## 再現性メモ

実験 artifact には、可能な範囲で **model / tokenizer / seed / dtype / device / 量子化設定 / prompt hash / タイムスタンプ / パッケージ版本** を残す（`AGENTS.md` 参照）。
