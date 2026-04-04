# 実装ログ — VRAM 多群比較 + Multiscreen×Triality ベクトル結合

| 項目 | 値 |
|------|-----|
| 記録日時（ローカル） | 2026-04-04 17:13:32 +09:00 |
| ワークツリー | `hub_Qwen3.5-9B-SOT` |
| リポジトリ根 | `H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT` |

## 概要

PyTorch 上の **オフライン KV リプレイ**で、複数評価モードの **peak VRAM・理論メモリ比・hidden cosine** をまとめて CSV / Markdown / PNG（SEM エラーバー）出力する。対象は **llama.cpp / Hypura 本体ではない**（Hypura は `rust/hypura` 別ビルド）。

## 実装内容

- **`scripts/research_vram_multigroup_qwen.py`**
  - 既定モード: `exact`, `key_only_random`, `key_only_block_so8_static`, `multiscreen_relevance`（Triality は回転必須のため既定から除外）。
  - `key_only_block_so8_triality_vector` と **`multiscreen_triality_vector`** の両方で `--rotation-dir`（または既定の `artifacts/research_extension/triality_full_train/rotations`）を要求。
  - Multiscreen 割当は `multiscreen_relevance` / `multiscreen_triality_vector` のとき `ms_alloc` を渡す。
  - 出力: `metrics/vram_trials.csv`, `vram_summary_by_mode.csv`, **`vram_summary_by_mode.md`**, `vram_run_meta.json`, `plots/vram_peak_mb_by_mode.png`, `plots/kv_memory_ratio_by_mode.png`, `plots/hidden_cosine_similarity_by_mode.png`。
  - **matplotlib**: `matplotlib.use("Agg")` を `pyplot` より前に設定（Windows + uv Python で Tk/Tcl 未配置でも CI/テストが落ちないように）。
- **`turboquant/research_extension/captured_kv_modes.py`**
  - `CAPTURED_KEY_EVAL_MODES` に `multiscreen_triality_vector` を追加。
  - `eval_captured_key_mode_row` から `evaluate_multiscreen_triality_vector_row` を呼ぶ分岐（学習済み `vector` artifact 必須）。
- **`turboquant/research_extension/k_triality.py`**
  - `MULTISCREEN_TRIALITY_VECTOR_MODE` / `evaluate_multiscreen_triality_vector_row`（Multiscreen relevance ビットマップ + TrialityProxyProd + `quantize_with_bitwidths`）。
- **`turboquant/research_extension/__init__.py`**
  - 上記シンボルの export。
- **`tests/test_vram_multigroup_smoke.py`**
  - CPU・2 モード smoke + 成果物ファイル存在確認。
  - ダミー `*.pt` 回転（layer 0, bits 3.0, view `vector`）で `multiscreen_triality_vector` 1 本。
- **`CLAUDE.md`**
  - VRAM スクリプト例（CUDA・`multiscreen_triality_vector` + `--rotation-dir`）。

## 検証

```powershell
cd H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT
uv run python -m pytest -q
```

最終確認時点: **66 passed, 1 skipped**（他モジュール含む全テスト）。

## 解釈メモ（運用）

- **`memory_ratio_vs_exact`**: KV の理論ビット比で下がりやすい。
- **`peak_vram_mb`**: リプレイ中のピーク割り当て。量子化が軽くても一時テンソルで大きく見えることがある。
- 傾向は **サマリー CSV/MD と 3 枚の図**で横並び比較すること。

---

## CoT（仮説 → 検証）

1. **仮説**: Triality を `--modes` に混ぜると回転ディレクトリが無いと即死する → **検証**: `_triality_path_for_modes` で必要モード集合の積を取り、欠落時は `SystemExit` で明示。
2. **仮説**: `multiscreen_triality_vector` は `ms_alloc` 無しだと `eval_captured_key_mode_row` が落ちる → **検証**: VRAM スクリプト側で `multiscreen_relevance` と同様に `ms_use` を付与。
3. **仮説**: テスト環境で Matplotlib が Tk を引くと Tcl 無しで落ちる → **検証**: `Agg` 強制後、`test_vram_multigroup_smoke.py` 全通過。

---

## なんj風（一行）

VRAMはピークでドンと出るけどビット比はスッと下がることもあるから、CSVと3枚の棒グラフ見比べないと「え、軽いのにVRAMやばい」みたいな釣りが発生するわ。**図表セットで殴れ。**
