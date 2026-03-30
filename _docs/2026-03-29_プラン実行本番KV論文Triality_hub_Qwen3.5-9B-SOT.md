# 実装ログ: プラン実行（bnb なし 9B キャプチャ + paper baseline + Triality フル）

- **日付**: 2026-03-29（作業完了時点の記録）
- **ワークツリー**: hub_Qwen3.5-9B-SOT

## コード変更

- [`scripts/capture_qwen_kv.py`](../scripts/capture_qwen_kv.py): `--weight-load none` 時に `device_map="auto"` を付与（accelerate、量子化なし）。

## 実行結果サマリ

### 1. KV キャプチャ（フル重み・bnb なし）

- **コマンド**: `uv run python scripts\capture_qwen_kv.py --weight-load none --dtype bfloat16 --trust-remote-code --model-id H:\Qwen3.5-9B-official-hf --output-dir artifacts\kv_full_bf16 --max-length 96`
- **結果**: exit **0**。4 プロンプトキャプチャ保存。ロード時に一部パラメータが CPU/ディスクオフロードのメッセージあり（`device_map=auto`）。
- **層数**: 各キャプチャ **8 層**（計 32 bundles）。

### 2. 論文 baseline（キャプチャ再生）

- **コマンド**: `uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv_full_bf16 --output-dir artifacts\paper_baseline\qwen_captured_full_bf16`
- **結果**: exit **0**。所要おおよそ **110 分**（6 ビット × 複数モード × 32 bundles × 3 trials が主因）。
- **成果物**: `artifacts\paper_baseline\qwen_captured_full_bf16\metrics\` と `plots\`。

### 3. Triality フルパイプライン

- **学習 + 評価起動**: `uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv_full_bf16 --train-output-dir artifacts\research_extension\triality_full_train_prod_bf16 --eval-output-dir artifacts\research_extension\triality_full_eval_prod_bf16`
- **学習**: 完了（144 行の training summary / rotation manifest 出力を確認）。
- **評価**: 長時間ジョブのためセッション上限で一度中断（exit -1）。`metrics\triality_trials_partial.csv` に途中経過あり（約 2100+ 行規模）。
- **再開**: バックグラウンドで以下を実行中または手元で完走待ち:

```powershell
Set-Location H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT
$env:PYTHONUNBUFFERED = '1'
uv run python scripts\research_validate_k_triality.py `
  --kv-dir artifacts\kv_full_bf16 `
  --rotation-dir artifacts\research_extension\triality_full_train_prod_bf16\rotations `
  --bits 2,2.5,3,3.5,4,8 --max-layers 0 --trials 3 --eval-device cuda `
  --output-dir artifacts\research_extension\triality_full_eval_prod_bf16 `
  --resume
```

- **完了の目安**: `artifacts\research_extension\triality_full_eval_prod_bf16\metrics\triality_summary_captured.csv` が生成され、ログに最終ステージ完了が出ること。

## 再現用ワンライナー（キャプチャ〜論文まで）

```powershell
Set-Location H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT
uv run python scripts\capture_qwen_kv.py --weight-load none --dtype bfloat16 --trust-remote-code --model-id H:\Qwen3.5-9B-official-hf --output-dir artifacts\kv_full_bf16 --max-length 96
uv run python scripts\paper_validate_captured_qwen.py --kv-dir artifacts\kv_full_bf16 --output-dir artifacts\paper_baseline\qwen_captured_full_bf16
```
