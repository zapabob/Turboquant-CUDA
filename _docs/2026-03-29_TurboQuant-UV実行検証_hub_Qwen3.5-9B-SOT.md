# 実装ログ: TurboQuant UV / 論文 baseline / Triality CUDA 検証

- **日付 (UTC 相当)**: 2026-03-29
- **ワークツリー**: hub_Qwen3.5-9B-SOT
- **目的**: プラン「TurboQuant UV 実行マップ」の To-do を実コマンドで検証

## 実施内容

1. **環境**
   - `uv sync --extra cu128 --extra dev` → 成功
   - 続けて `uv sync --extra hf_qwen`（captured 検証用）
   - `scripts/env_check.py` → `artifacts/reports/env_check.txt` 更新。最終状態: torch 2.11.0+cu128、`cuda_available: True`、`target_cuda_match: True`、`transformers` / `bitsandbytes` 導入後 `status: ok`
   - `import torch` 一行確認: `cuda_available True`, `cuda 12.8`

2. **論文 baseline**
   - `paper_validate_synthetic.py --trials 8` → exit 0
   - `paper_validate_attention.py --trials 2 --synthetic-layers 2` → exit 0（README の 8×4 は時間超過のため縮小）
   - `paper_validate_captured_qwen.py --kv-dir artifacts\kv --trials 1 --max-layers 1 --bits 2,3` → exit 0

3. **研究 Triality（CUDA）**
   - `run_triality_full_pipeline.py`  
     `--kv-dir artifacts\kv --max-layers 1 --bits 2,3 --train-steps 20 --trials 1`  
     `--train-output-dir artifacts\research_extension\triality_plan_smoke_train`  
     `--eval-output-dir artifacts\research_extension\triality_plan_smoke_eval`  
     → train `--device cuda`、eval `--eval-device cuda` で exit 0（評価部 ~504s + 全体 ~10min 規模）

## 備考

- 初回 PyTorch ロードが環境によって数十秒かかることがある
- eval 中 scipy Wilcoxon で `invalid value encountered in scalar divide` 警告（サンプル少のスモーク時）

## 再現コマンド（抜粋）

```powershell
Set-Location H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT
uv sync --extra cu128 --extra dev --extra hf_qwen
uv run python scripts\env_check.py
uv run python scripts\paper_validate_synthetic.py --trials 8
uv run python scripts\run_triality_full_pipeline.py --kv-dir artifacts\kv --max-layers 1 --bits 2,3 --train-steps 20 --trials 1
```
