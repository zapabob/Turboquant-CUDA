# 2026-03-27 TurboQuant faithful rebuild key-only analysis Codex

## Summary

`arXiv:2504.19874` に合わせて `hub_Qwen3.5-9B-SOT` 側の TurboQuant コアを再整理し、
offline-first の key-only vs full-KV 評価パイプラインを追加した。

今回の主眼は runtime の text 品質改善そのものではなく、以下を定量化できる状態を作ることだった。

1. Stage 1 と Stage 2 を paper-faithful に分離する
2. key-only と full-KV の誤差源を offline で切り分ける
3. SciPy の要約統計量と error bar 付きの図を自動生成する
4. README に最新の定量結果とボトルネックを反映する

## Main code changes

### Canonical core

- `turboquant/types.py`
  - `TurboQuantProdConfig.resolved_qjl_dim()` の既定値を `dim` に変更
  - `QuantizedMSEBatch`, `QJLSketch`, `QuantizedProdBatch` に `total_bits()` 系を追加
- `turboquant/lloyd_max.py`
  - decision boundary 取得用の `decision_boundaries_tensor()` を追加
- `turboquant/turboquant_mse.py`
  - `sphere-lloyd-max` 以外の codebook kind を拒否
  - silent cast を避けるため、device/dtype を明示検証
  - argmin ではなく Lloyd-Max boundary に基づく bucketize を使用
- `turboquant/turboquant_prod.py`
  - `b-1` bit MSE + `1` bit QJL の前提を明示
  - input tensor の device/dtype を明示検証
- `turboquant/qjl.py`
  - zero residual を安全に処理
  - device/dtype の整合性を明示検証

### Mixed-bit policy

- `turboquant/allocation.py`
  - mixed-bit preset に `1.5` bit を追加
  - key 側の mixed total bits から QJL 1bit を差し引けるように準備
- `turboquant/kv_codec.py`
  - key の mixed-bit は `total_bits - qjl_bits` を Stage 1 allocation に渡すよう修正
  - `key_storage_bits()` / `value_storage_bits()` を追加

### Offline analysis

- `turboquant/analysis.py`
  - synthetic / captured KV の layer replay
  - exact / key_only / full_kv の比較
  - scaled attention output を使った hidden-state proxy 評価
  - `logit_cosine`, `hidden_cosine`, `memory_ratio_vs_exact`, `prefill_seconds`, `decode_seconds` 等の集計
  - layer threshold summary の生成
- `scripts/validate_attention_scores.py`
  - wide trial CSV, long metric CSV, summary CSV, threshold CSV, markdown summary を出力
- `scripts/test_synthetic.py`
  - Stage 1, Stage 2, mixed-bit Stage 2, synthetic replay をまとめて出力
- `scripts/export_report.py`
  - Matplotlib の `synthetic_errorbars.png`
  - Matplotlib / Plotly の `attention_tradeoffs.png`, `attention_tradeoffs.html`
  - `artifacts/reports/summary.md`

### Cache backend and public surface

- `turboquant/hf_cache.py`
  - `TurboQuantCacheBackend(..., quantize_values=False)` を既定に変更
  - key-only path では raw V を保持するように変更
- `turboquant/__init__.py`
  - analysis summary helper を export

### Tests

- 追加: `tests/test_analysis.py`
- 更新:
  - `tests/test_qjl.py`
  - `tests/test_turboquant_prod.py`
  - `tests/test_kv_codec.py`
  - `tests/test_rotation.py`

## Important mathematical fix

最重要の修正は mixed-bit key policy の解釈だった。

`TurboQuantProd` は total bits `b` を

- Stage 1: `b - 1`
- Stage 2: residual QJL `1`

に分ける。

そのため、key の `2.5` / `3.5` bit をそのまま Stage 1 allocation に入れると、
QJL 分を二重計上してしまう。

今回の修正では:

- total `2.5` bit key -> Stage 1 `1.5` bit mixed + QJL `1`
- total `3.5` bit key -> Stage 1 `2.5` bit mixed + QJL `1`

として扱うようにした。

## Commands run

### Test

```powershell
uv run python -m pytest -q
```

Result:

- `16 passed in 36.06s`

### Synthetic core validation

```powershell
uv run python scripts\test_synthetic.py --trials 4
```

Outputs:

- `artifacts/metrics/synthetic_metrics_trials.csv`
- `artifacts/metrics/synthetic_metrics.csv`
- `artifacts/metrics/synthetic_replay_trials.csv`
- `artifacts/metrics/synthetic_replay_metrics_long.csv`
- `artifacts/metrics/synthetic_replay_summary.csv`

### Offline replay summary

```powershell
uv run python scripts\validate_attention_scores.py --query-source synthetic --trials 4 --synthetic-layers 4
```

Outputs:

- `artifacts/metrics/attention_trials.csv`
- `artifacts/metrics/attention_metrics_long.csv`
- `artifacts/metrics/attention_summary.csv`
- `artifacts/metrics/attention_thresholds.csv`
- `artifacts/metrics/attention_summary.md`

### Report export

```powershell
uv run python scripts\export_report.py
```

Outputs:

- `artifacts/plots/synthetic_errorbars.png`
- `artifacts/plots/attention_tradeoffs.png`
- `artifacts/plots/attention_tradeoffs.html`
- `artifacts/reports/summary.md`

## Headline results

Current replay summary is synthetic-only in this repo, but it already shows the same qualitative split seen in the runtime harness:

- key-only preserves logits while keeping hidden-state cosine essentially at `1.0`
- full-KV saves far more memory, but hidden-state drift is materially worse

Representative means from `artifacts/metrics/attention_summary.csv`:

| Mode | Bits | Logit Cosine | Hidden Cosine | Memory / Exact |
| --- | ---: | ---: | ---: | ---: |
| key_only | 2.0 | 0.9166 | 1.0000 | 0.5391 |
| key_only | 3.5 | 0.9789 | 1.0000 | 0.5586 |
| key_only | 4.0 | 0.9911 | 1.0000 | 0.5703 |
| full_kv | 2.0 | 0.9166 | 0.9399 | 0.0742 |
| full_kv | 3.5 | 0.9789 | 0.9885 | 0.1133 |
| full_kv | 4.0 | 0.9911 | 0.9953 | 0.1367 |

## Interpretation

現時点の数理的ボトルネックは value path である。

- key-only と full-KV は logit cosine が同一
  - 理由: 両者とも K のみが logit を決めるため
- それでも hidden cosine は full-KV が一貫して低い
  - 理由: V の量子化誤差が softmax 後にそのまま hidden-state に乗るため

このため、次に狙うべき memory reduction は

1. key-only を標準 path に固定
2. key-only の mixed-bit `2.5 / 3.5` を優先評価
3. value quantization は別枝で selective / layer-aware に扱う

という順序が妥当。

## Notes

- `scripts/validate_attention_scores.py` の captured path は、`artifacts/kv` に key/value 両方がある場合に有効
- 現在の repo では `artifacts/kv` に十分な captured pair が揃っていないため、今回の report は synthetic replay を採用
- sibling runtime harness `qwen35_rtx3060` 側ではすでに key-only default が入っているため、今回は hub 側の canonical core と analysis を優先した
