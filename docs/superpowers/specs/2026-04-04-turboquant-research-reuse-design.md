# TurboQuant Research Reuse Design

**Date:** 2026-04-04  
**Repo:** zapabob/Turboquant-CUDA  
**Status:** Approved

---

## 目的

1. TurboQuant 論文忠実実装と Triality SO(8) 回転を**他の研究実装が再利用できる形**に整備する
2. 任意の HuggingFace モデル（Llama、Mistral、Gemma 等）に差し込める**プラグインプロトコル**を追加する
3. Multiscreen の KV 重要度スコアリング（cosine + MiPE + trim-and-square）を**混合ビット割り当てポリシー**として統合する
4. `zapabob/Hypura` の推論パイプラインに接続できる**Hypura Bridge**を実装する

---

## アーキテクチャ概要

既存コードは**一切変更しない**。新規追加は全て「外側の境界」に留める。

```
turboquant/
├── compressor.py                           # NEW: KVCompressor プロトコル + CompressedKV 型
├── compressors/
│   ├── paper_baseline.py                   # NEW: PaperBaselineCompressor
│   └── triality.py                         # NEW: TrialityCompressor
├── integrations/
│   ├── hf_patcher.py                       # NEW: HFModelPatcher
│   └── hypura_bridge.py                    # NEW: HypuraBridgeExporter
├── adapters/
│   ├── hf_qwen/                            # 既存 (変更なし)
│   ├── hf_llama.py                         # NEW: Llama 用 HFAttentionModuleSpec
│   └── hf_mistral.py                       # NEW: Mistral 用 HFAttentionModuleSpec
├── research_extension/
│   ├── multiscreen_kv.py                   # NEW: Multiscreen KV relevance scoring
│   └── ...                                 # 既存 (変更なし)
├── allocation.py                           # EXTEND: make_bitwidths_from_relevance() 追加
└── ...                                     # 既存 (変更なし)

scripts/
├── research_validate_multiscreen_kv.py     # NEW: Multiscreen 比較検証スクリプト
└── export_hypura_bridge.py                 # NEW: Hypura bridge export CLI

notebooks/
└── quickstart_turboquant.ipynb             # NEW: Llama/Mistral/Qwen リファレンスノートブック
```

---

## Track 1: Multiscreen KV スコアリング統合

### 背景

`multiscreen-pytorch` の `ScreeningUnit` 信号フローを、TurboQuant の混合ビット割り当てにおける K 位置重要度スコアとして移植する。外部依存なし（コードをコピー移植）。

### 新規ファイル: `turboquant/research_extension/multiscreen_kv.py`

```python
"""Multiscreen-derived KV relevance scoring for mixed-bit allocation.

Ports the ScreeningUnit signal flow from multiscreen-pytorch/multiscreen/layers.py
as a parameter-free importance estimator for K positions.
"""

from __future__ import annotations
import torch


def normalize_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit-normalize along the last dimension. Shape: [..., d] -> [..., d]"""
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def build_mipe_factor(
    relative_positions: torch.Tensor,   # [query, key]
    s_e: torch.Tensor,                  # [head_dim]
    s_f: torch.Tensor,                  # [head_dim]
) -> torch.Tensor:
    """Compute bounded MiPE factor. Returns [query, key]."""
    scale = torch.exp(s_e).clamp_min(1.0)
    threshold = torch.exp(s_f).clamp_min(1.0)
    rel = relative_positions.to(dtype=scale.dtype)
    phase = rel.unsqueeze(-1) / scale.view(1, 1, -1)
    per_dim = torch.cos(phase)
    activated = rel.abs().unsqueeze(-1) <= threshold.view(1, 1, -1)
    mixed = torch.where(activated, per_dim, torch.ones_like(per_dim))
    return mixed.mean(dim=-1)


def trim_and_square(similarity: torch.Tensor, s_r: torch.Tensor) -> torch.Tensor:
    """Sparse relevance gate. High similarity -> high alpha, low -> zero."""
    r = torch.exp(s_r).clamp_min(0.0) + 1.0
    s = (similarity + 1.0) * 0.5
    return torch.clamp(1.0 - r * (1.0 - s), min=0.0).square()


def compute_k_relevance(
    q: torch.Tensor,   # [batch, heads, q_len, head_dim]
    k: torch.Tensor,   # [batch, heads, k_len, head_dim]
    s_e: torch.Tensor | None = None,
    s_f: torch.Tensor | None = None,
    s_r: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Multiscreen relevance scores for each K position.

    Returns:
        relevance: [batch, heads, k_len]
            Max relevance across all query positions for each K position.
            Higher = more important for quantization bit allocation.
    """
    head_dim = q.shape[-1]
    device, dtype = q.device, q.dtype

    if s_e is None:
        s_e = torch.zeros(head_dim, device=device, dtype=dtype)
    if s_f is None:
        s_f = torch.zeros(head_dim, device=device, dtype=dtype)
    if s_r is None:
        s_r = torch.zeros(1, device=device, dtype=dtype)

    q_unit = normalize_unit(q)   # [batch, heads, q_len, head_dim]
    k_unit = normalize_unit(k)   # [batch, heads, k_len, head_dim]

    # cosine similarity: [batch, heads, q_len, k_len]
    base_sim = torch.einsum("bhqd,bhkd->bhqk", q_unit, k_unit).clamp(-1.0, 1.0)

    q_len = q.shape[-2]
    k_len = k.shape[-2]
    q_pos = torch.arange(q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    rel_pos = q_pos[:, None] - k_pos[None, :]  # [q_len, k_len]

    mipe = build_mipe_factor(rel_pos, s_e, s_f)   # [q_len, k_len]
    similarity = (base_sim * mipe.view(1, 1, q_len, k_len)).clamp(-1.0, 1.0)

    alpha = trim_and_square(similarity, s_r)  # [batch, heads, q_len, k_len]

    # K 位置ごとに全 Query の最大 relevance を取る: [batch, heads, k_len]
    return alpha.max(dim=-2).values


def multiscreen_relevance_topk_indices(
    relevance: torch.Tensor,   # [batch, heads, k_len]
    outlier_count: int,
) -> torch.Tensor:
    """Return indices of top-k most relevant K positions per head.

    Shape: [batch, heads, outlier_count]
    """
    return torch.topk(relevance, k=outlier_count, dim=-1).indices
```

### 移植元対応表

| turboquant 側 | multiscreen-pytorch 側 |
|---|---|
| `normalize_unit` | `layers.py:15` |
| `build_mipe_factor` | `layers.py:36–57` |
| `trim_and_square` | `layers.py:60–70` |
| `compute_k_relevance` | `layers.py:96–138` の `compute_similarity` + forward |

### `allocation.py` への追加

既存 `make_bitwidths()` には触れず、2 つを追加：

```python
@classmethod
def from_multiscreen_relevance(
    cls,
    regular_bits: int,
    outlier_bits: int,
    outlier_count: int,
) -> "ChannelBitAllocation":
    """Build allocation config for Multiscreen relevance-based outlier selection.

    Pass the returned instance to make_bitwidths_from_relevance() with the
    actual relevance tensor to get per-position bitwidths.
    """
    return cls(
        regular_bits=regular_bits,
        outlier_bits=outlier_bits,
        outlier_count=outlier_count,
        selection_policy="multiscreen-relevance",
    )

def make_bitwidths_from_relevance(
    self,
    relevance: torch.Tensor,  # [k_len] — K 位置ごとのスカラースコア
) -> torch.Tensor:
    """Select outlier K positions by Multiscreen relevance score.

    Returns [k_len] int tensor: outlier_bits for high-relevance, regular_bits otherwise.
    """
    k_len = relevance.shape[-1]
    bitwidths = torch.full((k_len,), self.regular_bits, dtype=torch.long,
                           device=relevance.device)
    _, topk_idx = torch.topk(relevance.flatten(), k=self.outlier_count)
    bitwidths[topk_idx] = self.outlier_bits
    return bitwidths
```

### Triality との組み合わせ方

```
Triality SO(8) 回転 → compress_keys()
    ↓
compute_k_relevance(q, k) → ChannelBitAllocation.from_multiscreen_relevance()
    ↓
compress_keys(keys, layer_idx, allocation=multiscreen_alloc)
    ↓
混合ビット割り当て（高重要度 K に outlier_bits、低重要度に regular_bits）
```

`compress_keys` の `allocation` 引数が `None` のとき magnitude-topk（既存）、Multiscreen 割り当てを渡したとき重要度ベースになる。KVCompressor 内部は変更なし。

### 検証スクリプト: `scripts/research_validate_multiscreen_kv.py`

```
uv run python scripts/research_validate_multiscreen_kv.py \
    --captured-dir artifacts/captured_qwen/ \
    --output-dir artifacts/research_extension/multiscreen_kv/ \
    --key-bits 3 \
    --outlier-bits 4 \
    --outlier-ratio 0.25
```

比較モード:
- `exact` — 量子化なし
- `key_only_random` — ランダム Haar 回転
- `key_only_block_so8_static` — 静的 SO(8) 回転
- `key_only_block_so8_triality_vector` — Triality（既存）
- `multiscreen_relevance` — 新規

評価指標: `attention_metrics.py` の既存指標（attention score MSE、hidden cosine similarity、logit KL divergence）をそのまま流用。

### テスト要件

`tests/test_multiscreen_kv.py`:
- `compute_k_relevance` の shape 確認: `[batch, heads, k_len]`
- relevance スコアの範囲確認: `[0, 1]`
- `s_r` が大きいほどスパースになることの確認
- `make_bitwidths_from_relevance` の shape・値確認

---

## Track 2: KVCompressor プロトコル + HF Patcher + Hypura Bridge

### `turboquant/compressor.py` — プロトコル定義

```python
from __future__ import annotations
from typing import Any, Protocol
from dataclasses import dataclass
import torch
from turboquant.allocation import ChannelBitAllocation
from turboquant.types import QuantizedProdBatch, ProtectedValueBatch


@dataclass
class CompressedKV:
    """圧縮済み KV テンソルの共通コンテナ。"""
    quantized: QuantizedProdBatch | ProtectedValueBatch
    layer_idx: int
    head_dim: int
    meta: dict[str, Any]   # rotation_policy, view, bits 等


class KVCompressor(Protocol):
    """任意モデルの KV に適用できる圧縮器の共通インターフェース。"""

    def compress_keys(
        self,
        keys: torch.Tensor,     # [..., head_dim]
        layer_idx: int,
        allocation: ChannelBitAllocation | None = None,
    ) -> CompressedKV: ...

    def compress_values(
        self,
        values: torch.Tensor,   # [..., head_dim]
        layer_idx: int,
    ) -> CompressedKV: ...

    def decompress_keys(self, compressed: CompressedKV) -> torch.Tensor: ...
    def decompress_values(self, compressed: CompressedKV) -> torch.Tensor: ...

    def export_config(self) -> dict[str, Any]: ...
```

### `turboquant/compressors/paper_baseline.py`

`PaperTurboQuantProd` を `KVCompressor` プロトコルに適合させるラッパー。  
コンストラクタ: `PaperBaselineCompressor(config: PaperProdConfig)`

### `turboquant/compressors/triality.py`

`TrialityProxyProd` を `KVCompressor` プロトコルに適合させるラッパー。  
コンストラクタ: `TrialityCompressor(config: KeyResearchConfig, rotation_dir: Path)`  
`compress_keys` 内で対応する回転行列を `load_triality_proxy_rotations` で読み込む。

### `turboquant/integrations/hf_patcher.py`

```python
@dataclass
class HFAttentionModuleSpec:
    """モデル別のアテンション層パス定義。"""
    module_pattern: str    # e.g. "model.layers.{i}.self_attn"
    num_layers: int
    head_dim: int
    num_heads: int
    k_out_hook_attr: str   # hook で KV を捕捉するための属性名ヒント

class HFModelPatcher:
    """任意の HF モデルに KVCompressor を register_forward_hook で注入。"""

    def __init__(
        self,
        model,
        spec: HFAttentionModuleSpec,
        compressor: KVCompressor,
    ) -> None: ...

    def attach(self) -> None: ...   # hook を全対象層に登録
    def detach(self) -> None: ...   # hook を全て解除
    def get_compressed_cache(self) -> dict[int, CompressedKV]: ...
    # key: layer_idx, value: 最後の forward で得た CompressedKV
```

### `turboquant/integrations/hypura_bridge.py`

```python
class HypuraBridgeExporter:
    """Hypura の Rust 側が読める形式に config + 回転行列を出力。"""

    def export(
        self,
        compressor: KVCompressor,
        output_dir: Path,
        model_id: str,
        dtype: str = "bfloat16",
    ) -> None:
        # 出力ファイル:
        # output_dir/hypura_turboquant_config.json  — 量子化設定
        # output_dir/rotations/layer_{i}.pt          — 回転行列
        # output_dir/manifest.json                   — バージョン + モデルメタ
```

`manifest.json` フォーマット（Hypura Rust 側が検証用に読む）:

```json
{
  "turboquant_version": "0.1.0",
  "model_id": "...",
  "dtype": "bfloat16",
  "num_layers": 32,
  "head_dim": 128,
  "rotation_policy": "block_so8_triality",
  "triality_view": "vector",
  "config_path": "hypura_turboquant_config.json",
  "rotation_dir": "rotations/"
}
```

### モデル別アダプター（ロジックなし、設定値のみ）

```python
# turboquant/adapters/hf_llama.py
LLAMA_ATTENTION_SPEC = HFAttentionModuleSpec(
    module_pattern="model.layers.{i}.self_attn",
    num_layers=32,    # モデルサイズにより上書き
    head_dim=128,
    num_heads=32,
    k_out_hook_attr="k_proj",
)

# turboquant/adapters/hf_mistral.py
MISTRAL_ATTENTION_SPEC = HFAttentionModuleSpec(
    module_pattern="model.layers.{i}.self_attn",
    num_layers=32,
    head_dim=128,
    num_heads=32,
    k_out_hook_attr="k_proj",
)
```

### CLI スクリプト: `scripts/export_hypura_bridge.py`

```
uv run python scripts/export_hypura_bridge.py \
    --rotation-dir artifacts/research_extension/triality_full_train/rotations \
    --model-id Qwen/Qwen3.5-9B \
    --output-dir artifacts/hypura_bridge \
    --view vector
```

### ノートブック: `notebooks/quickstart_turboquant.ipynb`

1. 環境セットアップ（`uv sync`）
2. Qwen3.5-9B での KV キャプチャ → Triality 圧縮 → metrics
3. Llama モデルでの `HFModelPatcher` 使用例（`HFAttentionModuleSpec` を指定）
4. Multiscreen 混合ビット割り当ての例
5. Hypura Bridge エクスポート例

---

## 実装順序と依存関係

```
Step 1   turboquant/research_extension/multiscreen_kv.py 新規作成   (依存ゼロ)
Step 2   turboquant/allocation.py 拡張                              (multiscreen_kv に依存)
Step 3   turboquant/research_extension/__init__.py 更新             (multiscreen_kv に依存)
Step 4   tests/test_multiscreen_kv.py                               (Step 1-2 に依存)
Step 5   scripts/research_validate_multiscreen_kv.py                (Step 1-2 に依存)
Step 6   turboquant/compressor.py プロトコル定義                     (既存型に依存のみ)
Step 7   turboquant/compressors/paper_baseline.py                   (Step 6 に依存)
Step 8   turboquant/compressors/triality.py                         (Step 6 に依存)
Step 9   turboquant/integrations/hf_patcher.py                      (Step 6 に依存)
Step 10  turboquant/integrations/hypura_bridge.py                   (Step 6 に依存)
Step 11  turboquant/adapters/hf_llama.py + hf_mistral.py            (Step 9 に依存)
Step 12  scripts/export_hypura_bridge.py                            (Step 10 に依存)
Step 13  notebooks/quickstart_turboquant.ipynb                      (全ステップ完了後)
```

Step 1–5 が Track 1（Multiscreen）、Step 6–13 が Track 2。Step 1–5 は Track 2 と独立して先行可能。

---

## 既存コードへの影響

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `turboquant/allocation.py` | 追加のみ | `from_multiscreen_relevance()` + `make_bitwidths_from_relevance()` |
| `turboquant/research_extension/__init__.py` | 追加のみ | `compute_k_relevance`, `multiscreen_relevance_topk_indices` を export |
| その他既存ファイル | 変更なし | — |

---

## 注意事項

- `multiscreen_kv.py` は `multiscreen-pytorch` への依存を持たない（コードをコピー移植）
- `s_r=0` のとき `r=2`、線形 trim-and-square に相当（デフォルト動作）
- `HFModelPatcher` の hook は attention の forward 後に KV テンソルを捕捉する設計。モデルごとに KV の出力形状が異なる場合は `HFAttentionModuleSpec` の `k_out_hook_attr` で調整する
- Hypura Bridge は Python 側でオフライン生成のみ。Hypura Rust 側のオンライン統合は別タスク
- 全実験 artifact に再現性メタデータ（model_id、seed、dtype、device、config、timestamp）を含める（AGENTS.md 準拠）
