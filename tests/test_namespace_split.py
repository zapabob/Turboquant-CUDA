from __future__ import annotations

import torch

from turboquant.paper_baseline import (
    PAPER_BASELINE_MODES,
    PaperMSEConfig,
    PaperMixedBitPolicy,
    PaperProdConfig,
    PaperTurboQuantMSE,
    PaperTurboQuantProd,
    evaluate_paper_attention_grid,
)
from turboquant.research_extension import KeyResearchConfig, ValueResearchConfig


def test_paper_mixed_bit_policy_uses_explicit_quarter_split() -> None:
    policy = PaperMixedBitPolicy.for_total_bits(total_bits=2.5, dim=128)
    assert policy.low_bits == 2
    assert policy.high_bits == 3
    assert policy.high_count == 32


def test_paper_prod_wrapper_hides_transport_decode() -> None:
    quantizer = PaperTurboQuantProd(PaperProdConfig(dim=8, bits_total=2))
    assert not hasattr(quantizer, "transport_decode")


def test_paper_attention_grid_filters_to_baseline_modes() -> None:
    keys = torch.randn((1, 1, 8, 8), dtype=torch.float32)
    values = torch.randn((1, 1, 8, 8), dtype=torch.float32)
    rows = evaluate_paper_attention_grid(
        dataset="test",
        keys=keys,
        values=values,
        trial=0,
        layer_idx=0,
        bit_grid=[2.0],
    )
    assert rows
    assert {row["mode"] for row in rows}.issubset(PAPER_BASELINE_MODES)


def test_paper_mse_wrapper_uses_random_haar_only() -> None:
    quantizer = PaperTurboQuantMSE(PaperMSEConfig(dim=8, bits=2))
    x = torch.randn((4, 8), dtype=torch.float32)
    encoded = quantizer.quantize(x)
    decoded = quantizer.dequantize(encoded)
    assert decoded.shape == x.shape


def test_research_config_conversion_keeps_kv_split_fields() -> None:
    key_cfg = KeyResearchConfig(head_dim=128, bits_total=4, rotation_policy="block_so8_learned")
    value_cfg = ValueResearchConfig(base_bits=2, high_bits=8, protected_fraction=0.2, low_rank_rank=4)
    kv_cfg = key_cfg.to_kv_codec_config(value_cfg.to_value_codec_config())
    assert kv_cfg.head_dim == 128
    assert kv_cfg.key_bits == 4
    assert kv_cfg.value_codec.base_bits == 2
    assert kv_cfg.value_codec.low_rank_rank == 4
