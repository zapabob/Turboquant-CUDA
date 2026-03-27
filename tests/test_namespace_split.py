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
from turboquant.schema import (
    PAPER_SCHEMA_KIND,
    RESEARCH_SCHEMA_KIND,
    build_paper_turboquant_config,
    build_research_turboquant_config,
    read_turboquant_config,
)


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


def test_paper_schema_roundtrip(tmp_path) -> None:
    payload = build_paper_turboquant_config(bit_grid=[2.0, 2.5, 3.5, 4.0], dim=128)
    path = tmp_path / "turboquant_config.paper.json"
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    loaded = read_turboquant_config(path, expected_kind=PAPER_SCHEMA_KIND)
    assert loaded["schema_kind"] == PAPER_SCHEMA_KIND
    assert loaded["evaluation_grid"]["bit_grid"] == [2.0, 2.5, 3.5, 4.0]
    assert loaded["mode_configs"]["key_only_random"]["mixed_bit_policy"]["presets"]["2.5"]["high_count"] == 32


def test_research_schema_roundtrip_and_cross_load_rejection(tmp_path) -> None:
    payload = build_research_turboquant_config(
        key_config=KeyResearchConfig(head_dim=128, bits_total=4),
        value_config=ValueResearchConfig(base_bits=2, high_bits=8, protected_fraction=0.2, low_rank_rank=4),
    )
    path = tmp_path / "turboquant_config.research.json"
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    loaded = read_turboquant_config(path, expected_kind=RESEARCH_SCHEMA_KIND)
    assert loaded["schema_kind"] == RESEARCH_SCHEMA_KIND
    assert loaded["k_codec"]["bits_total"] == 4
    assert loaded["v_codec"]["low_rank_rank"] == 4
    assert loaded["k_codec"]["view_mode"] == "triality_proxy"
    assert loaded["k_codec"]["views"] == ["vector", "spinor_plus_proxy", "spinor_minus_proxy"]
    assert loaded["view_selection"] == "report_all"
    try:
        read_turboquant_config(path, expected_kind=PAPER_SCHEMA_KIND)
    except ValueError as exc:
        assert "Expected schema kind" in str(exc)
    else:
        raise AssertionError("cross-load between paper and research schema should fail")
