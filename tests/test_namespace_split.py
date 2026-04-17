from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
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
    ARTIFACT_METADATA_SCHEMA_VERSION,
    PAPER_SCHEMA_KIND,
    RESEARCH_SCHEMA_KIND,
    build_turboquant_gguf_contract,
    build_uniform_turboquant_gguf_contract,
    build_turboquant_artifact_metadata,
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
    key_cfg = KeyResearchConfig(head_dim=128, bits_total=4, qjl_dim=96, rotation_policy="block_so8_learned")
    value_cfg = ValueResearchConfig(base_bits=2, high_bits=8, protected_fraction=0.2, low_rank_rank=4)
    kv_cfg = key_cfg.to_kv_codec_config(value_cfg.to_value_codec_config())
    assert kv_cfg.head_dim == 128
    assert kv_cfg.key_bits == 4
    assert kv_cfg.qjl_dim == 96
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
        key_config=KeyResearchConfig(head_dim=128, bits_total=4, qjl_dim=64),
        value_config=ValueResearchConfig(base_bits=2, high_bits=8, protected_fraction=0.2, low_rank_rank=4),
    )
    path = tmp_path / "turboquant_config.research.json"
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    loaded = read_turboquant_config(path, expected_kind=RESEARCH_SCHEMA_KIND)
    assert loaded["schema_kind"] == RESEARCH_SCHEMA_KIND
    assert loaded["k_codec"]["bits_total"] == 4.0
    assert loaded["v_codec"]["low_rank_rank"] == 4
    assert loaded["k_codec"]["view_mode"] == "triality_proxy"
    assert loaded["k_codec"]["triality_mode"] == "triality_proxy"
    assert loaded["k_codec"]["triality_view"] == "vector"
    assert loaded["k_codec"]["qjl_dim"] == 64
    assert loaded["k_codec"]["stage1_effective_bits"] == 3.0
    assert loaded["k_codec"]["stage1_bitwidth_payload_dtype"] == "uint8"
    assert loaded["k_codec"]["sign_pack_format"] == "int8_unpacked_binary"
    assert loaded["k_codec"]["views"] == ["vector", "spinor_plus_proxy", "spinor_minus_proxy"]
    assert loaded["view_selection"] == "report_all"
    try:
        read_turboquant_config(path, expected_kind=PAPER_SCHEMA_KIND)
    except ValueError as exc:
        assert "Expected schema kind" in str(exc)
    else:
        raise AssertionError("cross-load between paper and research schema should fail")


def test_artifact_metadata_explicitly_separates_total_and_stage1_bits() -> None:
    metadata = build_turboquant_artifact_metadata(
        total_bits=3.5,
        qjl_bits=1,
        qjl_dim=128,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=128,
        allocation=ChannelBitAllocation.preset(effective_bits=2.5, width=128),
    )
    assert metadata["tq_schema_version"] == ARTIFACT_METADATA_SCHEMA_VERSION
    assert metadata["tq_total_bits"] == 3.5
    assert metadata["tq_stage1_effective_bits"] == 2.25
    assert metadata["tq_qjl_bits"] == 1
    assert metadata["tq_runtime_bits_per_channel"] == 3.25
    assert metadata["tq_triality_mode"] == "triality_proxy"
    assert metadata["tq_stage1_allocation_scheme"] == "magnitude-topk"


def test_build_turboquant_gguf_contract_uses_strict_tq_arrays() -> None:
    metadata = build_turboquant_artifact_metadata(
        total_bits=3.5,
        qjl_bits=1,
        qjl_dim=128,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=128,
        allocation=ChannelBitAllocation.preset(effective_bits=2.5, width=128),
    )
    contract = build_turboquant_gguf_contract([metadata, metadata])
    assert contract["tq_schema_version"] == ARTIFACT_METADATA_SCHEMA_VERSION
    assert contract["tq_total_bits"] == [3.5, 3.5]
    assert contract["tq_runtime_bits_per_channel"] == [3.25, 3.25]
    assert contract["tq_triality_mode"] == ["triality_proxy", "triality_proxy"]
    assert contract["tq_triality_view"] == ["vector", "vector"]


def test_build_uniform_turboquant_gguf_contract_expands_one_layer_payload() -> None:
    metadata = build_turboquant_artifact_metadata(
        total_bits=3.5,
        qjl_bits=1,
        qjl_dim=128,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=128,
        allocation=ChannelBitAllocation.preset(effective_bits=2.5, width=128),
    )
    contract = build_uniform_turboquant_gguf_contract(artifact_metadata=metadata, num_layers=3)
    assert contract["tq_qjl_bits"] == [1, 1, 1]
    assert contract["tq_rotation_seed"] == [17, 17, 17]
    assert contract["tq_sign_pack_format"] == ["int8_unpacked_binary"] * 3
