from __future__ import annotations

import torch

from turboquant.kv_codec import KVCodec, KVCodecConfig
from turboquant.types import ValueCodecConfig


def test_kv_codec_roundtrip_shapes() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    keys = torch.randn((1, 2, 8, 16), generator=generator)
    values = torch.randn((1, 2, 8, 16), generator=generator)
    codec = KVCodec(KVCodecConfig(head_dim=16, key_bits=3, value_bits=3))
    decoded_keys = codec.decode_keys(codec.encode_keys(keys))
    decoded_values = codec.decode_values(codec.encode_values(values))
    assert decoded_keys.shape == keys.shape
    assert decoded_values.shape == values.shape


def test_attention_estimator_returns_expected_shape() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    keys = torch.randn((1, 2, 8, 16), generator=generator)
    queries = torch.randn((1, 2, 3, 16), generator=generator)
    codec = KVCodec(KVCodecConfig(head_dim=16, key_bits=3, value_bits=3))
    scores = codec.estimator.turboquant(queries, codec.encode_keys(keys))
    assert scores.shape == (1, 2, 3, 8)


def test_kv_codec_reports_quantized_storage_bits() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)
    keys = torch.randn((1, 1, 4, 16), generator=generator, dtype=torch.float32)
    codec = KVCodec(KVCodecConfig(head_dim=16, key_bits=3, value_bits=3))
    encoded = codec.encode_keys(keys)
    assert codec.key_storage_bits(encoded) > 0


def test_kv_codec_calibrates_and_encodes_protected_values() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    keys = torch.randn((1, 2, 8, 16), generator=generator, dtype=torch.float32)
    values = torch.randn((1, 2, 8, 16), generator=generator, dtype=torch.float32)
    queries = torch.randn((1, 2, 3, 16), generator=generator, dtype=torch.float32)
    codec = KVCodec(
        KVCodecConfig(
            head_dim=16,
            key_bits=3,
            value_bits=3,
            rotation_policy="block_so8_learned",
            value_codec=ValueCodecConfig(base_bits=3, protected_fraction=0.25, secondary_fraction=0.25),
        )
    )
    codec.calibrate(keys=keys, values=values, queries=queries)
    encoded = codec.encode_protected_values(values)
    decoded = codec.decode_protected_values(encoded)
    assert decoded.shape == values.shape
    assert codec.protected_value_storage_bits(encoded) > 0
