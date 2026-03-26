from __future__ import annotations

import torch

from turboquant.kv_codec import KVCodec, KVCodecConfig


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
