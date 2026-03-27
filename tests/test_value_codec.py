from __future__ import annotations

import torch

from turboquant.types import SensitivitySpec, ValueCodecConfig
from turboquant.value_codec import ProtectedValueCodec


def test_protected_value_codec_builds_expected_masks() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    values = torch.randn((1, 2, 8, 16), generator=generator, dtype=torch.float32)
    attention_weights = torch.softmax(
        torch.randn((1, 2, 3, 8), generator=generator, dtype=torch.float32),
        dim=-1,
    )
    codec = ProtectedValueCodec(
        dim=16,
        config=ValueCodecConfig(
            base_bits=3,
            protected_fraction=0.25,
            secondary_fraction=0.25,
            high_bits=8,
            sensitivity=SensitivitySpec(calibration_samples=8),
        ),
        rotation_seed=0,
        rotation_policy="block_so8_static",
        device="cpu",
        dtype="float32",
    )
    codec.calibrate(values, attention_weights=attention_weights)
    assert codec.exact_channel_mask is not None
    assert codec.high_precision_mask is not None
    assert int(codec.exact_channel_mask.sum().item()) == 16
    assert int(codec.high_precision_mask.sum().item()) == 16


def test_protected_value_codec_roundtrip_and_ratio() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    values = torch.randn((1, 2, 8, 16), generator=generator, dtype=torch.float32)
    codec = ProtectedValueCodec(
        dim=16,
        config=ValueCodecConfig(base_bits=3, protected_fraction=0.125, secondary_fraction=0.125, low_rank_rank=2),
        rotation_seed=0,
        rotation_policy="block_so8_static",
        device="cpu",
        dtype="float32",
    )
    codec.calibrate(values)
    encoded = codec.encode(values)
    decoded = codec.decode(encoded)
    ratio = codec.memory_ratio_vs_exact(encoded, values)
    assert decoded.shape == values.shape
    assert 0.0 < ratio <= 1.5
