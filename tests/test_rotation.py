from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.rotation import block_so8_from_skew, block_so8_rotation, random_rotation


def test_mixed_bit_preset_preserves_fraction_for_wider_heads() -> None:
    allocation = ChannelBitAllocation.preset(effective_bits=2.5, width=256)
    assert allocation.outlier_count == 64


def test_random_rotation_is_orthogonal() -> None:
    rotation = random_rotation(dim=16, seed=7, device=torch.device("cpu"), dtype=torch.float32)
    ident = rotation.transpose(0, 1) @ rotation
    assert torch.allclose(ident, torch.eye(16), atol=1e-5, rtol=1e-5)


def test_block_so8_rotation_is_orthogonal() -> None:
    rotation = block_so8_rotation(dim=16, seed=3, device=torch.device("cpu"), dtype=torch.float32)
    ident = rotation.transpose(0, 1) @ rotation
    assert torch.allclose(ident, torch.eye(16), atol=1e-5, rtol=1e-5)


def test_block_so8_from_skew_is_orthogonal() -> None:
    skew_blocks = torch.randn((2, 8, 8), dtype=torch.float32)
    rotation = block_so8_from_skew(skew_blocks, dtype=torch.float32)
    ident = rotation.transpose(0, 1) @ rotation
    assert torch.allclose(ident, torch.eye(16), atol=1e-5, rtol=1e-5)
