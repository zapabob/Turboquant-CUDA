from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.rotation import (
    apply_fast_rotation,
    block_so8_from_skew,
    block_so8_rotation,
    fast_hadamard_rotation,
    fast_walsh_hadamard_transform,
    random_rotation,
    rotation_from_policy,
    so8_block_diagonal_rotation_metrics,
)


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


def test_so8_block_diagonal_rotation_metrics_near_zero_for_learned_blocks() -> None:
    skew_blocks = torch.randn((2, 8, 8), dtype=torch.float32)
    rotation = block_so8_from_skew(skew_blocks, dtype=torch.float32)
    ortho, det_err = so8_block_diagonal_rotation_metrics(rotation)
    assert ortho < 1e-4
    assert det_err < 1e-4


def test_so8_block_diagonal_rotation_metrics_identity_sixteen() -> None:
    eye = torch.eye(16, dtype=torch.float64)
    ortho, det_err = so8_block_diagonal_rotation_metrics(eye)
    assert ortho == 0.0
    assert det_err == 0.0


# ---------------------------------------------------------------------------
# Fast Hadamard (FWHT / D1*H*D2) tests
# ---------------------------------------------------------------------------


def test_fwht_is_orthogonal_power_of_two() -> None:
    """H^T H == I for the normalised WHT on a power-of-two dim."""
    dim = 16
    x = torch.eye(dim, dtype=torch.float32)
    hx = fast_walsh_hadamard_transform(x)
    ident = hx.transpose(0, 1) @ hx
    assert torch.allclose(ident, torch.eye(dim), atol=1e-5), (
        f"max err: {(ident - torch.eye(dim)).abs().max()}"
    )


def test_fwht_rejects_non_power_of_two() -> None:
    import pytest

    x = torch.zeros((3, 6), dtype=torch.float32)
    with pytest.raises(ValueError, match="power of two"):
        fast_walsh_hadamard_transform(x)


def test_fast_hadamard_rotation_produces_unit_sign_vectors() -> None:
    d1, d2, padded_dim = fast_hadamard_rotation(
        dim=13, seed=42, device=torch.device("cpu"), dtype=torch.float32
    )
    assert padded_dim == 16, f"expected 16, got {padded_dim}"
    assert d1.shape == (16,) and d2.shape == (16,)
    assert torch.all((d1 == 1.0) | (d1 == -1.0))
    assert torch.all((d2 == 1.0) | (d2 == -1.0))


def test_apply_fast_rotation_preserves_norm() -> None:
    """D1*H*D2 rotation is norm-preserving when dim is a power of two (no truncation)."""
    dim = 16  # power of two: no padding or truncation occurs
    d1, d2, _ = fast_hadamard_rotation(dim=dim, seed=7, device=torch.device("cpu"), dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    x = torch.randn((8, dim), generator=generator)
    rotated = apply_fast_rotation(x, d1, d2)
    assert rotated.shape == x.shape
    orig_norms = torch.linalg.vector_norm(x, dim=-1)
    rot_norms = torch.linalg.vector_norm(rotated, dim=-1)
    assert torch.allclose(orig_norms, rot_norms, atol=1e-4), (
        f"norm mismatch max: {(orig_norms - rot_norms).abs().max()}"
    )


def test_apply_fast_rotation_different_from_identity() -> None:
    """Sanity: rotated output is not the same as input (non-trivial)."""
    dim = 16
    d1, d2, _ = fast_hadamard_rotation(dim=dim, seed=99, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.ones((1, dim), dtype=torch.float32)
    rotated = apply_fast_rotation(x, d1, d2)
    assert not torch.allclose(x, rotated)


def test_rotation_from_policy_fast_hadamard_is_orthogonal() -> None:
    """rotation_from_policy('fast_hadamard') returns an orthogonal matrix."""
    dim = 16
    rotation = rotation_from_policy(
        dim=dim, seed=3, policy="fast_hadamard",
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert rotation.shape == (dim, dim)
    ident = rotation.transpose(0, 1) @ rotation
    assert torch.allclose(ident, torch.eye(dim), atol=1e-5)


def test_rotation_from_policy_fast_hadamard_rejects_non_power_of_two_dim() -> None:
    """fast_hadamard policy raises for non-power-of-two dim (pad-truncate breaks orthogonality)."""
    import pytest

    with pytest.raises(ValueError, match="power of two"):
        rotation_from_policy(
            dim=13, seed=5, policy="fast_hadamard",
            device=torch.device("cpu"), dtype=torch.float32,
        )
