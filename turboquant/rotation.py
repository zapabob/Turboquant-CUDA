"""Rotation helpers used by TurboQuant."""

from __future__ import annotations

from functools import lru_cache

import torch

from turboquant.types import RotationPolicy


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype name: {name}") from exc


@lru_cache(maxsize=64)
def _rotation_cpu(dim: int, seed: int, dtype_name: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn((dim, dim), generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    rotation = q * signs
    return rotation.to(dtype=resolve_dtype(dtype_name))


def random_rotation(dim: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached Haar-like orthogonal rotation matrix."""

    rotation = _rotation_cpu(dim=dim, seed=seed, dtype_name=str(dtype).split(".")[-1])
    return rotation.to(device=device, dtype=dtype)


def _block_rotation_cpu(dim: int, seed: int, block_size: int, dtype_name: str) -> torch.Tensor:
    if dim % block_size != 0:
        raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
    blocks = []
    for block_idx in range(dim // block_size):
        blocks.append(_rotation_cpu(block_size, seed + block_idx, dtype_name))
    return torch.block_diag(*blocks)


def block_so8_rotation(dim: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rotation = _block_rotation_cpu(dim=dim, seed=seed, block_size=8, dtype_name=str(dtype).split(".")[-1])
    return rotation.to(device=device, dtype=dtype)


def block_so8_from_skew(skew_blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert `[num_blocks, 8, 8]` skew generators into a block-diagonal SO(8) matrix."""

    blocks: list[torch.Tensor] = []
    for block in skew_blocks:
        skew = block - block.transpose(0, 1)
        blocks.append(torch.matrix_exp(skew).to(dtype=dtype))
    return torch.block_diag(*blocks)


def so8_block_diagonal_rotation_metrics(
    rotation: torch.Tensor,
    *,
    block_size: int = 8,
) -> tuple[float, float]:
    """Aggregate orthogonality and SO(block_size) determinant drift for a block-diagonal rotation.

    Expects a square matrix whose diagonal ``block_size``×``block_size`` blocks are the SO factors
    (as produced by :func:`block_so8_from_skew` / :func:`block_so8_rotation`).

    Returns:
        orthogonality_error: max absolute entry of ``R^T R - I`` (float64 accumulation).
        determinant_error_max: ``max_i |det(B_i) - 1|`` over diagonal blocks ``B_i``.
    """

    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"Expected square 2D rotation, got {tuple(rotation.shape)}")
    dim = int(rotation.shape[0])
    if dim % block_size != 0:
        raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
    r64 = rotation.to(dtype=torch.float64)
    ident = torch.eye(dim, dtype=torch.float64, device=r64.device)
    ortho_err = float((r64.transpose(0, 1) @ r64 - ident).abs().max().item())
    n_blocks = dim // block_size
    det_err_max = 0.0
    for block_idx in range(n_blocks):
        sl = slice(block_idx * block_size, (block_idx + 1) * block_size)
        block = r64[sl, sl]
        det_err_max = max(det_err_max, abs(float(torch.linalg.det(block).item()) - 1.0))
    return ortho_err, det_err_max


def rotation_from_policy(
    *,
    dim: int,
    seed: int,
    policy: RotationPolicy,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if policy == "random_haar":
        return random_rotation(dim=dim, seed=seed, device=device, dtype=dtype)
    if policy in {"block_so8_static", "block_so8_learned"}:
        return block_so8_rotation(dim=dim, seed=seed, device=device, dtype=dtype)
    raise ValueError(f"Unsupported rotation policy: {policy!r}")
