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


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    p = 1
    while p < n:
        p <<= 1
    return p


def fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply the normalised Walsh-Hadamard transform along the last axis.

    The last dimension of ``x`` must be a power of two.  The transform is
    its own inverse up to a 1/sqrt(n) factor; the returned tensor is the
    *normalised* WHT, so ``||Hx|| == ||x||``.

    Args:
        x: Float tensor whose last dimension is a power of two.

    Returns:
        New tensor of the same shape and dtype with WHT applied on last axis.

    Raises:
        ValueError: If the last dimension is not a power of two.
    """
    d = x.shape[-1]
    if d == 0 or (d & (d - 1)) != 0:
        raise ValueError(
            f"fast_walsh_hadamard_transform requires last dim to be a power of two, got {d}"
        )
    out = x.clone()
    h = 1
    while h < d:
        out = out.reshape(*out.shape[:-1], d // (2 * h), 2, h)
        a = out[..., 0, :]  # [..., groups, h]
        b = out[..., 1, :]
        out = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-1], d)
        h <<= 1
    return out * (d ** -0.5)


def fast_hadamard_rotation(
    dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a fast D1*H*D2 randomised Hadamard rotation.

    For non-power-of-two ``dim``, pads to the next power of two internally.
    :func:`apply_fast_rotation` handles padding and truncation at apply time.

    Args:
        dim:    Logical (unpadded) last dimension.
        seed:   Integer RNG seed for the two diagonal sign matrices.
        device: Target device.
        dtype:  Target float dtype.

    Returns:
        Tuple ``(d1, d2, padded_dim)`` where ``d1`` and ``d2`` are 1-D sign
        tensors of length ``padded_dim``.

    Raises:
        ValueError: If dim < 1.
    """
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}")
    padded_dim = _next_power_of_two(dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    d1_int = torch.randint(0, 2, (padded_dim,), generator=generator, dtype=torch.int8)
    d1 = (d1_int * 2 - 1).to(dtype=dtype, device=device)
    generator.manual_seed(seed + 1)
    d2_int = torch.randint(0, 2, (padded_dim,), generator=generator, dtype=torch.int8)
    d2 = (d2_int * 2 - 1).to(dtype=dtype, device=device)
    return d1, d2, padded_dim


@lru_cache(maxsize=64)
def _fast_hadamard_signs_cpu(
    dim: int,
    seed: int,
    dtype_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Cached CPU computation of fast Hadamard sign vectors."""
    return fast_hadamard_rotation(
        dim=dim,
        seed=seed,
        device=torch.device("cpu"),
        dtype=resolve_dtype(dtype_name),
    )


def apply_fast_rotation(
    x: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
) -> torch.Tensor:
    """Apply a pre-built D1*H*D2 fast Hadamard rotation to x along the last dim.

    Pads x to ``len(d1)``, applies D2 → WHT → D1, then truncates back to the
    original last dimension.

    Args:
        x:  Input float tensor; last dim <= len(d1).
        d1: 1-D sign tensor of length padded_dim.
        d2: 1-D sign tensor of length padded_dim.

    Returns:
        Tensor of the same shape and dtype as ``x``.

    Raises:
        ValueError: On shape or dtype/device mismatches.
    """
    if d1.shape != d2.shape or d1.ndim != 1:
        raise ValueError(
            f"d1 and d2 must be 1-D tensors of equal length, got {d1.shape}, {d2.shape}"
        )
    padded_dim = d1.shape[0]
    orig_dim = x.shape[-1]
    if orig_dim > padded_dim:
        raise ValueError(f"x last dim {orig_dim} exceeds padded_dim {padded_dim}")
    if x.dtype != d1.dtype:
        raise ValueError(
            f"x dtype {x.dtype} must match d1 dtype {d1.dtype}"
        )
    if x.device != d1.device:
        raise ValueError(
            f"x device {x.device} must match d1 device {d1.device}"
        )
    if orig_dim < padded_dim:
        x_padded = torch.nn.functional.pad(x, (0, padded_dim - orig_dim))
    else:
        x_padded = x
    rotated = x_padded * d2
    rotated = fast_walsh_hadamard_transform(rotated)
    rotated = rotated * d1
    return rotated[..., :orig_dim]


def _materialise_fast_hadamard(
    dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialise a dense (dim, dim) fast Hadamard rotation matrix.

    ``dim`` must be a power of two.  For non-power-of-two dimensions, pad-and-
    truncate breaks orthogonality; use ``random_haar`` or ``block_so8_*`` instead.

    Applies D1*H*D2 to each standard basis vector to build the full matrix.
    O(d^2) memory — used by :func:`rotation_from_policy` so downstream matmul
    call sites need no changes.  For O(d log d) application use
    :func:`apply_fast_rotation` directly.

    Raises:
        ValueError: If dim is not a power of two.
    """
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(
            f"fast_hadamard rotation_from_policy requires dim to be a power of two, got {dim}. "
            "Use 'random_haar' or 'block_so8_static' for non-power-of-two dims."
        )
    dtype_name = str(dtype).split(".")[-1]
    d1, d2, _ = _fast_hadamard_signs_cpu(dim=dim, seed=seed, dtype_name=dtype_name)
    eye = torch.eye(dim, dtype=resolve_dtype(dtype_name), device=torch.device("cpu"))
    cols = apply_fast_rotation(eye, d1, d2)  # (dim, dim)
    return cols.to(device=device, dtype=dtype)


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
    """Return an orthogonal rotation matrix for the given policy.

    For ``"fast_hadamard"`` the matrix is materialised as a dense (dim, dim)
    tensor (O(d^2) memory / matmul).  The O(d log d) application path is
    available via :func:`apply_fast_rotation` for callers that hold the sign
    vectors explicitly via :func:`fast_hadamard_rotation`.
    """
    if policy == "random_haar":
        return random_rotation(dim=dim, seed=seed, device=device, dtype=dtype)
    if policy in {"block_so8_static", "block_so8_learned"}:
        return block_so8_rotation(dim=dim, seed=seed, device=device, dtype=dtype)
    if policy == "fast_hadamard":
        return _materialise_fast_hadamard(dim=dim, seed=seed, device=device, dtype=dtype)
    raise ValueError(f"Unsupported rotation policy: {policy!r}")
