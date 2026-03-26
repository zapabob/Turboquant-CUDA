"""Random rotation helpers used by TurboQuant."""

from __future__ import annotations

from functools import lru_cache

import torch


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
