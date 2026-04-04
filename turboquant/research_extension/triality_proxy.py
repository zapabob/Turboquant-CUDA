"""Research-only SO(8) triality-inspired proxy adapters.

These adapters are explicitly empirical proxies rather than strict Spin(8)
spinor representations. They provide three fixed 8D orthogonal "views" that
can be composed with the learned block-SO(8) key-only path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import QuantizedProdBatch


TRIALITY_PROXY_VIEWS = (
    "vector",
    "spinor_plus_proxy",
    "spinor_minus_proxy",
)


def _vector_adapter(dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(8, dtype=dtype)


def _spinor_plus_adapter(dtype: torch.dtype) -> torch.Tensor:
    adapter = torch.zeros((8, 8), dtype=dtype)
    adapter[0, 0] = 1.0
    for source in range(1, 7):
        adapter[source + 1, source] = 1.0
    adapter[1, 7] = 1.0
    return adapter


def _spinor_minus_adapter(dtype: torch.dtype) -> torch.Tensor:
    adapter = torch.zeros((8, 8), dtype=dtype)
    adapter[0, 0] = 1.0
    adapter[4, 4] = 1.0
    adapter[1, 7] = -1.0
    adapter[7, 1] = 1.0
    adapter[2, 6] = 1.0
    adapter[6, 2] = 1.0
    adapter[3, 5] = 1.0
    adapter[5, 3] = 1.0
    return adapter


def get_triality_proxy_adapters(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    target_device = device if device is not None else torch.device("cpu")
    return {
        "vector": _vector_adapter(dtype).to(device=target_device),
        "spinor_plus_proxy": _spinor_plus_adapter(dtype).to(device=target_device),
        "spinor_minus_proxy": _spinor_minus_adapter(dtype).to(device=target_device),
    }


def triality_proxy_adapter(
    view: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    adapters = get_triality_proxy_adapters(device=device, dtype=dtype)
    try:
        return adapters[view]
    except KeyError as exc:
        raise ValueError(f"Unsupported triality proxy view: {view!r}") from exc


def _ensure_block_dim(x: torch.Tensor) -> None:
    if x.shape[-1] % 8 != 0:
        raise ValueError(f"Expected last dimension divisible by 8, got {x.shape[-1]}")


def apply_blockwise_view(x: torch.Tensor, adapter: torch.Tensor) -> torch.Tensor:
    _ensure_block_dim(x)
    if adapter.shape != (8, 8):
        raise ValueError(f"Expected adapter shape (8, 8), got {tuple(adapter.shape)}")
    if adapter.device != x.device:
        adapter = adapter.to(device=x.device, dtype=x.dtype)
    else:
        adapter = adapter.to(dtype=x.dtype)
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 8, 8)
    transformed = torch.einsum("...bi,ji->...bj", blocks, adapter)
    return transformed.reshape_as(x)


def apply_triality_proxy_view(x: torch.Tensor, view: str) -> torch.Tensor:
    adapter = triality_proxy_adapter(view, device=x.device, dtype=x.dtype)
    return apply_blockwise_view(x, adapter)


def invert_triality_proxy_view(x: torch.Tensor, view: str) -> torch.Tensor:
    adapter = triality_proxy_adapter(view, device=x.device, dtype=x.dtype)
    return apply_blockwise_view(x, adapter.transpose(0, 1))


@dataclass(slots=True)
class TrialityProxyMSE:
    """Wrap a ``TurboQuantMSE`` in a fixed triality proxy view."""

    quantizer: TurboQuantMSE
    view: str

    def quantize(self, x: torch.Tensor, allocation: ChannelBitAllocation | None = None):
        return self.quantizer.quantize(apply_triality_proxy_view(x, self.view), allocation=allocation)

    def dequantize(self, encoded) -> torch.Tensor:
        decoded = self.quantizer.dequantize(encoded)
        return invert_triality_proxy_view(decoded, self.view)

    def fit_rotation(self, x: torch.Tensor, *, queries: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        view_x = apply_triality_proxy_view(x, self.view)
        view_queries = apply_triality_proxy_view(queries, self.view) if queries is not None else None
        return self.quantizer.fit_rotation(view_x, queries=view_queries, **kwargs)

    def set_rotation(self, rotation: torch.Tensor) -> None:
        self.quantizer.set_rotation(rotation)


@dataclass(slots=True)
class TrialityProxyProd:
    """Wrap a ``TurboQuantProd`` in a fixed triality proxy view."""

    quantizer: TurboQuantProd
    view: str

    def quantize(self, x: torch.Tensor, allocation: ChannelBitAllocation | None = None):
        return self.quantizer.quantize(apply_triality_proxy_view(x, self.view), allocation=allocation)

    def quantize_with_bitwidths(self, x: torch.Tensor, bitwidths: torch.Tensor) -> QuantizedProdBatch:
        """Stage-1 per-element bits in proxy space (e.g. Multiscreen relevance on original ``q,k``)."""

        if bitwidths.shape != x.shape:
            raise ValueError(f"Expected bitwidths shape {tuple(x.shape)}, got {tuple(bitwidths.shape)}")
        view_x = apply_triality_proxy_view(x, self.view)
        return self.quantizer.quantize_with_bitwidths(view_x, bitwidths)

    def dequantize(self, encoded) -> torch.Tensor:
        decoded = self.quantizer.dequantize(encoded)
        return invert_triality_proxy_view(decoded, self.view)

    def transport_decode(self, encoded) -> torch.Tensor:
        decoded = self.quantizer.transport_decode(encoded)
        return invert_triality_proxy_view(decoded, self.view)

    def estimate_inner_product(self, y: torch.Tensor, encoded) -> torch.Tensor:
        return self.quantizer.estimate_inner_product(apply_triality_proxy_view(y, self.view), encoded)

    def pairwise_estimate(self, q: torch.Tensor, encoded) -> torch.Tensor:
        q_view = apply_triality_proxy_view(q, self.view)
        return self.quantizer.qjl.pairwise_estimate(q=q_view, sketch=encoded.qjl) + torch.einsum(
            "...qd,...sd->...qs",
            q_view,
            self.quantizer.dequantize(encoded),
        )

    def fit_rotation(self, x: torch.Tensor, *, queries: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        view_x = apply_triality_proxy_view(x, self.view)
        view_queries = apply_triality_proxy_view(queries, self.view) if queries is not None else None
        return self.quantizer.fit_rotation(view_x, queries=view_queries, **kwargs)

    def set_rotation(self, rotation: torch.Tensor) -> None:
        self.quantizer.mse_quantizer.set_rotation(rotation)
