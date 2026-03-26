"""Stage 1 TurboQuant implementation."""

from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.lloyd_max import codebook_tensor
from turboquant.rotation import random_rotation, resolve_dtype
from turboquant.types import QuantizedMSEBatch, TurboQuantMSEConfig


class TurboQuantMSE:
    """Random rotation + per-coordinate scalar quantization."""

    def __init__(self, config: TurboQuantMSEConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = resolve_dtype(config.dtype)
        self.rotation = random_rotation(
            dim=config.dim,
            seed=config.rotation_seed,
            device=self.device,
            dtype=self.dtype,
        )
        self._codebooks: dict[int, torch.Tensor] = {}

    def fit_codebook(self, dim: int | None = None, bits: int | None = None) -> torch.Tensor:
        target_dim = self.config.dim if dim is None else dim
        target_bits = self.config.bits if bits is None else bits
        if target_dim != self.config.dim:
            raise ValueError("This quantizer is fixed to its configured dimension")
        if target_bits not in self._codebooks:
            self._codebooks[target_bits] = codebook_tensor(
                dim=target_dim,
                bits=target_bits,
                device=self.device,
                dtype=self.dtype,
            )
        return self._codebooks[target_bits]

    def _encode_with_bits(self, values: torch.Tensor, bits: int) -> torch.Tensor:
        codebook = self.fit_codebook(bits=bits)
        distances = (values.unsqueeze(-1) - codebook).abs()
        return distances.argmin(dim=-1).to(torch.uint8)

    def _decode_with_bits(self, indices: torch.Tensor, bits: int) -> torch.Tensor:
        codebook = self.fit_codebook(bits=bits)
        return codebook[indices.long()]

    def quantize(
        self,
        x: torch.Tensor,
        allocation: ChannelBitAllocation | None = None,
    ) -> QuantizedMSEBatch:
        if x.shape[-1] != self.config.dim:
            raise ValueError(f"Expected last dimension {self.config.dim}, got {x.shape[-1]}")
        x = x.to(device=self.device, dtype=self.dtype)
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        safe_norms = torch.clamp(norms, min=torch.finfo(x.dtype).eps)
        unit = torch.where(norms > 0, x / safe_norms, torch.zeros_like(x))
        rotated = torch.matmul(unit, self.rotation.transpose(0, 1))
        bitwidths = (
            allocation.make_bitwidths(rotated)
            if allocation is not None
            else torch.full_like(rotated, fill_value=self.config.bits, dtype=torch.uint8)
        )
        indices = torch.zeros_like(rotated, dtype=torch.uint8)
        for bits in sorted({int(v) for v in bitwidths.unique().tolist()}):
            mask = bitwidths == bits
            if mask.any():
                indices[mask] = self._encode_with_bits(rotated[mask], bits=bits)
        return QuantizedMSEBatch(
            norms=norms,
            indices=indices,
            bitwidths=bitwidths,
            shape=tuple(x.shape),
            dim=self.config.dim,
        )

    def dequantize(self, encoded: QuantizedMSEBatch) -> torch.Tensor:
        values = torch.zeros(encoded.shape, device=self.device, dtype=self.dtype)
        for bits in sorted({int(v) for v in encoded.bitwidths.unique().tolist()}):
            mask = encoded.bitwidths == bits
            if mask.any():
                values[mask] = self._decode_with_bits(encoded.indices[mask], bits=bits)
        reconstructed = torch.matmul(values, self.rotation)
        return reconstructed * encoded.norms
