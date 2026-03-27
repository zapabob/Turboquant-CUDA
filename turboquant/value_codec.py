"""Output-preserving value codec with sensitivity-based protection."""

from __future__ import annotations

import math

import torch

from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.types import ProtectedValueBatch, TurboQuantMSEConfig, ValueCodecConfig


class ProtectedValueCodec:
    """Exact/high/low mixed value codec with optional low-rank residual correction."""

    def __init__(
        self,
        *,
        dim: int,
        config: ValueCodecConfig,
        rotation_seed: int,
        rotation_policy: str,
        device: str,
        dtype: str,
    ) -> None:
        self.dim = dim
        self.config = config
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.base_quantizer = TurboQuantMSE(
            TurboQuantMSEConfig(
                dim=dim,
                bits=config.base_bits,
                rotation_seed=rotation_seed,
                rotation_policy=rotation_policy,
                device=device,
                dtype=dtype,
            )
        )
        self.exact_channel_mask: torch.Tensor | None = None
        self.high_precision_mask: torch.Tensor | None = None
        self.low_rank_basis: torch.Tensor | None = None

    def _prefix_masks(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_shape = (*values.shape[:-2], self.dim)
        exact_mask = self.exact_channel_mask
        high_mask = self.high_precision_mask
        if exact_mask is None or high_mask is None:
            exact_mask = torch.zeros(prefix_shape, device=self.device, dtype=torch.bool)
            high_mask = torch.zeros(prefix_shape, device=self.device, dtype=torch.bool)
            return exact_mask, high_mask
        if exact_mask.shape != prefix_shape:
            exact_mask = exact_mask.expand(prefix_shape)
        if high_mask.shape != prefix_shape:
            high_mask = high_mask.expand(prefix_shape)
        return exact_mask, high_mask

    def calibrate(self, values: torch.Tensor, attention_weights: torch.Tensor | None = None) -> None:
        if values.device != self.device or values.dtype != self.dtype:
            raise ValueError("values must match codec device/dtype")
        if attention_weights is None:
            score = values.square().mean(dim=-2)
        else:
            if attention_weights.device != self.device or attention_weights.dtype != self.dtype:
                raise ValueError("attention_weights must match codec device/dtype")
            # [B,H,Q,S] x [B,H,S,D] -> [B,H,Q,D] -> [B,H,D]
            score = torch.einsum("bhqs,bhsd->bhqd", attention_weights.square(), values.square()).mean(dim=-2)

        mean_score = score.mean(dim=0, keepdim=True)
        exact_k = min(self.dim, int(round(self.config.protected_fraction * self.dim)))
        secondary_k = min(
            self.dim - exact_k,
            int(round(self.config.secondary_fraction * self.dim)),
        )
        exact_mask = torch.zeros_like(mean_score, dtype=torch.bool)
        high_mask = torch.zeros_like(mean_score, dtype=torch.bool)

        if exact_k > 0:
            exact_idx = mean_score.topk(exact_k, dim=-1).indices
            exact_mask.scatter_(-1, exact_idx, True)
        if secondary_k > 0:
            masked_score = mean_score.masked_fill(exact_mask, float("-inf"))
            high_idx = masked_score.topk(secondary_k, dim=-1).indices
            high_mask.scatter_(-1, high_idx, True)

        self.exact_channel_mask = exact_mask.to(device=self.device)
        self.high_precision_mask = high_mask.to(device=self.device)

    def _fit_low_rank_basis(self, residual: torch.Tensor) -> None:
        if self.config.low_rank_rank <= 0:
            self.low_rank_basis = None
            return
        flat = residual.reshape(-1, self.dim)
        if flat.numel() == 0:
            self.low_rank_basis = None
            return
        working = flat.to(dtype=torch.float32)
        _, _, vh = torch.linalg.svd(working, full_matrices=False)
        rank = min(self.config.low_rank_rank, vh.shape[0])
        self.low_rank_basis = vh[:rank].transpose(0, 1).contiguous().to(device=self.device, dtype=self.dtype)

    def encode(self, values: torch.Tensor) -> ProtectedValueBatch:
        if values.device != self.device or values.dtype != self.dtype:
            raise ValueError("values must match codec device/dtype")
        exact_mask, high_mask = self._prefix_masks(values)
        bitwidths = torch.full(values.shape, fill_value=self.config.base_bits, device=self.device, dtype=torch.uint8)
        if self.config.high_bits > self.config.base_bits:
            high_expanded = high_mask.unsqueeze(-2).expand(values.shape)
            bitwidths = torch.where(
                high_expanded,
                torch.full_like(bitwidths, fill_value=self.config.high_bits),
                bitwidths,
            )

        base = self.base_quantizer.quantize_with_bitwidths(values, bitwidths=bitwidths)
        decoded = self.base_quantizer.dequantize(base)
        exact_expanded = exact_mask.unsqueeze(-2).expand(values.shape)
        exact_values = torch.where(exact_expanded, values, torch.zeros_like(values))
        decoded = torch.where(exact_expanded, values, decoded)

        coefficients = None
        if self.config.low_rank_rank > 0:
            residual = values - decoded
            if self.low_rank_basis is None:
                self._fit_low_rank_basis(residual)
            if self.low_rank_basis is not None:
                coefficients = torch.einsum("...d,dr->...r", residual, self.low_rank_basis)

        return ProtectedValueBatch(
            base=base,
            exact_channel_mask=exact_mask,
            exact_values=exact_values,
            high_precision_mask=high_mask,
            low_rank_coefficients=coefficients,
            shape=tuple(values.shape),
            dim=self.dim,
        )

    def decode(self, encoded: ProtectedValueBatch) -> torch.Tensor:
        decoded = self.base_quantizer.dequantize(encoded.base)
        exact_expanded = encoded.exact_channel_mask.unsqueeze(-2).expand(encoded.shape)
        decoded = torch.where(exact_expanded, encoded.exact_values, decoded)
        if encoded.low_rank_coefficients is not None:
            if self.low_rank_basis is None:
                raise RuntimeError("low_rank_basis must be fitted before decoding coefficients")
            decoded = decoded + torch.einsum("...r,dr->...d", encoded.low_rank_coefficients, self.low_rank_basis)
        return decoded

    def storage_bits(self, encoded: ProtectedValueBatch) -> int:
        return encoded.total_bits()

    def memory_ratio_vs_exact(self, encoded: ProtectedValueBatch, exact_values: torch.Tensor) -> float:
        exact_bits = exact_values.numel() * exact_values.element_size() * 8
        return float(self.storage_bits(encoded)) / float(exact_bits)
