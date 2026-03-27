"""Output-preserving value codec with sensitivity-based protection."""

from __future__ import annotations

import math

import torch

from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.types import ProtectedValueBatch, TurboQuantMSEConfig, ValueCodecConfig


class ProtectedValueCodec:
    """Exact/high/low mixed value codec with optional low-rank residual correction.

    Protection operates on contiguous channel groups so protected and unprotected
    coordinates are not mixed under the block-SO(8) rotation used by the current
    value-side research branch.
    """

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
        self.last_channel_scores: torch.Tensor | None = None
        self.last_group_scores: torch.Tensor | None = None

    def _require_matching(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.device != self.device or tensor.dtype != self.dtype:
            raise ValueError(f"{name} must match codec device/dtype")

    def _group_size(self) -> int:
        size = max(1, int(self.config.channel_group_size))
        if self.dim % size != 0:
            raise ValueError(f"dim={self.dim} must be divisible by channel_group_size={size}")
        return size

    def _reshape_scores(self, score: torch.Tensor) -> torch.Tensor:
        if score.dim() == 1:
            return score.unsqueeze(0)
        if score.dim() == 2:
            return score
        head_dim = score.shape[-2]
        return score.reshape(-1, head_dim, self.dim).mean(dim=0)

    def _group_view(self, channel_scores: torch.Tensor) -> torch.Tensor:
        group_size = self._group_size()
        return channel_scores.reshape(*channel_scores.shape[:-1], self.dim // group_size, group_size)

    def _expand_group_mask(self, group_mask: torch.Tensor) -> torch.Tensor:
        group_size = self._group_size()
        return group_mask.repeat_interleave(group_size, dim=-1)

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

    def _attention_output_energy(self, values: torch.Tensor, attention_weights: torch.Tensor | None) -> torch.Tensor:
        if attention_weights is None:
            return values.square().mean(dim=-2)
        self._require_matching(attention_weights, "attention_weights")
        return torch.einsum("...qs,...sd->...qd", attention_weights.square(), values.square()).mean(dim=-2)

    def _gradient_proxy(self, values: torch.Tensor, attention_weights: torch.Tensor | None) -> torch.Tensor:
        if attention_weights is None:
            return values.square().mean(dim=-2)
        self._require_matching(attention_weights, "attention_weights")
        output = torch.einsum("...qs,...sd->...qd", attention_weights, values)
        gradient = torch.einsum("...qs,...qd->...sd", attention_weights, output)
        return gradient.square().mean(dim=-2)

    def calibrate(self, values: torch.Tensor, attention_weights: torch.Tensor | None = None) -> None:
        self._require_matching(values, "values")
        if self.config.sensitivity.score_source == "teacher-gradient-proxy":
            channel_score = self._gradient_proxy(values, attention_weights)
        else:
            channel_score = self._attention_output_energy(values, attention_weights)

        channel_score = self._reshape_scores(channel_score).to(device=self.device, dtype=self.dtype)
        group_score = self._group_view(channel_score).mean(dim=-1)
        self.last_channel_scores = channel_score
        self.last_group_scores = group_score

        num_groups = group_score.shape[-1]
        exact_k = min(
            num_groups,
            int(math.ceil(self.config.protected_fraction * num_groups)) if self.config.protected_fraction > 0 else 0,
        )
        secondary_k = min(
            num_groups - exact_k,
            int(math.ceil(self.config.secondary_fraction * num_groups)) if self.config.secondary_fraction > 0 else 0,
        )

        exact_group_mask = torch.zeros_like(group_score, dtype=torch.bool)
        high_group_mask = torch.zeros_like(group_score, dtype=torch.bool)

        if exact_k > 0:
            exact_idx = group_score.topk(exact_k, dim=-1).indices
            exact_group_mask.scatter_(-1, exact_idx, True)
        if secondary_k > 0:
            masked_group_score = group_score.masked_fill(exact_group_mask, float("-inf"))
            high_idx = masked_group_score.topk(secondary_k, dim=-1).indices
            high_group_mask.scatter_(-1, high_idx, True)

        self.exact_channel_mask = self._expand_group_mask(exact_group_mask).unsqueeze(0).to(device=self.device)
        self.high_precision_mask = self._expand_group_mask(high_group_mask).unsqueeze(0).to(device=self.device)

    def channel_sensitivity(self) -> torch.Tensor | None:
        return self.last_channel_scores

    def group_sensitivity(self) -> torch.Tensor | None:
        return self.last_group_scores

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
        self._require_matching(values, "values")
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
