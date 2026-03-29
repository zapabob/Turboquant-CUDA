"""Stage 1 TurboQuant implementation."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from turboquant.allocation import ChannelBitAllocation
from turboquant.lloyd_max import codebook_tensor, decision_boundaries_tensor
from turboquant.rotation import block_so8_from_skew, resolve_dtype, rotation_from_policy
from turboquant.types import QuantizedMSEBatch, TurboQuantMSEConfig


class TurboQuantMSE:
    """Rotation + per-coordinate scalar quantization."""

    def __init__(self, config: TurboQuantMSEConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = resolve_dtype(config.dtype)
        if config.codebook_kind != "sphere-lloyd-max":
            raise ValueError(f"Unsupported codebook_kind={config.codebook_kind!r}")
        self.rotation = rotation_from_policy(
            dim=config.dim,
            seed=config.rotation_seed,
            policy=config.rotation_policy,
            device=self.device,
            dtype=self.dtype,
        )
        self._codebooks: dict[int, torch.Tensor] = {}
        self._boundaries: dict[int, torch.Tensor] = {}

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

    def decision_boundaries(self, bits: int | None = None) -> torch.Tensor:
        target_bits = self.config.bits if bits is None else bits
        if target_bits not in self._boundaries:
            self._boundaries[target_bits] = decision_boundaries_tensor(
                dim=self.config.dim,
                bits=target_bits,
                device=self.device,
                dtype=self.dtype,
            )
        return self._boundaries[target_bits]

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.config.dim:
            raise ValueError(f"Expected last dimension {self.config.dim}, got {x.shape[-1]}")
        if x.device != self.device:
            raise ValueError(f"Expected tensor on {self.device}, got {x.device}")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected tensor dtype {self.dtype}, got {x.dtype}")

    def _encode_with_bits(self, values: torch.Tensor, bits: int) -> torch.Tensor:
        boundaries = self.decision_boundaries(bits=bits)
        bucket_ids = torch.bucketize(values, boundaries[1:-1], right=False)
        return bucket_ids.to(torch.uint8)

    def _ste_quantize_with_bits(self, values: torch.Tensor, bits: int) -> torch.Tensor:
        indices = self._encode_with_bits(values, bits=bits)
        decoded = self._decode_with_bits(indices, bits=bits)
        return values + (decoded - values).detach()

    def _decode_with_bits(self, indices: torch.Tensor, bits: int) -> torch.Tensor:
        codebook = self.fit_codebook(bits=bits)
        return codebook[indices.long()]

    def quantize(
        self,
        x: torch.Tensor,
        allocation: ChannelBitAllocation | None = None,
    ) -> QuantizedMSEBatch:
        self._validate_input(x)
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        safe_norms = torch.clamp(norms, min=torch.finfo(x.dtype).eps)
        unit = torch.where(norms > 0, x / safe_norms, torch.zeros_like(x))
        rotated = torch.matmul(unit, self.rotation.transpose(0, 1))
        bitwidths = (
            allocation.make_bitwidths(rotated)
            if allocation is not None
            else torch.full_like(rotated, fill_value=self.config.bits, dtype=torch.uint8)
        )
        return self.quantize_with_bitwidths(x=x, bitwidths=bitwidths)

    def quantize_with_bitwidths(self, x: torch.Tensor, bitwidths: torch.Tensor) -> QuantizedMSEBatch:
        self._validate_input(x)
        if bitwidths.shape != x.shape:
            raise ValueError(f"Expected bitwidths shape {tuple(x.shape)}, got {tuple(bitwidths.shape)}")
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        safe_norms = torch.clamp(norms, min=torch.finfo(x.dtype).eps)
        unit = torch.where(norms > 0, x / safe_norms, torch.zeros_like(x))
        rotated = torch.matmul(unit, self.rotation.transpose(0, 1))
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

    def set_rotation(self, rotation: torch.Tensor) -> None:
        if rotation.shape != (self.config.dim, self.config.dim):
            raise ValueError(
                f"Expected rotation shape {(self.config.dim, self.config.dim)}, got {tuple(rotation.shape)}"
            )
        self.rotation = rotation.to(device=self.device, dtype=self.dtype)

    def fit_rotation(
        self,
        x: torch.Tensor,
        *,
        queries: torch.Tensor | None = None,
        steps: int = 60,
        lr: float = 5e-2,
        rank_weight: float = 0.1,
        step_metrics_callback: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Fit a block-SO(8) rotation using a quantization-aware STE proxy.

        If ``step_metrics_callback`` is set, it is invoked after each optimizer step with
        ``(step_index, rotation_tensor)`` where ``rotation_tensor`` is the current block-diagonal SO(8) matrix.
        """

        if self.config.rotation_policy != "block_so8_learned":
            return self.rotation
        self._validate_input(x)
        if self.config.dim % 8 != 0:
            raise ValueError("block_so8_learned requires dim divisible by 8")
        num_blocks = self.config.dim // 8

        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        safe_norms = torch.clamp(norms, min=torch.finfo(x.dtype).eps)
        unit = torch.where(norms > 0, x / safe_norms, torch.zeros_like(x))
        exact_scores = None
        if queries is not None:
            if queries.device != self.device or queries.dtype != self.dtype:
                raise ValueError("queries must match the quantizer device/dtype")
            exact_scores = torch.einsum("...qd,...sd->...qs", queries, x)

        skew_blocks = torch.zeros((num_blocks, 8, 8), device=self.device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([skew_blocks], lr=lr)

        for step_idx in range(steps):
            optimizer.zero_grad(set_to_none=True)
            rotation = block_so8_from_skew(skew_blocks, dtype=self.dtype).to(device=self.device, dtype=self.dtype)
            rotated = torch.matmul(unit, rotation.transpose(0, 1))
            quantized = self._ste_quantize_with_bits(rotated, bits=self.config.bits)
            reconstructed = torch.matmul(quantized, rotation) * norms
            loss = F.mse_loss(reconstructed, x)
            if exact_scores is not None:
                approx_scores = torch.einsum("...qd,...sd->...qs", queries, reconstructed)
                flat_exact = exact_scores.reshape(-1)
                flat_approx = approx_scores.reshape(-1)
                loss = loss + (1.0 - F.cosine_similarity(flat_exact, flat_approx, dim=0))
                if rank_weight > 0:
                    exact_probs = torch.softmax(exact_scores / math.sqrt(float(self.config.dim)), dim=-1)
                    approx_probs = torch.softmax(approx_scores / math.sqrt(float(self.config.dim)), dim=-1)
                    loss = loss + (rank_weight * F.kl_div(approx_probs.log(), exact_probs, reduction="batchmean"))
            loss.backward()
            optimizer.step()
            if step_metrics_callback is not None:
                with torch.no_grad():
                    current = block_so8_from_skew(skew_blocks.detach(), dtype=self.dtype).to(
                        device=self.device, dtype=self.dtype
                    )
                step_metrics_callback(step_idx, current)

        with torch.no_grad():
            fitted = block_so8_from_skew(skew_blocks.detach(), dtype=self.dtype).to(device=self.device, dtype=self.dtype)
        self.rotation = fitted
        return self.rotation
