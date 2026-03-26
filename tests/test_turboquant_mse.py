from __future__ import annotations

import torch

from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.types import TurboQuantMSEConfig


def test_turboquant_mse_reduces_error_with_more_bits() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    x = torch.randn((256, 64), generator=generator)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    errors = []
    for bits in (2, 3, 4):
        quantizer = TurboQuantMSE(TurboQuantMSEConfig(dim=64, bits=bits))
        reconstructed = quantizer.dequantize(quantizer.quantize(x))
        errors.append(torch.mean((x - reconstructed) ** 2).item())

    assert errors[0] > errors[1] > errors[2]
