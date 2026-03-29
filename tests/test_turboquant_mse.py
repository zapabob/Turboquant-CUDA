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


def test_fit_rotation_step_metrics_callback_once_per_step() -> None:
    steps = 4
    seen: list[int] = []

    def callback(step: int, rot: torch.Tensor) -> None:
        assert rot.shape == (8, 8)
        seen.append(step)

    quantizer = TurboQuantMSE(
        TurboQuantMSEConfig(
            dim=8,
            bits=2,
            rotation_policy="block_so8_learned",
            device="cpu",
            dtype="float32",
        )
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    x = torch.randn((8, 8), generator=generator, dtype=torch.float32)
    quantizer.fit_rotation(x, steps=steps, lr=0.05, step_metrics_callback=callback)
    assert seen == list(range(steps))
