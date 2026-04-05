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


def test_turboquant_mse_eight_bit_roundtrip_finite_small_dim() -> None:
    """Smoke: 8-bit Lloyd–Max path; use small dim to keep test fast."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    dim = 32
    x = torch.randn((8, dim), generator=generator, dtype=torch.float32)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    quantizer = TurboQuantMSE(TurboQuantMSEConfig(dim=dim, bits=8, device="cpu", dtype="float32"))
    recon = quantizer.dequantize(quantizer.quantize(x))
    assert torch.isfinite(recon).all()
    assert recon.shape == x.shape


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


# ---------------------------------------------------------------------------
# norm_correction tests
# ---------------------------------------------------------------------------


def test_norm_correction_false_is_default() -> None:
    """TurboQuantMSEConfig default norm_correction is False."""
    cfg = TurboQuantMSEConfig(dim=16, bits=4)
    assert cfg.norm_correction is False


def test_norm_correction_changes_reconstruction_at_low_bits() -> None:
    """norm_correction=True and False produce different outputs at 2-bit."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    x = torch.randn((32, 32), generator=generator, dtype=torch.float32)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    cfg_off = TurboQuantMSEConfig(dim=32, bits=2, rotation_seed=0, norm_correction=False)
    cfg_on = TurboQuantMSEConfig(dim=32, bits=2, rotation_seed=0, norm_correction=True)
    q_off = TurboQuantMSE(cfg_off)
    q_on = TurboQuantMSE(cfg_on)
    # Share the same rotation so the only difference is norm_correction
    q_on.set_rotation(q_off.rotation)

    recon_off = q_off.dequantize(q_off.quantize(x))
    recon_on = q_on.dequantize(q_on.quantize(x))

    assert recon_off.shape == x.shape
    assert recon_on.shape == x.shape
    assert torch.isfinite(recon_on).all()
    assert not torch.allclose(recon_off, recon_on), (
        "Expected norm_correction to change reconstruction at 2 bits"
    )


def test_norm_correction_no_nan_on_zero_input() -> None:
    """norm_correction=True must not produce NaN on zero input."""
    cfg = TurboQuantMSEConfig(dim=16, bits=2, norm_correction=True)
    q = TurboQuantMSE(cfg)
    x = torch.zeros((4, 16), dtype=torch.float32)
    encoded = q.quantize(x)
    recon = q.dequantize(encoded)
    assert torch.isfinite(recon).all()
