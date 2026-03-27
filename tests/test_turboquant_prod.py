from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantProdConfig


def test_turboquant_prod_matches_vector_shape() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    x = torch.randn((32, 64), generator=generator)
    y = torch.randn((32, 64), generator=generator)
    quantizer = TurboQuantProd(TurboQuantProdConfig(dim=64, total_bits=3))
    encoded = quantizer.quantize(x)
    estimate = quantizer.estimate_inner_product(y, encoded)
    assert estimate.shape == (32,)


def test_turboquant_prod_supports_mixed_bit_allocation() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    x = torch.randn((16, 128), generator=generator, dtype=torch.float32)
    quantizer = TurboQuantProd(TurboQuantProdConfig(dim=128, total_bits=3, device="cpu", dtype="float32"))
    allocation = ChannelBitAllocation.preset(effective_bits=3.5, width=128)
    encoded = quantizer.quantize(x, allocation=allocation)
    assert encoded.mse.bitwidths.shape == x.shape
