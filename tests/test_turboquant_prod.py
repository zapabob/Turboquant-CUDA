from __future__ import annotations

import torch

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
