from __future__ import annotations

import torch

from turboquant.qjl import GaussianSignSketch


def test_qjl_estimator_is_approximately_unbiased() -> None:
    dim = 32
    sketch = GaussianSignSketch(dim=dim, sketch_dim=1024, seed=1, device="cpu", dtype="float32")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    residual = torch.randn((4096, dim), generator=generator)
    y = torch.randn((4096, dim), generator=generator)
    encoded = sketch.encode(residual)
    estimate = sketch.estimate(y, encoded)
    exact = (residual * y).sum(dim=-1)
    assert abs((estimate - exact).mean().item()) < 0.1


def test_qjl_default_dim_is_paper_faithful() -> None:
    from turboquant.types import TurboQuantProdConfig

    cfg = TurboQuantProdConfig(dim=64, total_bits=4)
    assert cfg.resolved_qjl_dim() == 64


def test_qjl_decode_preserves_shape() -> None:
    dim = 16
    sketch = GaussianSignSketch(dim=dim, sketch_dim=64, seed=7, device="cpu", dtype="float32")
    residual = torch.randn((3, dim), dtype=torch.float32)
    encoded = sketch.encode(residual)
    decoded = sketch.decode(encoded)
    assert decoded.shape == residual.shape
