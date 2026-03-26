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
