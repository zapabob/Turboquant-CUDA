from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.research_extension import (
    TRIALITY_PROXY_VIEWS,
    TrialityProxyMSE,
    TrialityProxyProd,
    apply_triality_proxy_view,
    get_triality_proxy_adapters,
    invert_triality_proxy_view,
)
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantMSEConfig, TurboQuantProdConfig


def test_triality_proxy_adapters_are_orthogonal_with_positive_determinant() -> None:
    adapters = get_triality_proxy_adapters(dtype=torch.float32)
    for view in TRIALITY_PROXY_VIEWS:
        adapter = adapters[view]
        ident = adapter.transpose(0, 1) @ adapter
        det = torch.det(adapter)
        assert torch.allclose(ident, torch.eye(8), atol=1e-5, rtol=1e-5)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_triality_proxy_view_roundtrip_restores_tensor() -> None:
    x = torch.randn((2, 3, 16), dtype=torch.float32)
    for view in TRIALITY_PROXY_VIEWS:
        transformed = apply_triality_proxy_view(x, view)
        restored = invert_triality_proxy_view(transformed, view)
        assert torch.allclose(restored, x, atol=1e-5, rtol=1e-5)


def test_triality_proxy_mse_roundtrip_shape_matches() -> None:
    quantizer = TurboQuantMSE(TurboQuantMSEConfig(dim=16, bits=2, device="cpu", dtype="float32"))
    proxy = TrialityProxyMSE(quantizer=quantizer, view="spinor_plus_proxy")
    x = torch.randn((4, 16), dtype=torch.float32)
    encoded = proxy.quantize(x)
    decoded = proxy.dequantize(encoded)
    assert decoded.shape == x.shape


def test_triality_proxy_prod_quantize_and_score_shape_matches() -> None:
    quantizer = TurboQuantProd(TurboQuantProdConfig(dim=16, total_bits=2, device="cpu", dtype="float32"))
    proxy = TrialityProxyProd(quantizer=quantizer, view="spinor_minus_proxy")
    x = torch.randn((1, 2, 8, 16), dtype=torch.float32)
    q = torch.randn((1, 2, 4, 16), dtype=torch.float32)
    encoded = proxy.quantize(x, allocation=ChannelBitAllocation.preset(effective_bits=1.5, width=16))
    logits = proxy.pairwise_estimate(q, encoded)
    decoded = proxy.transport_decode(encoded)
    assert logits.shape == (1, 2, 4, 8)
    assert decoded.shape == x.shape
