from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import evaluate_asymmetric_q8_value_attention_row
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


def test_turboquant_prod_transport_decode_matches_input_shape() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)
    x = torch.randn((4, 32), generator=generator, dtype=torch.float32)
    quantizer = TurboQuantProd(TurboQuantProdConfig(dim=32, total_bits=3, device="cpu", dtype="float32"))
    encoded = quantizer.quantize(x)
    decoded = quantizer.transport_decode(encoded)
    assert decoded.shape == x.shape


def test_head_dim_256_asymmetric_value_modes_do_not_collapse() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    keys = torch.randn((1, 2, 5, 256), generator=generator, dtype=torch.float32)
    values = torch.randn((1, 2, 5, 256), generator=generator, dtype=torch.float32)

    turbo4 = evaluate_asymmetric_q8_value_attention_row(
        dataset="synthetic",
        trial=1,
        layer_idx=2,
        mode="asym_q8_turbo4",
        keys=keys,
        values=values,
    )
    turbo3 = evaluate_asymmetric_q8_value_attention_row(
        dataset="synthetic",
        trial=1,
        layer_idx=2,
        mode="asym_q8_turbo3",
        keys=keys,
        values=values,
    )

    assert float(turbo4["logit_cosine_similarity"]) > 0.0
    assert float(turbo4["hidden_cosine_similarity"]) > 0.0
    assert float(turbo3["logit_cosine_similarity"]) > 0.0
    assert float(turbo3["hidden_cosine_similarity"]) > 0.0
    assert float(turbo4["next_logit_kl"]) >= 0.0
    assert float(turbo3["next_logit_kl"]) >= 0.0
