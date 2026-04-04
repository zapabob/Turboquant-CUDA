"""Optional numerical parity against upstream ``multiscreen-pytorch`` (ScreeningUnit path).

Install with: ``uv sync --extra multiscreen_parity`` (or equivalent pip git install).
Default CI omits this extra; the module is skipped when ``multiscreen`` is absent.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from turboquant.research_extension.multiscreen_kv import compute_k_relevance

if importlib.util.find_spec("multiscreen") is None:
    pytest.skip("multiscreen-pytorch not installed (optional extra multiscreen_parity)", allow_module_level=True)

from multiscreen.layers import ScreeningUnit, trim_and_square  # noqa: E402


def test_compute_k_relevance_matches_screening_unit_alpha_max() -> None:
    """Match upstream similarity + trim (no causal softmask — our KV relevance port)."""

    head_dim = 16
    unit = ScreeningUnit(head_dim)
    b, h, q_len, k_len = 1, 2, 3, 5
    q = torch.randn(b, h, q_len, head_dim)
    k = torch.randn(b, h, k_len, head_dim)
    sim = unit.compute_similarity(q, k)
    alpha = trim_and_square(sim, unit.s_r.to(q.dtype))
    expected = alpha.max(dim=-2).values

    got = compute_k_relevance(
        q,
        k,
        s_e=unit.s_e.to(q.dtype),
        s_f=unit.s_f.to(q.dtype),
        s_r=unit.s_r.to(q.dtype),
    )
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-5)
