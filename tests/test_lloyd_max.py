from __future__ import annotations

from turboquant.lloyd_max import fit_lloyd_max_codebook


def test_lloyd_max_codebook_is_sorted_and_symmetric() -> None:
    codebook = fit_lloyd_max_codebook(dim=128, bits=3)
    assert len(codebook) == 8
    assert list(codebook) == sorted(codebook)
    for left, right in zip(codebook, reversed(codebook), strict=True):
        assert abs(left + right) < 1e-5
