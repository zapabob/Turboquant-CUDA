from __future__ import annotations

import torch

from turboquant.rotation import random_rotation


def test_random_rotation_is_orthogonal() -> None:
    rotation = random_rotation(dim=16, seed=7, device=torch.device("cpu"), dtype=torch.float32)
    ident = rotation.transpose(0, 1) @ rotation
    assert torch.allclose(ident, torch.eye(16), atol=1e-5, rtol=1e-5)
