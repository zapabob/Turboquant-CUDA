"""PyTorch-first paper-faithful TurboQuant baseline."""

from turboquant.paper_baseline.attention import PAPER_BASELINE_MODES, evaluate_paper_attention_grid
from turboquant.paper_baseline.mse import PaperTurboQuantMSE
from turboquant.paper_baseline.prod import PaperTurboQuantProd
from turboquant.paper_baseline.types import PaperMSEConfig, PaperMixedBitPolicy, PaperProdConfig

__all__ = [
    "PAPER_BASELINE_MODES",
    "PaperMSEConfig",
    "PaperMixedBitPolicy",
    "PaperProdConfig",
    "PaperTurboQuantMSE",
    "PaperTurboQuantProd",
    "evaluate_paper_attention_grid",
]
