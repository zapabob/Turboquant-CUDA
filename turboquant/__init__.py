"""TurboQuant PyTorch baseline plus K/V-separated research extensions."""

from turboquant.analysis import summarize_layer_thresholds, summarize_trial_metrics
from turboquant.allocation import ChannelBitAllocation
from turboquant.kv_codec import KVCodec, KVCodecConfig
from turboquant.paper_baseline import (
    PAPER_BASELINE_MODES,
    PaperMSEConfig,
    PaperMixedBitPolicy,
    PaperProdConfig,
    PaperTurboQuantMSE,
    PaperTurboQuantProd,
    evaluate_paper_attention_grid,
)
from turboquant.reporting import summarize_metric_trials
from turboquant.research_extension import (
    KeyResearchConfig,
    ProtectedValueCodec,
    V_ABLATION_MODES,
    ValueResearchConfig,
)
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantMSEConfig, TurboQuantProdConfig

__all__ = [
    "ChannelBitAllocation",
    "KeyResearchConfig",
    "KVCodec",
    "KVCodecConfig",
    "PAPER_BASELINE_MODES",
    "PaperMSEConfig",
    "PaperMixedBitPolicy",
    "PaperProdConfig",
    "PaperTurboQuantMSE",
    "PaperTurboQuantProd",
    "ProtectedValueCodec",
    "V_ABLATION_MODES",
    "ValueResearchConfig",
    "evaluate_paper_attention_grid",
    "summarize_layer_thresholds",
    "summarize_metric_trials",
    "summarize_trial_metrics",
    "TurboQuantMSE",
    "TurboQuantMSEConfig",
    "TurboQuantProd",
    "TurboQuantProdConfig",
]
