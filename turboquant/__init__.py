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
from turboquant.schema import (
    PAPER_SCHEMA_KIND,
    RESEARCH_SCHEMA_KIND,
    build_paper_turboquant_config,
    build_research_turboquant_config,
    read_turboquant_config,
    validate_paper_turboquant_config,
    validate_research_turboquant_config,
    write_turboquant_config,
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
    "PAPER_SCHEMA_KIND",
    "PaperMSEConfig",
    "PaperMixedBitPolicy",
    "PaperProdConfig",
    "PaperTurboQuantMSE",
    "PaperTurboQuantProd",
    "ProtectedValueCodec",
    "RESEARCH_SCHEMA_KIND",
    "V_ABLATION_MODES",
    "ValueResearchConfig",
    "build_paper_turboquant_config",
    "build_research_turboquant_config",
    "evaluate_paper_attention_grid",
    "read_turboquant_config",
    "summarize_layer_thresholds",
    "summarize_metric_trials",
    "summarize_trial_metrics",
    "TurboQuantMSE",
    "TurboQuantMSEConfig",
    "TurboQuantProd",
    "TurboQuantProdConfig",
    "validate_paper_turboquant_config",
    "validate_research_turboquant_config",
    "write_turboquant_config",
]
