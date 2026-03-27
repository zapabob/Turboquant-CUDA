"""K/V-separated research extensions built on top of the paper baseline."""

from turboquant.research_extension.evaluation import (
    V_ABLATION_MODES,
    captured_v_ablation_rows,
    compute_value_sensitivity_rows,
    evaluate_value_protection_grid,
    filter_v_ablation_rows,
    synthetic_v_ablation_rows,
)
from turboquant.research_extension.k_triality import (
    TRIALITY_MODE_BY_VIEW,
    compute_triality_statistics,
    evaluate_triality_proxy_captured,
    fit_triality_proxy_rotations,
    load_triality_proxy_rotations,
    save_triality_proxy_rotations,
    triality_mode_name,
)
from turboquant.research_extension.triality_proxy import (
    TRIALITY_PROXY_VIEWS,
    TrialityProxyMSE,
    TrialityProxyProd,
    apply_blockwise_view,
    apply_triality_proxy_view,
    get_triality_proxy_adapters,
    invert_triality_proxy_view,
    triality_proxy_adapter,
)
from turboquant.research_extension.types import KeyResearchConfig, ValueResearchConfig
from turboquant.value_codec import ProtectedValueCodec

__all__ = [
    "KeyResearchConfig",
    "ProtectedValueCodec",
    "TRIALITY_MODE_BY_VIEW",
    "TRIALITY_PROXY_VIEWS",
    "TrialityProxyMSE",
    "TrialityProxyProd",
    "V_ABLATION_MODES",
    "ValueResearchConfig",
    "apply_blockwise_view",
    "apply_triality_proxy_view",
    "captured_v_ablation_rows",
    "compute_value_sensitivity_rows",
    "compute_triality_statistics",
    "evaluate_value_protection_grid",
    "evaluate_triality_proxy_captured",
    "filter_v_ablation_rows",
    "fit_triality_proxy_rotations",
    "get_triality_proxy_adapters",
    "invert_triality_proxy_view",
    "load_triality_proxy_rotations",
    "save_triality_proxy_rotations",
    "synthetic_v_ablation_rows",
    "triality_mode_name",
    "triality_proxy_adapter",
]
