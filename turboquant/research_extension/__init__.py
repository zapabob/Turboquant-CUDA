"""K/V-separated research extensions built on top of the paper baseline."""

from turboquant.research_extension.evaluation import (
    V_ABLATION_MODES,
    captured_v_ablation_rows,
    compute_value_sensitivity_rows,
    evaluate_value_protection_grid,
    filter_v_ablation_rows,
    synthetic_v_ablation_rows,
)
from turboquant.research_extension.types import KeyResearchConfig, ValueResearchConfig
from turboquant.value_codec import ProtectedValueCodec

__all__ = [
    "KeyResearchConfig",
    "ProtectedValueCodec",
    "V_ABLATION_MODES",
    "ValueResearchConfig",
    "captured_v_ablation_rows",
    "compute_value_sensitivity_rows",
    "evaluate_value_protection_grid",
    "filter_v_ablation_rows",
    "synthetic_v_ablation_rows",
]
