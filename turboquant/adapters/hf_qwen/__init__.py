"""Optional Hugging Face / Qwen adapter exports."""

from turboquant.capture import DEFAULT_PROMPT_PANEL
from turboquant.hf_cache import CacheBackend, ExactCacheBackend, TurboQuantCacheBackend
from turboquant.runtime import BASE_MODEL_ID, DEFAULT_MODEL_ID, LOCAL_CAPTURE_MODEL_PATH
from turboquant.adapters.hf_qwen.online_eval import (
    QwenOnlineEvalConfig,
    SUPPORTED_QWEN_ONLINE_MODES,
    TurboQuantQwenCache,
    load_qwen_online_eval_model,
    patch_qwen_for_online_eval,
)

__all__ = [
    "BASE_MODEL_ID",
    "CacheBackend",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROMPT_PANEL",
    "ExactCacheBackend",
    "LOCAL_CAPTURE_MODEL_PATH",
    "QwenOnlineEvalConfig",
    "SUPPORTED_QWEN_ONLINE_MODES",
    "TurboQuantCacheBackend",
    "TurboQuantQwenCache",
    "load_qwen_online_eval_model",
    "patch_qwen_for_online_eval",
]
