from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM  # type: ignore[assignment]

from turboquant.adapters.hf_qwen.online_eval import (
    QwenOnlineEvalConfig,
    TurboQuantQwenCache,
    load_qwen_online_eval_model,
    patch_qwen_for_online_eval,
)
from turboquant.hf_cache import ExactCacheBackend, TurboQuantCacheBackend


def _tiny_qwen_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def test_patch_qwen_for_online_eval_runs_key_only_prefill_and_decode() -> None:
    model = _tiny_qwen_model()
    config = QwenOnlineEvalConfig(mode="key_only_random", bits=3.0, device="cpu", torch_dtype="float32")
    patch_qwen_for_online_eval(model, config)

    cache = model.make_turboquant_cache()
    assert isinstance(cache, TurboQuantQwenCache)
    assert isinstance(cache.get_backend(0), TurboQuantCacheBackend)
    assert cache.get_backend(0).quantize_values is False

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6), generator=generator)

    prefill = model(input_ids=input_ids[:, :4], use_cache=True, past_key_values=cache)
    decode = model(input_ids=input_ids[:, 4:5], use_cache=True, past_key_values=cache)

    assert prefill.logits.shape == (1, 4, model.config.vocab_size)
    assert decode.logits.shape == (1, 1, model.config.vocab_size)
    assert cache.get_seq_length(0) == 5


def test_patch_qwen_for_online_eval_exposes_full_kv_backend() -> None:
    model = _tiny_qwen_model()
    config = QwenOnlineEvalConfig(mode="full_kv", bits=3.0, device="cpu", torch_dtype="float32")
    patch_qwen_for_online_eval(model, config)

    cache = model.make_turboquant_cache()
    backend = cache.get_backend(0)
    assert isinstance(backend, TurboQuantCacheBackend)
    assert backend.quantize_values is True


def test_patch_qwen_for_online_eval_exact_uses_reference_backend() -> None:
    model = _tiny_qwen_model()
    config = QwenOnlineEvalConfig(mode="exact", device="cpu", torch_dtype="float32")
    patch_qwen_for_online_eval(model, config)

    cache = model.make_turboquant_cache()
    assert isinstance(cache.get_backend(0), ExactCacheBackend)


def test_patch_qwen_for_online_eval_fails_loudly_without_cache() -> None:
    model = _tiny_qwen_model()
    config = QwenOnlineEvalConfig(mode="key_only_random", bits=3.0, device="cpu", torch_dtype="float32")
    patch_qwen_for_online_eval(model, config)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with pytest.raises(ValueError, match="TurboQuantQwenCache"):
        model(input_ids=input_ids, use_cache=False)


def test_load_qwen_online_eval_model_uses_quantized_loader_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class DummyModel:
        def eval(self) -> "DummyModel":
            return self

    class DummyTokenizer:
        pass

    def fake_model_loader(model_name_or_path: str, **kwargs) -> DummyModel:
        observed["model_name_or_path"] = model_name_or_path
        observed["kwargs"] = kwargs
        return DummyModel()

    def fake_tokenizer_loader(tokenizer_source: str, **kwargs) -> DummyTokenizer:
        observed["tokenizer_source"] = tokenizer_source
        observed["tokenizer_kwargs"] = kwargs
        return DummyTokenizer()

    monkeypatch.setattr(
        "turboquant.adapters.hf_qwen.online_eval.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=fake_model_loader),
    )
    monkeypatch.setattr(
        "turboquant.adapters.hf_qwen.online_eval.AutoTokenizer",
        SimpleNamespace(from_pretrained=fake_tokenizer_loader),
    )
    monkeypatch.setattr(
        "turboquant.adapters.hf_qwen.online_eval.patch_qwen_for_online_eval",
        lambda model, config: model,
    )

    config = QwenOnlineEvalConfig(
        mode="exact",
        model_name_or_path="local-model",
        device="cuda",
        torch_dtype="float16",
        weight_load="4bit",
    )
    model, tokenizer = load_qwen_online_eval_model(config)

    assert isinstance(model, DummyModel)
    assert isinstance(tokenizer, DummyTokenizer)
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert "quantization_config" in kwargs
    assert kwargs["device_map"] == "auto"
