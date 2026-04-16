from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from turboquant.capture import (
    CaptureMetadata,
    DEFAULT_PROMPT_PANEL,
    LayerCaptureRecord,
    PromptCaptureSpec,
    capture_slug,
    git_commit_hash,
    normalize_past_key_values,
    package_versions,
    save_capture_metadata,
)
from turboquant.io_utils import ensure_dir, stable_hash
from turboquant.runtime import (
    DEFAULT_CAPTURE_MODEL_PRESET,
    REQUIRED_CUDA,
    get_model_preset,
    model_preset_names,
    require_supported_python,
    torch_cuda_version,
)
from turboquant.schema import build_capture_quantization_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Qwen3.5 KV cache tensors.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-preset", choices=model_preset_names(), default=None)
    parser.add_argument("--lane-name", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-label", default="custom")
    parser.add_argument("--output-dir", default="artifacts/kv")
    parser.add_argument("--weight-load", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_model_kwargs(args: argparse.Namespace) -> dict[str, object]:
    from transformers import BitsAndBytesConfig

    model_kwargs: dict[str, object] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.weight_load in {"4bit", "8bit"}:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.weight_load == "4bit",
            load_in_8bit=args.weight_load == "8bit",
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = getattr(torch, args.dtype)
        # Full weights without bitsandbytes: accelerate device_map (no quantization).
        model_kwargs["device_map"] = "auto"
    return model_kwargs


def prompt_specs_from_args(args: argparse.Namespace) -> list[PromptCaptureSpec]:
    if args.prompt is not None:
        return [PromptCaptureSpec(label=args.prompt_label, prompt=args.prompt)]
    return list(DEFAULT_PROMPT_PANEL)


def resolve_capture_target(args: argparse.Namespace) -> tuple[str, str, str | None, str]:
    """Resolve the capture model name, source, preset, and VRAM lane."""

    explicit_preset = args.model_preset is not None
    preset = get_model_preset(args.model_preset or DEFAULT_CAPTURE_MODEL_PRESET)
    if args.model_id is None:
        model_name = preset.model_id
        model_source = preset.resolved_model_source()
        model_preset = preset.name
        lane_name = args.lane_name or preset.lane_name
        return model_name, model_source, model_preset, lane_name

    model_source = args.model_id
    model_name = preset.model_id if explicit_preset else args.model_id
    model_preset = preset.name if explicit_preset else None
    lane_name = args.lane_name or (preset.lane_name if explicit_preset else "custom")
    return model_name, model_source, model_preset, lane_name


def save_single_capture(
    *,
    output_root: Path,
    prompt_spec: PromptCaptureSpec,
    model_name: str,
    model_source: str,
    model_preset: str | None,
    lane_name: str,
    seed: int,
    quantization_config: dict[str, object],
    tokenizer_name: str,
    prompt_length: int,
    model_device: torch.device,
    model_dtype: torch.dtype,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
) -> Path:
    prompt_hash = stable_hash(prompt_spec.prompt)
    capture_id = capture_slug(prompt_spec.label, prompt_hash)
    output_dir = ensure_dir(output_root / capture_id)
    records: list[LayerCaptureRecord] = []
    for layer_idx, (key, value) in enumerate(layers):
        key_path = output_dir / f"layer_{layer_idx:02d}_key.pt"
        value_path = output_dir / f"layer_{layer_idx:02d}_value.pt"
        torch.save(key.detach().cpu(), key_path)
        torch.save(value.detach().cpu(), value_path)
        records.append(
            LayerCaptureRecord(
                layer_index=layer_idx,
                key_shape=list(key.shape),
                value_shape=list(value.shape),
                key_file=key_path.name,
                value_file=value_path.name,
            )
        )

    metadata = CaptureMetadata(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        prompt_hash=prompt_hash,
        prompt_length=prompt_length,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        device=str(model_device),
        dtype=str(model_dtype),
        layer_count=len(records),
        package_versions=package_versions(["torch", "transformers", "bitsandbytes"]),
        git_commit_hash=git_commit_hash(Path.cwd()),
        layers=records,
        model_source=model_source,
        prompt_label=prompt_spec.label,
        capture_id=capture_id,
        model_preset=model_preset,
        lane_name=lane_name,
        seed=seed,
        quantization_config=quantization_config,
    )
    save_capture_metadata(output_dir / "capture_manifest.json", metadata)
    return output_dir


def main() -> int:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "capture_qwen_kv.py requires the optional HF/Qwen dependencies. "
            "Install with `uv sync --extra cu128 --extra dev --extra hf_qwen`."
        ) from exc

    require_supported_python()
    args = parse_args()
    if torch_cuda_version(torch) != REQUIRED_CUDA or not torch.cuda.is_available():
        raise RuntimeError(
            f"capture_qwen_kv.py requires torch with CUDA {REQUIRED_CUDA} and a visible GPU."
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name, model_source, model_preset, lane_name = resolve_capture_target(args)
    output_dir = ensure_dir(Path(args.output_dir))
    prompt_specs = prompt_specs_from_args(args)
    quantization_config = build_capture_quantization_config(
        weight_load=args.weight_load,
        requested_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_source, **build_model_kwargs(args))
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    saved_paths: list[Path] = []
    for prompt_spec in prompt_specs:
        inputs = tokenizer(
            prompt_spec.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True, return_dict=True)
        layers = normalize_past_key_values(outputs.past_key_values)
        capture_dir = save_single_capture(
            output_root=output_dir,
            prompt_spec=prompt_spec,
            model_name=model_name,
            model_source=model_source,
            model_preset=model_preset,
            lane_name=lane_name,
            seed=args.seed,
            quantization_config=quantization_config,
            tokenizer_name=tokenizer.name_or_path,
            prompt_length=int(inputs["input_ids"].shape[-1]),
            model_device=model_device,
            model_dtype=model_dtype,
            layers=layers,
        )
        saved_paths.append(capture_dir)
    print(f"saved {len(saved_paths)} prompt captures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
