from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquant.capture import (
    CaptureMetadata,
    LayerCaptureRecord,
    git_commit_hash,
    normalize_past_key_values,
    package_versions,
    save_capture_metadata,
)
from turboquant.io_utils import ensure_dir, stable_hash
from turboquant.runtime import (
    BASE_MODEL_ID,
    DEFAULT_MODEL_ID,
    REQUIRED_CUDA,
    model_preset_to_id,
    require_supported_python,
    torch_cuda_version,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Qwen3.5 KV cache tensors.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-preset", choices=["default", "base"], default=None)
    parser.add_argument("--prompt", default="TurboQuant validation prompt.")
    parser.add_argument("--output-dir", default="artifacts/kv")
    parser.add_argument("--weight-load", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=96)
    return parser.parse_args()


def build_model_kwargs(args: argparse.Namespace) -> dict[str, object]:
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
    return model_kwargs


def main() -> int:
    require_supported_python()
    args = parse_args()
    if args.model_preset is not None:
        args.model_id = model_preset_to_id(args.model_preset)
    if torch_cuda_version(torch) != REQUIRED_CUDA or not torch.cuda.is_available():
        raise RuntimeError(
            f"capture_qwen_kv.py requires torch with CUDA {REQUIRED_CUDA} and a visible GPU."
        )
    output_dir = ensure_dir(Path(args.output_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **build_model_kwargs(args))
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True, return_dict=True)
    layers = normalize_past_key_values(outputs.past_key_values)

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
        model_name=args.model_id,
        tokenizer_name=tokenizer.name_or_path,
        prompt_hash=stable_hash(args.prompt),
        prompt_length=int(inputs["input_ids"].shape[-1]),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        device=str(model_device),
        dtype=str(next(model.parameters()).dtype),
        layer_count=len(records),
        package_versions=package_versions(["torch", "transformers", "bitsandbytes"]),
        git_commit_hash=git_commit_hash(Path.cwd()),
        layers=records,
    )
    save_capture_metadata(output_dir / "capture_manifest.json", metadata)
    print(f"saved {len(records)} layers to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
