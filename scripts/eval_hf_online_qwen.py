from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from turboquant.adapters.hf_qwen import QwenOnlineEvalConfig, load_qwen_online_eval_model
from turboquant.eval_stats import summarize_benchmark_items, summarize_continuous_metrics
from turboquant.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF online TurboQuant evaluation for Qwen3.")
    parser.add_argument("--mode", required=True, help="TurboQuant mode to evaluate.")
    parser.add_argument("--bits", type=float, default=4.0, help="Comparison bit setting.")
    parser.add_argument("--model-path", default=None, help="Model path or HF id. Defaults to local capture model path.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path override.")
    parser.add_argument("--device", default="cpu", help="Torch device for evaluation.")
    parser.add_argument("--dtype", default="float32", help="Torch dtype name for model load.")
    parser.add_argument("--weight-load", choices=["4bit", "8bit", "none"], default="4bit", help="Model weight-load path.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code to HF loaders.")
    parser.add_argument("--device-map", default=None, help="Optional Accelerate device_map override.")
    parser.add_argument("--triality-rotation-dir", default=None, help="Required for triality-vector mode.")
    parser.add_argument("--text-file", default=None, help="Plain-text corpus for perplexity evaluation.")
    parser.add_argument(
        "--choice-manifest",
        default=None,
        help="JSONL manifest with task/doc_id/prompt/choices/label fields for multiple-choice scoring.",
    )
    parser.add_argument("--chunk-length", type=int, default=256, help="Token chunk length for PPL evaluation.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional cap on PPL chunks (0=all).")
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap on benchmark items (0=all).")
    parser.add_argument("--output-dir", default="artifacts/hf_online_eval", help="Artifact output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands/metadata without loading the model.")
    return parser.parse_args()


def _load_choice_manifest(path: Path, *, max_items: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(payload)
        if max_items > 0 and index + 1 >= max_items:
            break
    return rows


def _score_choice(
    *,
    model,
    tokenizer,
    prompt: str,
    choice: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if prompt_ids.numel() == 0:
        raise ValueError("choice manifest prompt must produce at least one token")
    if choice_ids.numel() == 0:
        raise ValueError("choice manifest choices must produce at least one token")
    input_ids = torch.cat([prompt_ids, choice_ids], dim=-1)
    cache = model.make_turboquant_cache()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    choice_start = prompt_ids.shape[-1] - 1
    log_probs = torch.log_softmax(logits[:, choice_start:, :], dim=-1)
    target_slice = targets[:, choice_start:]
    gathered = log_probs.gather(dim=-1, index=target_slice.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


def _evaluate_benchmark_items(
    *,
    model,
    tokenizer,
    items: list[dict[str, object]],
    mode: str,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in items:
        task = str(item["task"])
        doc_id = str(item["doc_id"])
        prompt = str(item["prompt"])
        choices = list(item["choices"])
        label = int(item["label"])
        scores = [
            _score_choice(model=model, tokenizer=tokenizer, prompt=prompt, choice=str(choice), device=device)
            for choice in choices
        ]
        prediction = int(max(range(len(scores)), key=lambda idx: scores[idx]))
        rows.append(
            {
                "backend": "hf",
                "mode": mode,
                "task": task,
                "doc_id": doc_id,
                "prediction": prediction,
                "reference": label,
                "is_correct": int(prediction == label),
                "score_margin": float(scores[prediction] - scores[label]) if prediction != label else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_ppl_chunks(
    *,
    model,
    tokenizer,
    text: str,
    mode: str,
    chunk_length: int,
    max_chunks: int,
    device: torch.device,
) -> pd.DataFrame:
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)
    rows: list[dict[str, object]] = []
    chunk_index = 0
    for start in range(0, max(int(token_ids.shape[0]) - 1, 0), chunk_length):
        stop = min(start + chunk_length, int(token_ids.shape[0]))
        chunk = token_ids[start:stop]
        if chunk.numel() < 2:
            continue
        cache = model.make_turboquant_cache()
        with torch.no_grad():
            outputs = model(input_ids=chunk.unsqueeze(0), use_cache=True, past_key_values=cache)
        logits = outputs.logits[:, :-1, :]
        targets = chunk[1:].unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        rows.append(
            {
                "backend": "hf",
                "mode": mode,
                "chunk_id": chunk_index,
                "token_count": int(chunk.numel()),
                "log_perplexity": float(loss.item()),
                "perplexity": float(torch.exp(loss).item()),
            }
        )
        chunk_index += 1
        if max_chunks > 0 and chunk_index >= max_chunks:
            break
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    logs_dir = ensure_dir(output_dir / "logs")
    run_meta = {
        "mode": args.mode,
        "bits": args.bits,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "device": args.device,
        "dtype": args.dtype,
        "weight_load": args.weight_load,
        "trust_remote_code": args.trust_remote_code,
        "device_map": args.device_map,
        "triality_rotation_dir": args.triality_rotation_dir,
        "text_file": args.text_file,
        "choice_manifest": args.choice_manifest,
        "chunk_length": args.chunk_length,
        "max_chunks": args.max_chunks,
        "max_items": args.max_items,
        "dry_run": args.dry_run,
    }
    (metrics_dir / "hf_online_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (logs_dir / "hf_online_command.txt").write_text("python " + " ".join(sys.argv), encoding="utf-8")

    if args.dry_run:
        return 0

    config = QwenOnlineEvalConfig(
        mode=args.mode,
        bits=float(args.bits),
        model_name_or_path=args.model_path or str(QwenOnlineEvalConfig(mode=args.mode).model_name_or_path),
        tokenizer_name_or_path=args.tokenizer_path,
        device=args.device,
        torch_dtype=args.dtype,
        weight_load=args.weight_load,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        triality_rotation_dir=args.triality_rotation_dir,
    )
    model, tokenizer = load_qwen_online_eval_model(config)
    device = config.resolved_device()

    if args.text_file is not None:
        text = Path(args.text_file).read_text(encoding="utf-8")
        ppl_frame = _evaluate_ppl_chunks(
            model=model,
            tokenizer=tokenizer,
            text=text,
            mode=args.mode,
            chunk_length=int(args.chunk_length),
            max_chunks=int(args.max_chunks),
            device=device,
        )
        ppl_frame.to_csv(metrics_dir / "hf_online_ppl_chunks.csv", index=False)
        ppl_long = pd.concat(
            [
                ppl_frame[["backend", "mode", "chunk_id", "perplexity"]]
                .rename(columns={"chunk_id": "sample_id", "perplexity": "value"})
                .assign(metric="perplexity"),
                ppl_frame[["backend", "mode", "chunk_id", "log_perplexity"]]
                .rename(columns={"chunk_id": "sample_id", "log_perplexity": "value"})
                .assign(metric="log_perplexity"),
            ],
            ignore_index=True,
        )
        ppl_summary = summarize_continuous_metrics(
            ppl_long,
            group_columns=["backend", "mode", "metric"],
        )
        ppl_summary.to_csv(metrics_dir / "hf_online_ppl_summary.csv", index=False)

    if args.choice_manifest is not None:
        items = _load_choice_manifest(Path(args.choice_manifest), max_items=int(args.max_items))
        benchmark_frame = _evaluate_benchmark_items(
            model=model,
            tokenizer=tokenizer,
            items=items,
            mode=args.mode,
            device=device,
        )
        benchmark_frame.to_csv(metrics_dir / "hf_online_benchmark_items.csv", index=False)
        benchmark_summary = summarize_benchmark_items(
            benchmark_frame,
            group_columns=["backend", "mode", "task"],
        )
        benchmark_summary.to_csv(metrics_dir / "hf_online_benchmark_summary.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
