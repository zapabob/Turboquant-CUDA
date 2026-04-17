from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

from turboquant.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare standard online-eval inputs for TurboQuant Qwen experiments.")
    parser.add_argument("--output-dir", default="artifacts/online_eval_inputs", help="Prepared input artifact root.")
    parser.add_argument("--wikitext-config", default="wikitext-2-raw-v1", help="WikiText config name.")
    parser.add_argument("--wikitext-split", default="test", help="WikiText split name.")
    parser.add_argument("--max-wikitext-records", type=int, default=256, help="Max non-empty WikiText rows (0=all).")
    parser.add_argument("--mcq-items-per-task", type=int, default=16, help="Per-task MCQ manifest cap.")
    parser.add_argument("--gsm8k-items", type=int, default=16, help="GSM8K manifest cap.")
    return parser.parse_args()


def _select_rows(dataset, *, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(len(dataset)):
        rows.append(dict(dataset[index]))
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _prepare_wikitext(*, config_name: str, split: str, limit: int) -> str:
    dataset = load_dataset("wikitext", config_name, split=split)
    rows = _select_rows(dataset, limit=0)
    texts: list[str] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        texts.append(text)
        if limit > 0 and len(texts) >= limit:
            break
    return "\n\n".join(texts)


def _format_hellaswag(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for row in rows:
        ctx = f"{row['ctx_a']} {str(row['ctx_b']).capitalize()}".strip()
        manifest.append(
            {
                "task": "hellaswag",
                "doc_id": str(row["ind"]),
                "prompt": f"{row['activity_label']}: {ctx}",
                "choices": list(row["endings"]),
                "label": int(row["label"]),
            }
        )
    return manifest


def _format_piqa(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "task": "piqa",
            "doc_id": str(index),
            "prompt": str(row["goal"]),
            "choices": [str(row["sol1"]), str(row["sol2"])],
            "label": int(row["label"]),
        }
        for index, row in enumerate(rows)
    ]


def _format_arc(rows: list[dict[str, object]], *, task_name: str) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for row in rows:
        choice_labels = list(row["choices"]["label"])
        choice_texts = [str(text) for text in row["choices"]["text"]]
        manifest.append(
            {
                "task": task_name,
                "doc_id": str(row["id"]),
                "prompt": str(row["question"]),
                "choices": choice_texts,
                "label": choice_labels.index(str(row["answerKey"])),
            }
        )
    return manifest


def _format_mmlu(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        manifest.append(
            {
                "task": "mmlu",
                "doc_id": f"{row['subject']}:{index}",
                "prompt": str(row["question"]),
                "choices": list(row["choices"]),
                "label": int(row["answer"]),
                "subject": str(row["subject"]),
            }
        )
    return manifest


def _extract_gsm8k_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        return answer.strip()
    return answer.split(marker, maxsplit=1)[1].strip()


def _format_gsm8k(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "task": "gsm8k",
            "doc_id": str(index),
            "prompt": str(row["question"]),
            "reference": _extract_gsm8k_answer(str(row["answer"])),
        }
        for index, row in enumerate(rows)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    data_dir = ensure_dir(output_dir / "data")
    metrics_dir = ensure_dir(output_dir / "metrics")

    wikitext_text = _prepare_wikitext(
        config_name=args.wikitext_config,
        split=args.wikitext_split,
        limit=int(args.max_wikitext_records),
    )
    (data_dir / "wikitext2_test.txt").write_text(wikitext_text, encoding="utf-8")

    mcq_limit = int(args.mcq_items_per_task)
    mcq_rows: list[dict[str, object]] = []
    mcq_rows.extend(
        _format_hellaswag(
            _select_rows(load_dataset("Rowan/hellaswag", split=f"validation[:{mcq_limit}]"), limit=mcq_limit)
        )
    )
    mcq_rows.extend(
        _format_piqa(
            _select_rows(load_dataset("gimmaru/piqa", split=f"validation[:{mcq_limit}]"), limit=mcq_limit)
        )
    )
    mcq_rows.extend(
        _format_arc(
            _select_rows(load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"validation[:{mcq_limit}]"), limit=mcq_limit),
            task_name="arc_easy",
        )
    )
    mcq_rows.extend(
        _format_arc(
            _select_rows(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"validation[:{mcq_limit}]"), limit=mcq_limit),
            task_name="arc_challenge",
        )
    )
    mcq_rows.extend(
        _format_mmlu(
            _select_rows(load_dataset("cais/mmlu", "all", split=f"validation[:{mcq_limit}]"), limit=mcq_limit)
        )
    )
    _write_jsonl(data_dir / "mcq_manifest.jsonl", mcq_rows)

    gsm8k_rows = _format_gsm8k(
        _select_rows(load_dataset("gsm8k", "main", split=f"test[:{int(args.gsm8k_items)}]"), limit=int(args.gsm8k_items))
    )
    _write_jsonl(data_dir / "gsm8k_manifest.jsonl", gsm8k_rows)

    run_meta = {
        "wikitext_config": args.wikitext_config,
        "wikitext_split": args.wikitext_split,
        "max_wikitext_records": int(args.max_wikitext_records),
        "mcq_items_per_task": mcq_limit,
        "gsm8k_items": int(args.gsm8k_items),
        "mcq_task_counts": {
            task: sum(1 for row in mcq_rows if row["task"] == task)
            for task in ("hellaswag", "piqa", "arc_easy", "arc_challenge", "mmlu")
        },
        "gsm8k_count": len(gsm8k_rows),
    }
    (metrics_dir / "prepare_online_eval_inputs_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
