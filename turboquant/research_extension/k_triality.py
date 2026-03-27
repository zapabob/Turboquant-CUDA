"""Research-only K-side triality proxy training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path

import pandas as pd
from scipy import stats
import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import (
    _peak_vram_mb,
    _reset_peak_if_cuda,
    _sync_if_cuda,
    evaluate_layer_grid,
    load_captured_runs,
    raw_storage_bits,
    scaled_attention_output,
    select_queries,
    summarize_trial_metrics,
)
from turboquant.attention_metrics import summarize_attention_scores
from turboquant.research_extension.triality_proxy import TRIALITY_PROXY_VIEWS, TrialityProxyProd
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantProdConfig


TRIALITY_MODE_BY_VIEW = {
    "vector": "key_only_block_so8_triality_vector",
    "spinor_plus_proxy": "key_only_block_so8_triality_plus",
    "spinor_minus_proxy": "key_only_block_so8_triality_minus",
}


def _canonical_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else device.index}"
    return str(device)


@dataclass(slots=True)
class TrialityRotationArtifact:
    layer_idx: int
    bits: float
    view: str
    rotation: torch.Tensor
    rotation_seed: int
    qjl_seed: int


def triality_mode_name(view: str) -> str:
    try:
        return TRIALITY_MODE_BY_VIEW[view]
    except KeyError as exc:
        raise ValueError(f"Unsupported triality view: {view!r}") from exc


def _seed_value(layer_idx: int, bit_value: float, view_index: int, salt: int) -> int:
    return 70_000 + (layer_idx * 2_003) + int(bit_value * 100) + (view_index * 257) + salt


def _stage1_allocation(bits: float, qjl_bits: int = 1, width: int = 128) -> ChannelBitAllocation | None:
    if float(bits).is_integer():
        return None
    effective_bits = bits - qjl_bits
    if effective_bits <= 0:
        raise ValueError("effective stage-1 bits must remain positive")
    return ChannelBitAllocation.preset(effective_bits=effective_bits, width=width)


def _bundle_groups(kv_root: Path, max_layers: int = 0) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for bundle in load_captured_runs(kv_root):
        if max_layers > 0 and bundle.layer_idx >= max_layers:
            continue
        grouped.setdefault(bundle.layer_idx, []).append(bundle)
    return grouped


def _stack_training_keys_and_queries(
    bundles: list,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys_list: list[torch.Tensor] = []
    queries_list: list[torch.Tensor] = []
    for bundle_index, bundle in enumerate(bundles):
        keys = bundle.keys.to(device)
        keys_list.append(keys)
        queries_list.append(select_queries(keys, seed=90_000 + (bundle.layer_idx * 131) + bundle_index))
    keys = torch.cat(keys_list, dim=-2)
    queries = torch.cat(queries_list, dim=-2)
    return keys, queries


def fit_triality_proxy_rotations(
    *,
    kv_root: Path,
    bit_grid: list[float],
    max_layers: int = 0,
    steps: int = 60,
    lr: float = 5e-2,
    device: str | torch.device = "cpu",
) -> tuple[list[TrialityRotationArtifact], pd.DataFrame]:
    target_device = torch.device(device)
    device_name = _canonical_device_name(target_device)
    artifacts: list[TrialityRotationArtifact] = []
    rows: list[dict[str, float | int | str]] = []
    grouped = _bundle_groups(kv_root, max_layers=max_layers)
    for layer_idx, bundles in sorted(grouped.items()):
        keys, queries = _stack_training_keys_and_queries(bundles, device=target_device)
        for bit_value in bit_grid:
            total_bits = int(math.floor(bit_value))
            allocation = _stage1_allocation(bit_value, width=keys.shape[-1])
            for view_index, view in enumerate(TRIALITY_PROXY_VIEWS):
                rotation_seed = _seed_value(layer_idx, bit_value, view_index, 17)
                qjl_seed = _seed_value(layer_idx, bit_value, view_index, 71)
                quantizer = TurboQuantProd(
                    TurboQuantProdConfig(
                        dim=keys.shape[-1],
                        total_bits=total_bits,
                        rotation_seed=rotation_seed,
                        rotation_policy="block_so8_learned",
                        qjl_seed=qjl_seed,
                        device=device_name,
                        dtype=str(keys.dtype).split(".")[-1],
                    )
                )
                proxy = TrialityProxyProd(quantizer=quantizer, view=view)
                proxy.fit_rotation(keys, queries=queries, steps=steps, lr=lr)
                encoded = proxy.quantize(keys, allocation=allocation)
                estimated_logits = proxy.pairwise_estimate(queries, encoded)
                exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
                metrics = summarize_attention_scores(exact_logits, estimated_logits)
                rotation = proxy.quantizer.mse_quantizer.rotation.detach().cpu()
                artifacts.append(
                    TrialityRotationArtifact(
                        layer_idx=layer_idx,
                        bits=float(bit_value),
                        view=view,
                        rotation=rotation,
                        rotation_seed=rotation_seed,
                        qjl_seed=qjl_seed,
                    )
                )
                ident = rotation.transpose(0, 1) @ rotation
                rows.append(
                    {
                        "layer": layer_idx,
                        "bits": float(bit_value),
                        "bit_setting": f"{bit_value:g}",
                        "view": view,
                        "mode": triality_mode_name(view),
                        "rotation_seed": rotation_seed,
                        "qjl_seed": qjl_seed,
                        "prompt_count": len(bundles),
                        "token_count": int(keys.shape[-2]),
                        "orthogonality_error": float((ident - torch.eye(rotation.shape[0])).abs().max().item()),
                        "train_logit_cosine_similarity": metrics["cosine_similarity"],
                        "train_logit_mse": metrics["mse"],
                    }
                )
    return artifacts, pd.DataFrame(rows)


def save_triality_proxy_rotations(artifacts: list[TrialityRotationArtifact], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        file_name = f"layer_{artifact.layer_idx:02d}_bits_{artifact.bits:g}_{artifact.view}.pt"
        path = output_dir / file_name
        torch.save(
            {
                "layer": artifact.layer_idx,
                "bits": artifact.bits,
                "view": artifact.view,
                "rotation": artifact.rotation,
                "rotation_seed": artifact.rotation_seed,
                "qjl_seed": artifact.qjl_seed,
            },
            path,
        )
        rows.append(
            {
                "layer": artifact.layer_idx,
                "bits": artifact.bits,
                "bit_setting": f"{artifact.bits:g}",
                "view": artifact.view,
                "mode": triality_mode_name(artifact.view),
                "rotation_path": str(path).replace("\\", "/"),
                "rotation_seed": artifact.rotation_seed,
                "qjl_seed": artifact.qjl_seed,
            }
        )
    return pd.DataFrame(rows)


def load_triality_proxy_rotations(rotation_dir: Path) -> dict[tuple[int, float, str], TrialityRotationArtifact]:
    artifacts: dict[tuple[int, float, str], TrialityRotationArtifact] = {}
    for path in sorted(rotation_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        artifact = TrialityRotationArtifact(
            layer_idx=int(payload["layer"]),
            bits=float(payload["bits"]),
            view=str(payload["view"]),
            rotation=payload["rotation"].to(dtype=torch.float32),
            rotation_seed=int(payload["rotation_seed"]),
            qjl_seed=int(payload["qjl_seed"]),
        )
        artifacts[(artifact.layer_idx, artifact.bits, artifact.view)] = artifact
    if not artifacts:
        raise FileNotFoundError(f"No triality rotation artifacts found under {rotation_dir}")
    return artifacts


def _validate_triality_artifacts(
    *,
    bundles: list,
    bit_grid: list[float],
    artifacts: dict[tuple[int, float, str], TrialityRotationArtifact],
    max_layers: int = 0,
) -> None:
    expected: set[tuple[int, float, str]] = set()
    included_layers = sorted({bundle.layer_idx for bundle in bundles if max_layers <= 0 or bundle.layer_idx < max_layers})
    for layer_idx in included_layers:
        for bit_value in bit_grid:
            for view in TRIALITY_PROXY_VIEWS:
                expected.add((layer_idx, float(bit_value), view))
    missing = sorted(expected.difference(artifacts.keys()))
    if missing:
        preview = ", ".join(f"(layer={layer}, bits={bits:g}, view={view})" for layer, bits, view in missing[:12])
        suffix = "" if len(missing) <= 12 else f" ... and {len(missing) - 12} more"
        raise KeyError(f"Missing triality rotation artifacts: {preview}{suffix}")


def _evaluate_triality_proxy_mode(
    *,
    dataset: str,
    trial: int,
    layer_idx: int,
    bit_value: float,
    view: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    artifact: TrialityRotationArtifact,
) -> dict[str, float | int | str]:
    device = keys.device
    allocation = _stage1_allocation(bit_value, width=keys.shape[-1])
    quantizer = TurboQuantProd(
        TurboQuantProdConfig(
            dim=keys.shape[-1],
            total_bits=int(math.floor(bit_value)),
            rotation_seed=artifact.rotation_seed,
            rotation_policy="block_so8_static",
            qjl_seed=artifact.qjl_seed,
            device=_canonical_device_name(torch.device(device)),
            dtype=str(keys.dtype).split(".")[-1],
        )
    )
    proxy = TrialityProxyProd(quantizer=quantizer, view=view)
    proxy.set_rotation(artifact.rotation.to(device=device, dtype=keys.dtype))

    exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
    exact_hidden = scaled_attention_output(exact_logits, values, head_dim=keys.shape[-1])
    exact_memory_bits = raw_storage_bits(keys) + raw_storage_bits(values)

    _sync_if_cuda(device)
    _reset_peak_if_cuda(device)
    started = time.perf_counter()
    encoded_keys = proxy.quantize(keys, allocation=allocation)
    estimated_logits = proxy.pairwise_estimate(queries, encoded_keys)
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - started

    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    last_query = queries[..., -1:, :]
    decode_logits = proxy.pairwise_estimate(last_query, encoded_keys)
    _ = scaled_attention_output(decode_logits, values, head_dim=keys.shape[-1])
    _sync_if_cuda(device)
    decode_seconds = time.perf_counter() - decode_started

    hidden = scaled_attention_output(estimated_logits, values, head_dim=keys.shape[-1])
    logit_metrics = summarize_attention_scores(exact_logits, estimated_logits)
    hidden_metrics = summarize_attention_scores(exact_hidden, hidden)
    memory_bits = float(encoded_keys.total_bits() + raw_storage_bits(values))
    return {
        "dataset": dataset,
        "trial": trial,
        "layer": layer_idx,
        "mode": triality_mode_name(view),
        "bit_setting": f"{bit_value:g}",
        "bits": float(bit_value),
        "view": view,
        "logit_cosine_similarity": logit_metrics["cosine_similarity"],
        "logit_mae": logit_metrics["mae"],
        "logit_mse": logit_metrics["mse"],
        "next_logit_kl": logit_metrics["kl_divergence"],
        "logit_spearman": logit_metrics["spearman"],
        "logit_top1_match": logit_metrics["top1_match"],
        "logit_top5_match": logit_metrics["top5_match"],
        "logit_top5_overlap": logit_metrics["top5_overlap"],
        "hidden_cosine_similarity": hidden_metrics["cosine_similarity"],
        "hidden_mae": hidden_metrics["mae"],
        "hidden_mse": hidden_metrics["mse"],
        "attention_output_relative_error": hidden_metrics["relative_fro_error"],
        "memory_bits": memory_bits,
        "memory_ratio_vs_exact": memory_bits / float(exact_memory_bits),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "peak_vram_mb": _peak_vram_mb(device),
        "key_mode": triality_mode_name(view),
        "value_mode": "exact",
        "value_rotation_policy": "exact",
        "rotation_seed": artifact.rotation_seed,
        "qjl_seed": artifact.qjl_seed,
    }


def evaluate_triality_proxy_captured(
    *,
    kv_root: Path,
    trial_count: int,
    bit_grid: list[float],
    rotation_dir: Path,
    max_layers: int = 0,
    eval_device: str | torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = load_triality_proxy_rotations(rotation_dir)
    rows: list[dict[str, float | int | str]] = []
    target_device = torch.device(eval_device) if eval_device is not None else None
    bundles = load_captured_runs(kv_root)
    _validate_triality_artifacts(
        bundles=bundles,
        bit_grid=bit_grid,
        artifacts=artifacts,
        max_layers=max_layers,
    )
    for trial in range(trial_count):
        for bundle in bundles:
            if max_layers > 0 and bundle.layer_idx >= max_layers:
                continue
            keys = bundle.keys.to(target_device) if target_device is not None else bundle.keys
            values = bundle.values.to(target_device) if target_device is not None else bundle.values
            rows.extend(
                {
                    **row,
                    "capture_id": bundle.metadata.capture_id or bundle.capture_dir.name,
                    "prompt_label": bundle.metadata.prompt_label or "unknown",
                    "prompt_hash": bundle.metadata.prompt_hash,
                }
                for row in evaluate_layer_grid(
                    dataset=f"research_captured:{bundle.metadata.prompt_label}",
                    keys=keys,
                    values=values,
                    trial=trial,
                    layer_idx=bundle.layer_idx,
                    bit_grid=bit_grid,
                    eval_device=target_device,
                )
                if row["mode"] in {"exact", "key_only_random", "key_only_block_so8_learned", "full_kv"}
            )
            queries = select_queries(keys, seed=10_000 + (trial * 257) + bundle.layer_idx)
            for bit_value in bit_grid:
                for view in TRIALITY_PROXY_VIEWS:
                    artifact = artifacts[(bundle.layer_idx, float(bit_value), view)]
                    row = _evaluate_triality_proxy_mode(
                        dataset=f"research_captured:{bundle.metadata.prompt_label}",
                        trial=trial,
                        layer_idx=bundle.layer_idx,
                        bit_value=bit_value,
                        view=view,
                        keys=keys,
                        values=values,
                        queries=queries,
                        artifact=artifact,
                    )
                    row["capture_id"] = bundle.metadata.capture_id or bundle.capture_dir.name
                    row["prompt_label"] = bundle.metadata.prompt_label or "unknown"
                    row["prompt_hash"] = bundle.metadata.prompt_hash
                    rows.append(row)
    trial_frame = pd.DataFrame(rows)
    if trial_frame.empty:
        included_bundles = [bundle for bundle in bundles if max_layers <= 0 or bundle.layer_idx < max_layers]
        raise ValueError(
            "Triality replay produced no rows "
            f"(bundles={len(included_bundles)}, bits={bit_grid}, max_layers={max_layers}, artifacts={len(artifacts)})"
        )
    return trial_frame, summarize_trial_metrics(trial_frame)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        candidate = (total - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def compute_triality_statistics(trial_frame: pd.DataFrame) -> pd.DataFrame:
    if trial_frame.empty:
        raise ValueError("compute_triality_statistics received an empty trial_frame")
    required_columns = {"mode", "dataset", "trial", "layer", "bit_setting", "bits"}
    missing_columns = sorted(required_columns.difference(trial_frame.columns))
    if missing_columns:
        raise ValueError(f"trial_frame is missing required columns: {missing_columns}")
    pair_columns = ["dataset", "trial", "layer", "bit_setting", "bits"]
    for optional in ("capture_id", "prompt_label", "prompt_hash"):
        if optional in trial_frame.columns:
            pair_columns.append(optional)
    metrics = (
        ("hidden_cosine_similarity", "greater"),
        ("next_logit_kl", "less"),
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for metric, alternative in metrics:
        p_values: list[float] = []
        row_indices: list[int] = []
        baseline = trial_frame.loc[trial_frame["mode"] == "key_only_block_so8_learned", pair_columns + [metric]].rename(
            columns={metric: "baseline_value"}
        )
        for view in TRIALITY_PROXY_VIEWS:
            mode = triality_mode_name(view)
            candidate = trial_frame.loc[trial_frame["mode"] == mode, pair_columns + [metric]].rename(
                columns={metric: "candidate_value"}
            )
            paired = baseline.merge(candidate, on=pair_columns, how="inner")
            if paired.empty:
                continue
            try:
                result = stats.wilcoxon(
                    paired["candidate_value"],
                    paired["baseline_value"],
                    alternative=alternative,
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                statistic = 0.0
                p_value = 1.0
            row = {
                "metric": metric,
                "view": view,
                "mode": mode,
                "test": "wilcoxon",
                "alternative": alternative,
                "n_pairs": int(len(paired)),
                "statistic": statistic,
                "p_value": p_value,
                "candidate_mean": float(paired["candidate_value"].mean()),
                "baseline_mean": float(paired["baseline_value"].mean()),
                "delta_candidate_minus_baseline": float(
                    paired["candidate_value"].mean() - paired["baseline_value"].mean()
                ),
            }
            rows.append(row)
            p_values.append(p_value)
            row_indices.append(len(rows) - 1)
        adjusted = holm_bonferroni(p_values)
        for row_index, adjusted_p in zip(row_indices, adjusted, strict=True):
            rows[row_index]["p_value_holm"] = adjusted_p
            rows[row_index]["significant_0_05"] = adjusted_p < 0.05
    return pd.DataFrame(rows)
