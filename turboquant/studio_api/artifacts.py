"""Artifact discovery helpers for TurboQuant Studio."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from turboquant.studio_api.models import ArtifactRecord


def _iso_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat()


def summarize_artifacts(artifact_root: Path) -> dict[str, object]:
    """Build a small summary used by the Studio shell."""

    known_paths = {
        "kv_capture_root": artifact_root / "kv_rtx3060_qwen9b",
        "qwen_matrix_root": artifact_root / "qwen_3060_matrix",
        "runtime_eval_root": artifact_root / "runtime_eval",
        "hf_online_eval_root": artifact_root / "hf_online_eval",
        "online_eval_report_root": artifact_root / "online_eval_report",
        "studio_root": artifact_root / "studio",
    }
    existing_files = sum(1 for path in artifact_root.rglob("*") if path.is_file()) if artifact_root.exists() else 0
    existing_dirs = sum(1 for path in artifact_root.rglob("*") if path.is_dir()) if artifact_root.exists() else 0
    return {
        "artifact_root": str(artifact_root),
        "existing_files": existing_files,
        "existing_directories": existing_dirs,
        "known_paths": {
            name: {
                "path": str(path),
                "exists": path.exists(),
            }
            for name, path in known_paths.items()
        },
    }


def build_artifact_tree(
    root: Path,
    *,
    relative_path: str = ".",
    max_depth: int = 3,
    max_children: int = 80,
) -> ArtifactRecord:
    """Build a bounded artifact tree for UI browsing."""

    target = (root / relative_path).resolve() if relative_path not in {"", "."} else root.resolve()
    if not target.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {target}")
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"Artifact path escapes artifact root: {target}")
    return _build_node(root.resolve(), target, max_depth=max_depth, max_children=max_children)


def _build_node(root: Path, path: Path, *, max_depth: int, max_children: int) -> ArtifactRecord:
    relative = path.relative_to(root).as_posix() if path != root else "."
    if path.is_file():
        return ArtifactRecord(
            relative_path=relative,
            absolute_path=str(path),
            kind="file",
            size_bytes=path.stat().st_size,
            modified_at=_iso_timestamp(path),
        )

    children = None
    if max_depth > 0:
        children = []
        entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in entries[:max_children]:
            children.append(_build_node(root, child, max_depth=max_depth - 1, max_children=max_children))
    return ArtifactRecord(
        relative_path=relative,
        absolute_path=str(path),
        kind="directory",
        modified_at=_iso_timestamp(path),
        children=children,
    )
