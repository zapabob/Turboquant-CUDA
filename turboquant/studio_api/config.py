"""Configuration helpers for TurboQuant Studio."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from turboquant.io_utils import ensure_dir


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class StudioSettings:
    """Filesystem settings for the local TurboQuant Studio instance."""

    repo_root: Path
    artifact_root: Path
    studio_root: Path
    jobs_root: Path
    db_path: Path
    frontend_dist: Path
    frontend_index: Path

    @classmethod
    def from_env(cls) -> "StudioSettings":
        artifact_root = Path(
            os.environ.get("TURBOQUANT_STUDIO_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts"))
        ).resolve()
        studio_root = artifact_root / "studio"
        jobs_root = studio_root / "jobs"
        frontend_dist = Path(
            os.environ.get("TURBOQUANT_STUDIO_FRONTEND_DIST", str(REPO_ROOT / "studio-web" / "dist"))
        ).resolve()
        return cls(
            repo_root=REPO_ROOT,
            artifact_root=ensure_dir(artifact_root),
            studio_root=ensure_dir(studio_root),
            jobs_root=ensure_dir(jobs_root),
            db_path=studio_root / "studio.db",
            frontend_dist=frontend_dist,
            frontend_index=frontend_dist / "index.html",
        )
