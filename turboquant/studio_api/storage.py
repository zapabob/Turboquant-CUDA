"""SQLite-backed storage for TurboQuant Studio jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from turboquant.io_utils import ensure_dir
from turboquant.studio_api.models import JobSpec, JobStatus


def utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


class StudioStore:
    """Simple SQLite persistence for Studio job metadata."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        ensure_dir(db_path.parent)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    mode TEXT,
                    label TEXT,
                    artifact_root TEXT,
                    log_path TEXT,
                    exit_code INTEGER,
                    pid INTEGER,
                    error TEXT,
                    spec_json TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )

    def create_job(
        self,
        *,
        spec: JobSpec,
        mode: str | None,
        label: str | None,
        artifact_root: str | None,
        log_path: str | None,
    ) -> JobStatus:
        """Create and persist a queued job."""

        job_id = uuid.uuid4().hex
        created_at = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, dry_run, created_at, mode, label,
                    artifact_root, log_path, spec_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    spec.kind,
                    "queued",
                    int(spec.dry_run),
                    created_at,
                    mode,
                    label,
                    artifact_root,
                    log_path,
                    spec.model_dump_json(),
                ),
            )
        return self.get_job(job_id)

    def update_job(self, job_id: str, **updates: Any) -> JobStatus:
        """Apply partial updates and return the fresh job record."""

        if not updates:
            return self.get_job(job_id)
        normalized: dict[str, Any] = {}
        for key, value in updates.items():
            normalized["result_json" if key == "result" else key] = value
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = [self._serialize_value(value) for value in normalized.values()]
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {', '.join(f'{column} = ?' for column in normalized)} WHERE id = ?",
                (*values, job_id),
            )
        return self.get_job(job_id)

    def list_jobs(self) -> list[JobStatus]:
        """Return all jobs newest-first."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY datetime(created_at) DESC, id DESC"
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def get_job(self, job_id: str) -> JobStatus:
        """Return a single job by id."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown Studio job: {job_id}")
        return self._row_to_job(row)

    def _row_to_job(self, row: sqlite3.Row) -> JobStatus:
        spec = JobSpec.model_validate_json(row["spec_json"])
        result: dict[str, Any] | None = None
        if row["result_json"]:
            result = json.loads(row["result_json"])
        return JobStatus(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            dry_run=bool(row["dry_run"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            mode=row["mode"],
            label=row["label"],
            artifact_root=row["artifact_root"],
            log_path=row["log_path"],
            exit_code=row["exit_code"],
            pid=row["pid"],
            error=row["error"],
            spec=spec,
            result=result,
        )

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return json.dumps(value, indent=2, sort_keys=True)
        return value
