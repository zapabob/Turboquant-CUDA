"""Single-worker Studio job queue and execution helpers."""

from __future__ import annotations

import contextlib
import io
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from turboquant.io_utils import ensure_dir
from turboquant.studio_api.config import StudioSettings
from turboquant.studio_api.models import (
    CaptureFormState,
    ExportOnlineReportSpec,
    ExportReportSpec,
    JobSpec,
    JobStatus,
    MatrixRunSpec,
    PackageSpec,
    PaperValidateSpec,
    RuntimeEvalSpec,
    ServeSpec,
)
from turboquant.studio_api.storage import StudioStore, utc_now
from turboquant.studio_api.tasks import (
    JobCancelledError,
    build_serve_preview,
    capture_task,
    export_online_report_task,
    export_report_task,
    matrix_validate_task,
    package_gguf_task,
    paper_validate_task,
    runtime_eval_task,
)


class _LogWriter(io.TextIOBase):
    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8")

    def write(self, data: str) -> int:
        self._handle.write(data)
        self._handle.flush()
        return len(data)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class JobContext:
    """Runtime helpers for a single Studio job."""

    def __init__(
        self,
        *,
        job_id: str,
        log_path: Path,
        store: StudioStore,
        cancel_event: threading.Event,
    ) -> None:
        self.job_id = job_id
        self.log_path = log_path
        self.store = store
        self.cancel_event = cancel_event
        self._writer = _LogWriter(log_path)

    def log(self, message: str) -> None:
        self._writer.write(message if message.endswith("\n") else message + "\n")

    def run_python(self, callback) -> dict[str, Any]:
        if self.cancel_event.is_set():
            raise JobCancelledError("Job canceled before execution")
        with contextlib.redirect_stdout(self._writer), contextlib.redirect_stderr(self._writer):
            return callback()

    def run_process(
        self,
        *,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        long_running: bool = False,
    ) -> dict[str, Any]:
        if self.cancel_event.is_set():
            raise JobCancelledError("Job canceled before process launch")
        self.log("launch_command=" + subprocess.list2cmdline(command))
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.store.update_job(self.job_id, pid=process.pid)

        def _pump(stream, prefix: str) -> None:
            assert stream is not None
            for line in iter(stream.readline, ""):
                self.log(f"[{prefix}] {line.rstrip()}")
            stream.close()

        stdout_thread = threading.Thread(target=_pump, args=(process.stdout, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=_pump, args=(process.stderr, "stderr"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        while True:
            if self.cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise JobCancelledError("Job canceled while process was running")
            code = process.poll()
            if code is not None:
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                return {"exit_code": int(code), "pid": process.pid}
            if long_running:
                time.sleep(0.25)
            else:
                time.sleep(0.1)

    def close(self) -> None:
        self._writer.close()


def _model_from_spec(spec: JobSpec):
    mapping = {
        "capture": CaptureFormState,
        "paper-validate": PaperValidateSpec,
        "matrix-validate": MatrixRunSpec,
        "export-report": ExportReportSpec,
        "export-online-report": ExportOnlineReportSpec,
        "package-gguf": PackageSpec,
        "runtime-eval": RuntimeEvalSpec,
        "serve-llama": ServeSpec,
        "serve-hypura": ServeSpec,
    }
    model_cls = mapping.get(spec.kind)
    if model_cls is None:
        raise ValueError(f"Unsupported Studio job kind: {spec.kind}")
    return model_cls.model_validate(spec.payload | {"dry_run": spec.dry_run})


class StudioJobManager:
    """Single-worker job queue for the local Studio shell."""

    def __init__(self, *, settings: StudioSettings, store: StudioStore) -> None:
        self._settings = settings
        self._store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel_events: dict[str, threading.Event] = {}
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)

    def start(self) -> None:
        if not self._worker.is_alive():
            self._worker.start()

    def shutdown(self) -> None:
        self._worker_stop.set()
        self._queue.put("__shutdown__")
        if self._worker.is_alive():
            self._worker.join(timeout=5)

    def submit(self, *, spec: JobSpec, mode: str | None, label: str | None, artifact_root: str | None) -> JobStatus:
        job_dir = ensure_dir(self._settings.jobs_root / spec.kind)
        label_slug = (label or spec.kind).replace(":", "-").replace("/", "-")
        log_path = job_dir / f"{utc_now().replace(':', '-')}_{label_slug}.log"
        status = self._store.create_job(
            spec=spec,
            mode=mode,
            label=label,
            artifact_root=artifact_root,
            log_path=str(log_path),
        )
        self._cancel_events[status.id] = threading.Event()
        self._queue.put(status.id)
        return status

    def cancel(self, job_id: str) -> JobStatus:
        cancel_event = self._cancel_events.get(job_id)
        if cancel_event is None:
            raise KeyError(f"Unknown Studio job: {job_id}")
        job = self._store.get_job(job_id)
        if job.status == "queued":
            return self._store.update_job(job_id, status="canceled", finished_at=utc_now(), error="Canceled before start")
        cancel_event.set()
        if job.status == "running":
            return self._store.update_job(job_id, status="cancel_requested")
        return job

    def read_log_tail(self, job_id: str, *, max_chars: int = 12000) -> str:
        job = self._store.get_job(job_id)
        if not job.log_path:
            return ""
        path = Path(job.log_path)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]

    def _run_loop(self) -> None:
        while not self._worker_stop.is_set():
            job_id = self._queue.get()
            if job_id == "__shutdown__":
                return
            job = self._store.get_job(job_id)
            if job.status == "canceled":
                continue
            cancel_event = self._cancel_events[job_id]
            self._store.update_job(job_id, status="running", started_at=utc_now())
            log_path = Path(job.log_path or (self._settings.jobs_root / f"{job_id}.log"))
            context = JobContext(job_id=job_id, log_path=log_path, store=self._store, cancel_event=cancel_event)
            try:
                result = self._execute_job(context, job.spec)
            except JobCancelledError as exc:
                self._store.update_job(
                    job_id,
                    status="canceled",
                    finished_at=utc_now(),
                    error=str(exc),
                )
            except Exception as exc:  # pragma: no cover - exercised in integration
                context.log(f"error={exc!r}")
                self._store.update_job(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    error=repr(exc),
                )
            else:
                status = "completed" if int(result.get("exit_code", 0)) == 0 else "failed"
                self._store.update_job(
                    job_id,
                    status=status,
                    finished_at=utc_now(),
                    exit_code=int(result.get("exit_code", 0)) if "exit_code" in result else None,
                    result=result,
                    error=None if status == "completed" else result.get("error"),
                )
            finally:
                context.close()

    def _execute_job(self, context: JobContext, spec: JobSpec) -> dict[str, Any]:
        payload = _model_from_spec(spec)
        if spec.kind == "capture":
            return context.run_python(lambda: capture_task(payload))
        if spec.kind == "paper-validate":
            return context.run_python(lambda: paper_validate_task(payload))
        if spec.kind == "matrix-validate":
            return context.run_python(lambda: matrix_validate_task(payload))
        if spec.kind == "export-report":
            return context.run_python(lambda: export_report_task(payload))
        if spec.kind == "export-online-report":
            return context.run_python(lambda: export_online_report_task(payload))
        if spec.kind == "package-gguf":
            return context.run_python(lambda: package_gguf_task(payload))
        if spec.kind == "runtime-eval":
            return context.run_python(lambda: runtime_eval_task(payload))
        if spec.kind in {"serve-llama", "serve-hypura"}:
            preview = build_serve_preview(payload, kind=spec.kind)
            if payload.dry_run:
                return preview
            env = os.environ.copy()
            env.update(preview.get("env", {}))
            return context.run_process(
                command=preview["command"],
                cwd=self._settings.repo_root / "rust" if spec.kind == "serve-hypura" else None,
                env=env,
                long_running=True,
            )
        raise ValueError(f"Unsupported Studio job kind: {spec.kind}")
