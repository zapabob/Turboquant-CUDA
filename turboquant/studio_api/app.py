"""FastAPI application for TurboQuant Studio."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from turboquant.studio_api.artifacts import build_artifact_tree, summarize_artifacts
from turboquant.studio_api.config import StudioSettings
from turboquant.studio_api.jobs import StudioJobManager
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
from turboquant.studio_api.setup import build_setup_snapshot
from turboquant.studio_api.storage import StudioStore


def create_app(settings: StudioSettings | None = None) -> FastAPI:
    """Create the TurboQuant Studio FastAPI app."""

    resolved = settings or StudioSettings.from_env()
    store = StudioStore(resolved.db_path)
    manager = StudioJobManager(settings=resolved, store=store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved
        app.state.store = store
        app.state.manager = manager
        manager.start()
        yield
        manager.shutdown()

    app = FastAPI(title="TurboQuant Studio", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if resolved.artifact_root.exists():
        app.mount("/artifacts", StaticFiles(directory=resolved.artifact_root), name="artifacts")
    if resolved.frontend_dist.exists() and (resolved.frontend_dist / "assets").exists():
        app.mount("/studio/assets", StaticFiles(directory=resolved.frontend_dist / "assets"), name="studio-assets")

    def _job_with_log(job_id: str) -> JobStatus:
        job = store.get_job(job_id)
        return job.model_copy(update={"log_tail": manager.read_log_tail(job_id)})

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "artifact_root": str(resolved.artifact_root),
            "frontend_ready": resolved.frontend_index.exists(),
        }

    @app.get("/api/setup")
    def setup() -> dict[str, object]:
        return build_setup_snapshot(resolved).model_dump()

    @app.get("/api/runs")
    def runs() -> list[dict[str, object]]:
        return [job.model_dump() for job in store.list_jobs()]

    @app.get("/api/runs/{job_id}")
    def run_detail(job_id: str) -> dict[str, object]:
        try:
            return _job_with_log(job_id).model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/artifacts/summary")
    def artifacts_summary() -> dict[str, object]:
        return summarize_artifacts(resolved.artifact_root)

    @app.get("/api/artifacts/tree")
    def artifacts_tree(relative_path: str = ".", max_depth: int = 3) -> dict[str, object]:
        try:
            return build_artifact_tree(
                resolved.artifact_root,
                relative_path=relative_path,
                max_depth=max_depth,
            ).model_dump()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _submit(
        kind: str,
        payload_model,
        *,
        mode: str | None = None,
        label: str | None = None,
        artifact_root: str | None = None,
    ) -> dict[str, object]:
        spec = JobSpec(
            kind=kind,
            dry_run=payload_model.dry_run,
            payload=payload_model.model_dump(exclude={"dry_run"}),
        )
        return manager.submit(spec=spec, mode=mode, label=label, artifact_root=artifact_root).model_dump()

    @app.post("/api/jobs/capture")
    def capture_job(payload: CaptureFormState) -> dict[str, object]:
        return _submit(
            "capture",
            payload,
            mode=payload.model_preset or payload.model_id,
            label=payload.prompt_label,
            artifact_root=payload.output_dir,
        )

    @app.post("/api/jobs/paper-validate")
    def paper_job(payload: PaperValidateSpec) -> dict[str, object]:
        return _submit(
            "paper-validate",
            payload,
            mode=payload.variant,
            label=payload.variant,
            artifact_root=payload.output_dir
            or ("artifacts/paper_baseline/qwen_captured" if payload.variant == "captured_qwen" else "artifacts/paper_baseline"),
        )

    @app.post("/api/jobs/matrix-validate")
    def matrix_job(payload: MatrixRunSpec) -> dict[str, object]:
        return _submit(
            "matrix-validate",
            payload,
            mode="qwen_3060_matrix",
            label="qwen_3060_matrix",
            artifact_root=payload.output_dir,
        )

    @app.post("/api/jobs/runtime-eval")
    def runtime_job(payload: RuntimeEvalSpec) -> dict[str, object]:
        return _submit(
            "runtime-eval",
            payload,
            mode=payload.mode,
            label=payload.mode,
            artifact_root=payload.output_dir,
        )

    @app.post("/api/jobs/export-report")
    def export_job(payload: ExportReportSpec) -> dict[str, object]:
        return _submit(
            "export-report",
            payload,
            mode="export-report",
            label="export-report",
            artifact_root=payload.matrix_dir,
        )

    @app.post("/api/jobs/export-online-report")
    def export_online_job(payload: ExportOnlineReportSpec) -> dict[str, object]:
        return _submit(
            "export-online-report",
            payload,
            mode="online-eval-report",
            label="online-eval-report",
            artifact_root=payload.output_dir,
        )

    @app.post("/api/jobs/package-gguf")
    def package_job(payload: PackageSpec) -> dict[str, object]:
        return _submit(
            "package-gguf",
            payload,
            mode=payload.default_profile,
            label=Path(payload.output_gguf).name,
            artifact_root=str(Path(payload.output_gguf).parent),
        )

    @app.post("/api/jobs/serve-llama")
    def serve_llama_job(payload: ServeSpec) -> dict[str, object]:
        return _submit(
            "serve-llama",
            payload,
            mode=payload.model_alias,
            label=f"llama:{payload.port}",
            artifact_root=str(resolved.jobs_root),
        )

    @app.post("/api/jobs/serve-hypura")
    def serve_hypura_job(payload: ServeSpec) -> dict[str, object]:
        return _submit(
            "serve-hypura",
            payload,
            mode=payload.turboquant_mode,
            label=f"hypura:{payload.port}",
            artifact_root=str(resolved.jobs_root),
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, object]:
        try:
            return manager.cancel(job_id).model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    @app.get("/studio", include_in_schema=False)
    @app.get("/studio/{path:path}", include_in_schema=False)
    def studio_index(path: str = ""):
        if resolved.frontend_index.exists():
            return FileResponse(resolved.frontend_index)
        return JSONResponse(
            status_code=503,
            content={
                "detail": "TurboQuant Studio frontend has not been built yet.",
                "expected_index": str(resolved.frontend_index),
            },
        )

    return app
