from __future__ import annotations

from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient

from turboquant.studio_api.app import create_app
from turboquant.studio_api.config import REPO_ROOT, StudioSettings
from turboquant.studio_api.models import CaptureFormState, RuntimeEvalSpec
from turboquant.studio_api.tasks import build_capture_preview, build_runtime_preview


def _settings(tmp_path: Path) -> StudioSettings:
    artifact_root = tmp_path / "artifacts"
    studio_root = artifact_root / "studio"
    jobs_root = studio_root / "jobs"
    frontend_dist = tmp_path / "frontend-dist"
    frontend_dist.mkdir(parents=True, exist_ok=True)
    return StudioSettings(
        repo_root=REPO_ROOT,
        artifact_root=artifact_root,
        studio_root=studio_root,
        jobs_root=jobs_root,
        db_path=studio_root / "studio.db",
        frontend_dist=frontend_dist,
        frontend_index=frontend_dist / "index.html",
    )


def _wait_for_status(client: TestClient, job_id: str, *, timeout_sec: float = 5.0) -> dict[str, object]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        payload = client.get(f"/api/runs/{job_id}").json()
        if payload["status"] in {"completed", "failed", "canceled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


def test_artifact_summary_and_tree_use_canonical_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    metrics_dir = settings.artifact_root / "qwen_3060_matrix" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "summary.csv").write_text("mode,bits\nexact,4\n", encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        summary = client.get("/api/artifacts/summary").json()
        tree = client.get("/api/artifacts/tree", params={"relative_path": ".", "max_depth": 3}).json()

    assert health["ok"] is True
    assert Path(summary["artifact_root"]) == settings.artifact_root
    assert summary["existing_files"] >= 1
    assert tree["kind"] == "directory"
    assert any(child["relative_path"] == "qwen_3060_matrix" for child in tree["children"])


def test_setup_snapshot_endpoint_returns_workbench_defaults(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        payload = client.get("/api/setup").json()

    assert payload["artifact_root"] == str(settings.artifact_root)
    assert "key_only_block_so8_triality_vector" in payload["compare_modes"]
    assert "captured_qwen" in payload["paper_validate_variants"]


def test_studio_route_serves_built_frontend_when_present(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.frontend_dist.mkdir(parents=True, exist_ok=True)
    settings.frontend_index.write_text("<!doctype html><title>studio</title>", encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        response = client.get("/studio")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "studio" in response.text


def test_capture_dry_run_job_persists_preview_result(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/jobs/capture",
            json={
                "model_preset": "qwen35_9b_12gb",
                "prompt_label": "smoke",
                "prompt": "TurboQuant Studio dry run",
                "output_dir": "artifacts/kv",
                "dry_run": True,
            },
        )
        assert response.status_code == 200
        job = response.json()
        final = _wait_for_status(client, job["id"])

    assert final["status"] == "completed"
    assert final["result"]["command"][3] == "scripts/capture_qwen_kv.py"
    assert len(final["result"]["capture_targets"]) == 1


def test_runtime_eval_dry_run_returns_manifest_preview(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/jobs/runtime-eval",
            json={
                "mode": "exact",
                "model_path": "artifacts/models/qwen.gguf",
                "server_bin": "zapabob/llama.cpp/build/bin/Release/llama-server.exe",
                "llama_bench_bin": "zapabob/llama.cpp/build/bin/Release/llama-bench.exe",
                "output_dir": "artifacts/runtime_eval",
                "dry_run": True,
            },
        )
        final = _wait_for_status(client, response.json()["id"])

    assert final["status"] == "completed"
    assert "--dry-run" in final["result"]["command"]
    assert final["result"]["expected_outputs"][0].endswith("runtime_eval_commands.json")


def test_job_queue_allows_canceling_second_queued_job(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_capture(_spec) -> dict[str, object]:
        started.set()
        release.wait(timeout=2.0)
        return {"exit_code": 0, "expected_outputs": []}

    monkeypatch.setattr("turboquant.studio_api.jobs.capture_task", slow_capture)

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/api/jobs/capture",
            json={"prompt_label": "first", "prompt": "one", "dry_run": False},
        ).json()
        second = client.post(
            "/api/jobs/capture",
            json={"prompt_label": "second", "prompt": "two", "dry_run": False},
        ).json()
        assert started.wait(timeout=2.0)

        canceled = client.post(f"/api/jobs/{second['id']}/cancel").json()
        assert canceled["status"] == "canceled"

        release.set()
        first_final = _wait_for_status(client, first["id"])
        second_final = client.get(f"/api/runs/{second['id']}").json()

    assert first_final["status"] == "completed"
    assert second_final["status"] == "canceled"


def test_preview_builders_preserve_cli_contract() -> None:
    capture_preview = build_capture_preview(
        CaptureFormState(
            model_preset="qwen35_9b_12gb",
            prompt_label="custom",
            prompt="hello",
            output_dir="artifacts/kv",
        )
    )
    runtime_preview = build_runtime_preview(
        RuntimeEvalSpec(
            mode="exact",
            model_path="artifacts/models/qwen.gguf",
            server_bin="zapabob/llama.cpp/build/bin/Release/llama-server.exe",
            llama_bench_bin="zapabob/llama.cpp/build/bin/Release/llama-bench.exe",
            output_dir="artifacts/runtime_eval",
        )
    )

    assert capture_preview["command"][:4] == ["uv", "run", "python", "scripts/capture_qwen_kv.py"]
    assert runtime_preview["command"][:4] == ["uv", "run", "python", "scripts/eval_runtime_qwen.py"]
