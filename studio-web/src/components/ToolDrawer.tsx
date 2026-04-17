import type { JobStatus } from "../types";
import { GhostButton, Panel, StatusPill } from "./chrome";

function statusTone(status: JobStatus["status"]): "ok" | "bad" | "warn" | "run" | "idle" {
  if (status === "completed") return "ok";
  if (status === "failed") return "bad";
  if (status === "running") return "run";
  if (status === "canceled" || status === "cancel_requested") return "warn";
  return "idle";
}

export function ToolDrawer({
  job,
  onCancel,
}: {
  job: JobStatus | null;
  onCancel: (jobId: string) => Promise<void>;
}): JSX.Element {
  return (
    <aside className="tool-drawer">
      <Panel
        title="Tool Outputs"
        subtitle="Live job logs, exact commands, and reproducibility metadata."
        actions={
          job && job.status === "running" ? (
            <GhostButton onClick={() => void onCancel(job.id)}>Stop job</GhostButton>
          ) : null
        }
      >
        {job ? (
          <div className="stack-list">
            <div className="details-grid">
              <div>
                <span className="detail-label">Workflow</span>
                <strong>{job.kind}</strong>
              </div>
              <div>
                <span className="detail-label">State</span>
                <StatusPill tone={statusTone(job.status)}>{job.status}</StatusPill>
              </div>
              <div>
                <span className="detail-label">Mode</span>
                <strong>{job.mode ?? "-"}</strong>
              </div>
              <div>
                <span className="detail-label">Artifact root</span>
                <strong>{job.artifact_root ?? "-"}</strong>
              </div>
            </div>
            <div className="stack-card">
              <span className="detail-label">Resolved spec</span>
              <pre className="json-preview">{JSON.stringify(job.spec, null, 2)}</pre>
            </div>
            <div className="stack-card">
              <span className="detail-label">Result</span>
              <pre className="json-preview">{JSON.stringify(job.result ?? {}, null, 2)}</pre>
            </div>
            <div className="stack-card">
              <span className="detail-label">Logs</span>
              <pre className="log-tail">{job.log_tail ?? "No log output yet."}</pre>
            </div>
          </div>
        ) : (
          <p className="empty-copy">Select a run to inspect command previews, output paths, and logs.</p>
        )}
      </Panel>
    </aside>
  );
}
