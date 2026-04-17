import type { JobStatus, JobState } from "../types";
import { StatusPill } from "./chrome";

const STATE_TONE: Record<JobState, "ok" | "bad" | "warn" | "run" | "idle"> = {
  queued: "idle",
  running: "run",
  completed: "ok",
  failed: "bad",
  canceled: "warn",
  cancel_requested: "warn",
};

export function HistoryPanel({
  jobs,
  selectedJobId,
  statusFilter,
  workflowFilter,
  onStatusFilterChange,
  onWorkflowFilterChange,
  onSelect,
}: {
  jobs: JobStatus[];
  selectedJobId: string | null;
  statusFilter: string;
  workflowFilter: string;
  onStatusFilterChange: (value: string) => void;
  onWorkflowFilterChange: (value: string) => void;
  onSelect: (jobId: string) => void;
}): JSX.Element {
  const workflows = Array.from(new Set(jobs.map((job) => job.kind))).sort();
  return (
    <aside className="history-panel">
      <div className="history-toolbar">
        <div>
          <p className="eyebrow">Run History</p>
          <h2>Workflow queue</h2>
        </div>
        <div className="history-filters">
          <select value={workflowFilter} onChange={(event) => onWorkflowFilterChange(event.target.value)}>
            <option value="all">All workflows</option>
            {workflows.map((workflow) => (
              <option key={workflow} value={workflow}>
                {workflow}
              </option>
            ))}
          </select>
          <select value={statusFilter} onChange={(event) => onStatusFilterChange(event.target.value)}>
            <option value="all">All states</option>
            <option value="queued">Queued</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="canceled">Canceled</option>
          </select>
        </div>
      </div>
      <div className="history-list">
        {jobs.map((job) => (
          <button
            key={job.id}
            type="button"
            className={`history-item ${selectedJobId === job.id ? "selected" : ""}`}
            onClick={() => onSelect(job.id)}
          >
            <div>
              <p className="history-kind">{job.kind}</p>
              <p className="history-meta">{job.label ?? job.mode ?? "local run"}</p>
              <p className="history-meta">{new Date(job.created_at).toLocaleString()}</p>
            </div>
            <div className="history-right">
              <StatusPill tone={STATE_TONE[job.status]}>{job.status}</StatusPill>
              <span className="history-id">{job.id.slice(0, 8)}</span>
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}
