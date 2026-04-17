import { useEffect, useMemo, useState } from "react";

import { getArtifactSummary, getArtifactTree, getRun, getRuns, getSetupSnapshot, submitJob, cancelRun } from "./api";
import { HistoryPanel } from "./components/HistoryPanel";
import { Panel, StatusPill } from "./components/chrome";
import { ToolDrawer } from "./components/ToolDrawer";
import { parseCsv } from "./csv";
import { CaptureTab } from "./tabs/CaptureTab";
import { CompareTab } from "./tabs/CompareTab";
import { OfflineTab } from "./tabs/OfflineTab";
import { PackageTab } from "./tabs/PackageTab";
import { RuntimeTab } from "./tabs/RuntimeTab";
import { ServeTab } from "./tabs/ServeTab";
import { SetupTab } from "./tabs/SetupTab";
import type { ArtifactRecord, ArtifactSummary, JobStatus, SetupSnapshot } from "./types";

type TabKey = "setup" | "capture" | "offline" | "compare" | "runtime" | "package" | "serve";

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: "setup", label: "Setup" },
  { key: "capture", label: "Capture" },
  { key: "offline", label: "Offline Validate" },
  { key: "compare", label: "Compare" },
  { key: "runtime", label: "Runtime Eval" },
  { key: "package", label: "Package & Export" },
  { key: "serve", label: "Serve" },
];

export function App(): JSX.Element {
  const [activeTab, setActiveTab] = useState<TabKey>("setup");
  const [setup, setSetup] = useState<SetupSnapshot | null>(null);
  const [summary, setSummary] = useState<ArtifactSummary | null>(null);
  const [artifactTree, setArtifactTree] = useState<ArtifactRecord | null>(null);
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobStatus | null>(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [workflowFilter, setWorkflowFilter] = useState("all");
  const [compareRows, setCompareRows] = useState<Record<string, string>[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([getSetupSnapshot(), getArtifactSummary(), getArtifactTree(".", 2), getRuns()])
      .then(([setupData, summaryData, treeData, runData]) => {
        setSetup(setupData);
        setSummary(summaryData);
        setArtifactTree(treeData);
        setJobs(runData);
        if (!selectedJobId && runData[0]) {
          setSelectedJobId(runData[0].id);
        }
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, [selectedJobId]);

  useEffect(() => {
    const handle = window.setInterval(() => {
      void getRuns().then((runData) => setJobs(runData)).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(handle);
  }, []);

  useEffect(() => {
    if (!selectedJobId) {
      setSelectedJob(null);
      return;
    }
    void getRun(selectedJobId)
      .then((job) => setSelectedJob(job))
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, [selectedJobId, jobs]);

  useEffect(() => {
    void fetch("/artifacts/qwen_3060_matrix/metrics/qwen_3060_matrix_summary.csv")
      .then((response) => (response.ok ? response.text() : ""))
      .then((text) => setCompareRows(parseCsv(text)))
      .catch(() => setCompareRows([]));
  }, []);

  const visibleJobs = useMemo(
    () =>
      jobs.filter((job) => (statusFilter === "all" ? true : job.status === statusFilter)).filter((job) =>
        workflowFilter === "all" ? true : job.kind === workflowFilter,
      ),
    [jobs, statusFilter, workflowFilter],
  );

  async function handleSubmitJob(endpoint: string, payload: Record<string, unknown>, dryRun: boolean): Promise<void> {
    setError(null);
    const job = await submitJob(endpoint, payload as { dry_run: boolean }, dryRun);
    setJobs((current) => [job, ...current]);
    setSelectedJobId(job.id);
  }

  async function handleCancel(jobId: string): Promise<void> {
    const job = await cancelRun(jobId);
    setSelectedJob(job);
    setJobs((current) => current.map((item) => (item.id === job.id ? job : item)));
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">TurboQuant Studio</p>
          <h1>Offline-first KV compression workbench</h1>
        </div>
        <div className="topbar-actions">
          <StatusPill tone={setup?.repo_contract_ok ? "ok" : "warn"}>{setup?.repo_contract_ok ? "repo contract ok" : "check repo contract"}</StatusPill>
          <StatusPill tone={setup?.cuda_available ? "ok" : "warn"}>{setup?.gpu_names[0] ?? "cuda unavailable"}</StatusPill>
        </div>
      </header>

      <Panel title="Workspace metadata" subtitle="Reproducibility strip for the current machine and checkout.">
        <div className="meta-strip">
          <span>Python {setup?.python_version ?? "-"}</span>
          <span>{setup?.uv_version ?? "uv missing"}</span>
          <span>{setup?.node_version ?? "node missing"}</span>
          <span>CUDA {setup?.torch_cuda ?? "-"}</span>
          <span>{setup?.active_artifact_root ?? "-"}</span>
        </div>
      </Panel>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="workspace-grid">
        <main className="workspace-main">
          <nav className="tabbar" aria-label="Studio tabs">
            {TABS.map((tab) => (
              <button key={tab.key} type="button" className={`tab ${activeTab === tab.key ? "active" : ""}`} onClick={() => setActiveTab(tab.key)}>
                {tab.label}
              </button>
            ))}
          </nav>

          {activeTab === "setup" ? <SetupTab setup={setup} summary={summary} artifactTree={artifactTree} /> : null}
          {activeTab === "capture" ? <CaptureTab setup={setup} onSubmitJob={handleSubmitJob} /> : null}
          {activeTab === "offline" ? <OfflineTab onSubmitJob={handleSubmitJob} /> : null}
          {activeTab === "compare" ? <CompareTab compareRows={compareRows} /> : null}
          {activeTab === "runtime" ? <RuntimeTab setup={setup} onSubmitJob={handleSubmitJob} /> : null}
          {activeTab === "package" ? <PackageTab onSubmitJob={handleSubmitJob} /> : null}
          {activeTab === "serve" ? <ServeTab onSubmitJob={handleSubmitJob} /> : null}
        </main>

        <ToolDrawer job={selectedJob} onCancel={handleCancel} />
      </div>

      <HistoryPanel
        jobs={visibleJobs}
        selectedJobId={selectedJobId}
        statusFilter={statusFilter}
        workflowFilter={workflowFilter}
        onStatusFilterChange={setStatusFilter}
        onWorkflowFilterChange={setWorkflowFilter}
        onSelect={setSelectedJobId}
      />
    </div>
  );
}
