import type {
  ArtifactRecord,
  ArtifactSummary,
  JobStatus,
  SetupSnapshot,
} from "./types";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return (await response.json()) as T;
}

export function getSetupSnapshot(): Promise<SetupSnapshot> {
  return fetchJson("/api/setup");
}

export function getArtifactSummary(): Promise<ArtifactSummary> {
  return fetchJson("/api/artifacts/summary");
}

export function getArtifactTree(relativePath = ".", maxDepth = 3): Promise<ArtifactRecord> {
  const params = new URLSearchParams({ relative_path: relativePath, max_depth: String(maxDepth) });
  return fetchJson(`/api/artifacts/tree?${params.toString()}`);
}

export function getRuns(): Promise<JobStatus[]> {
  return fetchJson("/api/runs");
}

export function getRun(jobId: string): Promise<JobStatus> {
  return fetchJson(`/api/runs/${jobId}`);
}

export function cancelRun(jobId: string): Promise<JobStatus> {
  return fetchJson(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

export function submitJob<T extends Record<string, unknown>>(
  endpoint: string,
  payload: T,
  dryRun: boolean,
): Promise<JobStatus> {
  return fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, dry_run: dryRun }),
  });
}
