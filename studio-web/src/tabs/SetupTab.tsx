import type { ArtifactRecord, ArtifactSummary, SetupSnapshot } from "../types";
import { Panel, StatusPill } from "../components/chrome";

function renderNode(node: ArtifactRecord): JSX.Element {
  return (
    <li key={node.absolute_path}>
      <div className="artifact-line">
        <span>{node.relative_path}</span>
        <span>{node.kind}</span>
      </div>
      {node.children?.length ? <ul>{node.children.map((child) => renderNode(child))}</ul> : null}
    </li>
  );
}

export function SetupTab({
  setup,
  summary,
  artifactTree,
}: {
  setup: SetupSnapshot | null;
  summary: ArtifactSummary | null;
  artifactTree: ArtifactRecord | null;
}): JSX.Element {
  if (!setup) {
    return <p className="empty-copy">Loading setup snapshot...</p>;
  }

  return (
    <div className="panel-grid">
      <Panel title="Environment health" subtitle="Python, uv, CUDA, torch, and repo contract status.">
        <div className="card-grid">
          {setup.checks.map((check) => (
            <article key={check.name} className="status-card">
              <div className="status-card-head">
                <span>{check.name}</span>
                <StatusPill tone={check.ok ? "ok" : "bad"}>{check.ok ? "ok" : "needs attention"}</StatusPill>
              </div>
              <p>{check.detail}</p>
            </article>
          ))}
        </div>
      </Panel>

      <Panel title="Model presets" subtitle="Capture presets exposed through the Studio shell.">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Preset</th>
                <th>Model ID</th>
                <th>Lane</th>
                <th>Weight load</th>
                <th>Dtype</th>
              </tr>
            </thead>
            <tbody>
              {setup.model_presets.map((preset) => (
                <tr key={preset.name}>
                  <td>{preset.name}</td>
                  <td>{preset.model_id}</td>
                  <td>{preset.lane_name}</td>
                  <td>{preset.default_weight_load}</td>
                  <td>{preset.default_dtype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Prompt panel" subtitle="Reusable local prompts for capture reproducibility.">
        <div className="stack-list">
          {setup.prompt_panel.map((entry) => (
            <article key={entry.label} className="stack-card">
              <strong>{entry.label}</strong>
              <p>{entry.prompt}</p>
            </article>
          ))}
        </div>
      </Panel>

      <Panel title="Artifacts" subtitle="Canonical roots stay in place; Studio only indexes them.">
        <div className="details-grid">
          <div>
            <span className="detail-label">Artifact root</span>
            <strong>{summary?.artifact_root ?? setup.artifact_root}</strong>
          </div>
          <div>
            <span className="detail-label">Files</span>
            <strong>{summary?.existing_files ?? 0}</strong>
          </div>
          <div>
            <span className="detail-label">Directories</span>
            <strong>{summary?.existing_directories ?? 0}</strong>
          </div>
        </div>
        {artifactTree ? (
          <div className="artifact-tree">
            <ul>{renderNode(artifactTree)}</ul>
          </div>
        ) : (
          <p className="empty-copy">Artifact tree not loaded yet.</p>
        )}
      </Panel>
    </div>
  );
}
