import { useMemo, useState } from "react";

import { Field, Panel } from "../components/chrome";
import { WorkflowActions } from "../components/WorkflowActions";

export function OfflineTab({
  onSubmitJob,
}: {
  onSubmitJob: (endpoint: string, payload: Record<string, unknown>, dryRun: boolean) => Promise<void>;
}): JSX.Element {
  const [kvDir, setKvDir] = useState("artifacts/kv_rtx3060_qwen9b");
  const [bits, setBits] = useState("3,3.5,4");
  const [trials, setTrials] = useState("3");
  const [maxLayers, setMaxLayers] = useState("2");
  const [outputDir, setOutputDir] = useState("artifacts/qwen_3060_matrix");
  const [validationMessage, setValidationMessage] = useState("Matrix validation keeps exact, exact-score, and estimated-score separate.");

  const paperPayload = useMemo(
    () => ({
      variant: "captured_qwen",
      kv_dir: kvDir,
      trials: Number(trials),
      max_layers: Number(maxLayers),
      bits,
    }),
    [bits, kvDir, maxLayers, trials],
  );

  const matrixPayload = useMemo(
    () => ({
      kv_dir: kvDir,
      bits,
      trials: Number(trials),
      max_layers: Number(maxLayers),
      output_dir: outputDir,
    }),
    [bits, kvDir, maxLayers, outputDir, trials],
  );

  return (
    <div className="panel-grid">
      <Panel title="Paper baseline" subtitle="Offline baseline for captured Qwen artifacts.">
        <div className="form-grid">
          <Field label="KV directory">
            <input value={kvDir} onChange={(event) => setKvDir(event.target.value)} />
          </Field>
          <Field label="Bits">
            <input value={bits} onChange={(event) => setBits(event.target.value)} />
          </Field>
          <Field label="Trials">
            <input value={trials} onChange={(event) => setTrials(event.target.value)} />
          </Field>
          <Field label="Max layers">
            <input value={maxLayers} onChange={(event) => setMaxLayers(event.target.value)} />
          </Field>
        </div>
        <WorkflowActions
          validationMessage={validationMessage}
          onValidate={() => setValidationMessage(kvDir ? "Baseline spec is locally valid." : "KV directory is required.")}
          onPreview={() => void onSubmitJob("/api/jobs/paper-validate", paperPayload, true)}
          onRun={() => void onSubmitJob("/api/jobs/paper-validate", paperPayload, false)}
        />
      </Panel>

      <Panel title="Matrix run" subtitle="Reduced real matrix execution on the 3060 workflow.">
        <div className="form-grid">
          <Field label="Output root">
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
          </Field>
          <Field label="Stage separation" helper="UI keeps Stage 1/2 and metric families visually separate.">
            <input value="Stage 1 + Stage 2 preserved" readOnly />
          </Field>
        </div>
        <WorkflowActions
          validationMessage="Ready to preview reduced matrix validation."
          onValidate={() => setValidationMessage(outputDir ? "Matrix spec is locally valid." : "Output directory is required.")}
          onPreview={() => void onSubmitJob("/api/jobs/matrix-validate", matrixPayload, true)}
          onRun={() => void onSubmitJob("/api/jobs/matrix-validate", matrixPayload, false)}
        />
        <div className="image-grid">
          <img alt="Attention tradeoffs" src="/artifacts/qwen_3060_matrix/plots/qwen_3060_matrix_attention.png" />
          <img alt="Runtime tradeoffs" src="/artifacts/qwen_3060_matrix/plots/qwen_3060_matrix_runtime.png" />
        </div>
      </Panel>
    </div>
  );
}
