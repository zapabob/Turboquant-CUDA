import { useMemo, useState } from "react";

import { Field, Panel } from "../components/chrome";
import { WorkflowActions } from "../components/WorkflowActions";

export function PackageTab({
  onSubmitJob,
}: {
  onSubmitJob: (endpoint: string, payload: Record<string, unknown>, dryRun: boolean) => Promise<void>;
}): JSX.Element {
  const [matrixDir, setMatrixDir] = useState("artifacts/qwen_3060_matrix");
  const [outputDir, setOutputDir] = useState("artifacts/online_eval_report");
  const [inputGguf, setInputGguf] = useState("artifacts/models/qwen-runtime.gguf");
  const [outputGguf, setOutputGguf] = useState("artifacts/models/qwen-runtime-turboquant.gguf");
  const [validationMessage, setValidationMessage] = useState("Exports preserve the existing artifact layout.");

  const exportPayload = useMemo(() => ({ matrix_dir: matrixDir }), [matrixDir]);
  const onlinePayload = useMemo(() => ({ output_dir: outputDir }), [outputDir]);
  const packagePayload = useMemo(
    () => ({
      input_gguf: inputGguf,
      output_gguf: outputGguf,
      profiles: "paper,so8_triality_vector",
      default_profile: "exact",
      hypura_compatible_profile: "auto",
    }),
    [inputGguf, outputGguf],
  );

  return (
    <div className="panel-grid">
      <Panel title="Report export" subtitle="Offline and online report generation on canonical directories.">
        <div className="form-grid">
          <Field label="Matrix directory">
            <input value={matrixDir} onChange={(event) => setMatrixDir(event.target.value)} />
          </Field>
          <Field label="Online report output">
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
          </Field>
        </div>
        <WorkflowActions
          validationMessage={validationMessage}
          onValidate={() => setValidationMessage(matrixDir ? "Export paths look valid." : "Matrix directory is required.")}
          onPreview={() => void onSubmitJob("/api/jobs/export-report", exportPayload, true)}
          onRun={() => void onSubmitJob("/api/jobs/export-report", exportPayload, false)}
        />
        <WorkflowActions
          validationMessage="Online export merges runtime and replay summaries."
          onValidate={() => setValidationMessage(outputDir ? "Online export path looks valid." : "Output directory is required.")}
          onPreview={() => void onSubmitJob("/api/jobs/export-online-report", onlinePayload, true)}
          onRun={() => void onSubmitJob("/api/jobs/export-online-report", onlinePayload, false)}
        />
      </Panel>

      <Panel title="GGUF packaging" subtitle="Profile selection and Hypura-compatible packaging.">
        <div className="form-grid">
          <Field label="Input GGUF">
            <input value={inputGguf} onChange={(event) => setInputGguf(event.target.value)} />
          </Field>
          <Field label="Output GGUF">
            <input value={outputGguf} onChange={(event) => setOutputGguf(event.target.value)} />
          </Field>
        </div>
        <WorkflowActions
          validationMessage="Packaged GGUF keeps embedded metadata for runtime consumers."
          onValidate={() => setValidationMessage(inputGguf ? "Packaging spec is locally valid." : "Input GGUF is required.")}
          onPreview={() => void onSubmitJob("/api/jobs/package-gguf", packagePayload, true)}
          onRun={() => void onSubmitJob("/api/jobs/package-gguf", packagePayload, false)}
        />
      </Panel>
    </div>
  );
}
