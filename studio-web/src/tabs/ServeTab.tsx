import { useMemo, useState } from "react";

import { Field, Panel } from "../components/chrome";
import { WorkflowActions } from "../components/WorkflowActions";

export function ServeTab({
  onSubmitJob,
}: {
  onSubmitJob: (endpoint: string, payload: Record<string, unknown>, dryRun: boolean) => Promise<void>;
}): JSX.Element {
  const [serverBin, setServerBin] = useState("vendor/llama.cpp/build/bin/Release/llama-server.exe");
  const [modelPath, setModelPath] = useState("artifacts/models/qwen-runtime-turboquant.gguf");
  const [ggufPath, setGgufPath] = useState("artifacts/models/qwen-runtime-turboquant.gguf");
  const [port, setPort] = useState("8080");
  const [validationMessage, setValidationMessage] = useState("Serve controls wrap existing local launch flows only.");

  const llamaPayload = useMemo(
    () => ({
      server_bin: serverBin,
      model_path: modelPath,
      port: Number(port),
      model_alias: "qwen-runtime",
    }),
    [modelPath, port, serverBin],
  );

  const hypuraPayload = useMemo(
    () => ({
      gguf: ggufPath,
      port: Number(port),
      turboquant_mode: "gguf-auto",
    }),
    [ggufPath, port],
  );

  return (
    <div className="panel-grid">
      <Panel title="Serve llama.cpp" subtitle="Launch or preview llama-server locally from the Studio queue.">
        <div className="form-grid">
          <Field label="Server bin">
            <input value={serverBin} onChange={(event) => setServerBin(event.target.value)} />
          </Field>
          <Field label="Model path">
            <input value={modelPath} onChange={(event) => setModelPath(event.target.value)} />
          </Field>
          <Field label="Port">
            <input value={port} onChange={(event) => setPort(event.target.value)} />
          </Field>
        </div>
        <WorkflowActions
          validationMessage={validationMessage}
          onValidate={() => setValidationMessage(serverBin && modelPath ? "llama.cpp serve spec is locally valid." : "Server bin and model path are required.")}
          onPreview={() => void onSubmitJob("/api/jobs/serve-llama", llamaPayload, true)}
          onRun={() => void onSubmitJob("/api/jobs/serve-llama", llamaPayload, false)}
        />
      </Panel>

      <Panel title="Serve Hypura" subtitle="Wrapper around the current local Hypura proxy launch flow.">
        <div className="form-grid">
          <Field label="GGUF path">
            <input value={ggufPath} onChange={(event) => setGgufPath(event.target.value)} />
          </Field>
        </div>
        <WorkflowActions
          validationMessage="Hypura launch reuses embedded bridge metadata when present."
          onValidate={() => setValidationMessage(ggufPath ? "Hypura spec is locally valid." : "GGUF path is required.")}
          onPreview={() => void onSubmitJob("/api/jobs/serve-hypura", hypuraPayload, true)}
          onRun={() => void onSubmitJob("/api/jobs/serve-hypura", hypuraPayload, false)}
        />
      </Panel>
    </div>
  );
}
