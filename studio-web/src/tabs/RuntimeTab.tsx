import { useMemo, useState } from "react";

import type { SetupSnapshot } from "../types";
import { Field, Panel } from "../components/chrome";
import { WorkflowActions } from "../components/WorkflowActions";

export function RuntimeTab({
  setup,
  onSubmitJob,
}: {
  setup: SetupSnapshot | null;
  onSubmitJob: (endpoint: string, payload: Record<string, unknown>, dryRun: boolean) => Promise<void>;
}): JSX.Element {
  const [mode, setMode] = useState("exact");
  const [modelPath, setModelPath] = useState("artifacts/qwen-runtime/model.gguf");
  const [outputDir, setOutputDir] = useState("artifacts/runtime_eval");
  const [serverBin, setServerBin] = useState("zapabob/llama.cpp/build/bin/Release/llama-server.exe");
  const [llamaBenchBin, setLlamaBenchBin] = useState("zapabob/llama.cpp/build/bin/Release/llama-bench.exe");
  const [runtimeProfile, setRuntimeProfile] = useState(setup?.runtime_profiles[0] ?? "");
  const [validationMessage, setValidationMessage] = useState("Runtime evaluation stays dry-run previewable before execution.");

  const payload = useMemo(
    () => ({
      mode,
      model_path: modelPath,
      server_bin: serverBin,
      llama_bench_bin: llamaBenchBin,
      runtime_profile: runtimeProfile || null,
      output_dir: outputDir,
    }),
    [llamaBenchBin, mode, modelPath, outputDir, runtimeProfile, serverBin],
  );

  return (
    <div className="panel-grid">
      <Panel title="Runtime evaluation" subtitle="Perplexity, bench, and local server settings in one form.">
        <div className="form-grid">
          <Field label="Mode">
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              {(setup?.compare_modes ?? []).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Model path">
            <input value={modelPath} onChange={(event) => setModelPath(event.target.value)} />
          </Field>
          <Field label="llama-server bin">
            <input value={serverBin} onChange={(event) => setServerBin(event.target.value)} />
          </Field>
          <Field label="llama-bench bin">
            <input value={llamaBenchBin} onChange={(event) => setLlamaBenchBin(event.target.value)} />
          </Field>
          <Field label="Runtime profile">
            <select value={runtimeProfile} onChange={(event) => setRuntimeProfile(event.target.value)}>
              <option value="">None</option>
              {(setup?.runtime_profiles ?? []).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Output root">
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
          </Field>
        </div>
        <WorkflowActions
          validationMessage={validationMessage}
          onValidate={() => setValidationMessage(modelPath ? "Runtime spec is locally valid." : "Model path is required.")}
          onPreview={() => void onSubmitJob("/api/jobs/runtime-eval", payload, true)}
          onRun={() => void onSubmitJob("/api/jobs/runtime-eval", payload, false)}
        />
      </Panel>
    </div>
  );
}
