import { useEffect, useMemo, useState } from "react";

import type { SetupSnapshot } from "../types";
import { ButtonRow, Field, GhostButton, Panel } from "../components/chrome";
import { WorkflowActions } from "../components/WorkflowActions";

async function sha256Hex(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((item) => item.toString(16).padStart(2, "0"))
    .join("");
}

function slugify(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export function CaptureTab({
  setup,
  onSubmitJob,
}: {
  setup: SetupSnapshot | null;
  onSubmitJob: (endpoint: string, payload: Record<string, unknown>, dryRun: boolean) => Promise<void>;
}): JSX.Element {
  const preset = setup?.model_presets[0];
  const [modelPreset, setModelPreset] = useState(preset?.name ?? "qwen35_9b_12gb");
  const [promptLabel, setPromptLabel] = useState(setup?.prompt_panel[0]?.label ?? "custom");
  const [prompt, setPrompt] = useState(setup?.prompt_panel[0]?.prompt ?? "");
  const [outputDir, setOutputDir] = useState("artifacts/kv");
  const [weightLoad, setWeightLoad] = useState(preset?.default_weight_load ?? "4bit");
  const [dtype, setDtype] = useState(preset?.default_dtype ?? "float16");
  const [maxLength, setMaxLength] = useState("96");
  const [promptHash, setPromptHash] = useState("");
  const [validationMessage, setValidationMessage] = useState("Fill out the capture spec and validate before running.");

  useEffect(() => {
    void sha256Hex(prompt).then((hash) => setPromptHash(hash.slice(0, 16)));
  }, [prompt]);

  const captureId = useMemo(() => `${slugify(promptLabel)}-${promptHash}`, [promptHash, promptLabel]);

  function payload(): Record<string, unknown> {
    return {
      model_preset: modelPreset,
      prompt_label: promptLabel,
      prompt,
      output_dir: outputDir,
      weight_load: weightLoad,
      dtype,
      max_length: Number(maxLength),
    };
  }

  return (
    <div className="panel-grid">
      <Panel title="Capture KV artifacts" subtitle="Form wrapper around the current capture workflow.">
        <div className="form-grid">
          <Field label="Model preset">
            <select value={modelPreset} onChange={(event) => setModelPreset(event.target.value)}>
              {(setup?.capture_presets ?? []).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Prompt label">
            <input value={promptLabel} onChange={(event) => setPromptLabel(event.target.value)} />
          </Field>
          <Field label="Output root">
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
          </Field>
          <Field label="Weight load">
            <select value={weightLoad} onChange={(event) => setWeightLoad(event.target.value)}>
              <option value="4bit">4bit</option>
              <option value="8bit">8bit</option>
              <option value="none">none</option>
            </select>
          </Field>
          <Field label="Dtype">
            <input value={dtype} onChange={(event) => setDtype(event.target.value)} />
          </Field>
          <Field label="Max length">
            <input value={maxLength} onChange={(event) => setMaxLength(event.target.value)} />
          </Field>
          <Field label="Prompt" helper="Prompt hash and capture slug are shown below.">
            <textarea rows={8} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          </Field>
        </div>
        <WorkflowActions
          validationMessage={validationMessage}
          onValidate={() =>
            setValidationMessage(prompt.trim() ? `Ready: ${captureId} with dtype ${dtype} and ${weightLoad}.` : "Prompt is required.")
          }
          onPreview={() => void onSubmitJob("/api/jobs/capture", payload(), true)}
          onRun={() => void onSubmitJob("/api/jobs/capture", payload(), false)}
        />
      </Panel>

      <Panel title="Reproducibility preview" subtitle="Prompt hash, capture ID, and explicit output location.">
        <div className="details-grid">
          <div>
            <span className="detail-label">Prompt hash</span>
            <strong>{promptHash || "-"}</strong>
          </div>
          <div>
            <span className="detail-label">Capture ID</span>
            <strong>{captureId}</strong>
          </div>
          <div>
            <span className="detail-label">Expected output</span>
            <strong>{`${outputDir}/${captureId}`}</strong>
          </div>
        </div>
        <ButtonRow>
          {(setup?.prompt_panel ?? []).slice(0, 4).map((entry) => (
            <GhostButton
              key={entry.label}
              type="button"
              onClick={() => {
                setPromptLabel(entry.label);
                setPrompt(entry.prompt);
              }}
            >
              Load {entry.label}
            </GhostButton>
          ))}
        </ButtonRow>
      </Panel>
    </div>
  );
}
