import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

const FETCH_RESPONSES: Record<string, object> = {
  "/api/setup": {
    timestamp_utc: "2026-04-18T00:00:00Z",
    repo_root: "H:/repo",
    artifact_root: "H:/repo/artifacts",
    active_artifact_root: "H:/repo/artifacts",
    python_version: "3.12.9",
    uv_version: "uv 0.6.6",
    node_version: "v22.22.2",
    npm_version: "11.11.0",
    target_cuda: "cu128",
    torch_version: "2.8.0",
    torch_cuda: "12.8",
    cuda_available: true,
    gpu_names: ["RTX 3060"],
    repo_contract_ok: true,
    repo_contract_errors: [],
    vendored_runtime_ready: true,
    checks: [{ name: "python_3_12", ok: true, detail: "3.12.9" }],
    model_presets: [
      {
        name: "qwen35_9b_12gb",
        model_id: "Qwen/Qwen3.5-9B",
        lane_name: "rtx3060_desktop_12gb",
        model_source: "hf",
        default_weight_load: "4bit",
        default_dtype: "float16",
      },
    ],
    prompt_panel: [{ label: "baseline", prompt: "hello" }],
    capture_presets: ["qwen35_9b_12gb"],
    runtime_profiles: ["turboquant_enabled_audit"],
    compare_modes: ["exact", "key_only_block_so8_triality_vector"],
    paper_validate_variants: ["captured_qwen", "synthetic"],
  },
  "/api/artifacts/summary": {
    artifact_root: "H:/repo/artifacts",
    existing_files: 12,
    existing_directories: 4,
    known_paths: {},
  },
  "/api/artifacts/tree?relative_path=.&max_depth=2": {
    relative_path: ".",
    absolute_path: "H:/repo/artifacts",
    kind: "directory",
    children: [],
  },
  "/api/runs": [],
};

describe("App", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = FETCH_RESPONSES[url];
      if (payload !== undefined) {
        return new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/artifacts/qwen_3060_matrix/metrics/qwen_3060_matrix_summary.csv")) {
        return new Response("mode,bits\nexact,4\n", { status: 200 });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    }) as typeof fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("renders the main Studio chrome", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText("Offline-first KV compression workbench")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Capture" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Runtime Eval" })).toBeInTheDocument();
    expect(screen.getByText("Tool Outputs")).toBeInTheDocument();
    expect(screen.getByText("Workspace metadata")).toBeInTheDocument();
  });
});
