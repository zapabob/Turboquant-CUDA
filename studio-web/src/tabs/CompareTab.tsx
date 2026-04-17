import { MetricTable, type MetricRow } from "../components/MetricTable";
import { Panel } from "../components/chrome";

export function CompareTab({
  compareRows,
}: {
  compareRows: Record<string, string>[];
}): JSX.Element {
  const first = compareRows[0] ?? {};
  const reconstruction: MetricRow[] = [
    { label: "Mode", value: first.mode ?? "-" },
    { label: "Bits", value: first.bits ?? "-" },
    { label: "Reconstruction", value: first.reconstruction_error_mean ?? "-" },
    { label: "Inner-product bias", value: first.inner_product_bias_mean ?? "-" },
  ];
  const attention: MetricRow[] = [
    { label: "Hidden cosine", value: first.hidden_cosine_mean ?? "-" },
    { label: "Logit cosine", value: first.logit_cosine_mean ?? "-" },
    { label: "Attention MSE", value: first.attention_mse_mean ?? "-" },
  ];
  const runtime: MetricRow[] = [
    { label: "KV ratio", value: first.kv_bytes_ratio_mean ?? "-" },
    { label: "Prefill tok/s", value: first.prefill_tps_mean ?? "-" },
    { label: "Decode tok/s", value: first.decode_tps_mean ?? "-" },
  ];

  return (
    <div className="panel-grid">
      <Panel title="Mode comparison" subtitle="Metric families remain separated to match the research contract.">
        <div className="card-grid">
          <MetricTable title="Reconstruction metrics" rows={reconstruction} />
          <MetricTable title="Attention and logit metrics" rows={attention} />
          <MetricTable title="Runtime and memory" rows={runtime} />
        </div>
      </Panel>
      <Panel title="Compare modes" subtitle="The active matrix summary feeds this Pareto-style overview.">
        <div className="stack-list">
          {compareRows.slice(0, 12).map((row, index) => (
            <article key={`${row.mode}-${row.bits}-${index}`} className="stack-card">
              <strong>
                {row.mode} @ {row.bits} bits
              </strong>
              <p>
                hidden_cos={row.hidden_cosine_mean ?? "-"} | logit_cos={row.logit_cosine_mean ?? "-"} | kv_ratio=
                {row.kv_bytes_ratio_mean ?? "-"}
              </p>
            </article>
          ))}
        </div>
      </Panel>
    </div>
  );
}
