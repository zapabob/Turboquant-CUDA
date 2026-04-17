export interface MetricRow {
  label: string;
  value: string;
}

export function MetricTable({
  title,
  rows,
}: {
  title: string;
  rows: MetricRow[];
}): JSX.Element {
  return (
    <div className="stack-card">
      <span className="detail-label">{title}</span>
      {rows.length ? (
        <div className="table-wrap">
          <table>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <th>{row.label}</th>
                  <td>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="empty-copy">No metrics available yet.</p>
      )}
    </div>
  );
}
