export interface CsvRow {
  [key: string]: string;
}

export function parseCsv(text: string): CsvRow[] {
  const trimmed = text.trim();
  if (!trimmed) {
    return [];
  }
  const lines = trimmed.split(/\r?\n/).filter(Boolean);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: CsvRow = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
}
