/**
 * Parses raw CSV / TSV text handling quotes, escaped quotes, and newlines.
 * @param {string} text - The raw CSV/TSV input text.
 * @return {{ headers: string[], rows: Array<Object.<string, string>> }}
 */
export function parseCSV(text) {
  if (!text) return { headers: [], rows: [] };

  const firstLine = text.split(/\r?\n/)[0] || '';
  const delimiter = firstLine.includes('\t') ? '\t' : ',';

  const lines = [];
  let row = [''];
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i + 1];

    if (c === '"') {
      if (inQuotes && next === '"') {
        row[row.length - 1] += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (c === delimiter && !inQuotes) {
      row.push('');
    } else if ((c === '\r' || c === '\n') && !inQuotes) {
      if (c === '\r' && next === '\n') {
        i++;
      }
      lines.push(row);
      row = [''];
    } else {
      row[row.length - 1] += c;
    }
  }
  if (row.length > 1 || row[0] !== '') {
    lines.push(row);
  }

  if (lines.length === 0) return { headers: [], rows: [] };

  const headers = lines[0].map((h) => h.trim());
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i];
    if (values.length === 1 && values[0] === '') continue;
    const rowObj = {};
    headers.forEach((header, index) => {
      rowObj[header] = values[index] !== undefined ? values[index].trim() : '';
    });
    rows.push(rowObj);
  }

  return { headers, rows };
}

/**
 * Finds a matching column name in headers given candidate aliases.
 * @param {string[]} headers - List of CSV column header names.
 * @param {string[]} choices - Candidate alias names to match.
 * @return {string|undefined}
 */
export function findColumn(headers, choices) {
  return headers.find((h) => choices.includes(h.toLowerCase().trim()));
}

/**
 * Maps raw CSV rows to normalized SKU items.
 * @param {string[]} headers - List of CSV column header names.
 * @param {Array<Object.<string, string>>} rows - List of parsed CSV row objects.
 * @return {Array<{ name: string, price: string, description: string, category: string }>}
 */
export function mapRows(headers, rows) {
  const nameCol = findColumn(headers, ['name', 'sku_name', 'sku', 'title']) || headers[0];
  const priceCol = findColumn(headers, ['price', 'cost', 'mrp']);
  const descCol = findColumn(headers, ['description', 'desc']);
  const catCol = findColumn(headers, ['category', 'cat', 'type']);

  return rows.map((row) => ({
    name: row[nameCol] || '',
    price: row[priceCol] || '0',
    description: row[descCol] || '',
    category: row[catCol] || '',
  }));
}
