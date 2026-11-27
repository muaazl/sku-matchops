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

/**
 * Parses an uploaded Excel workbook (.xlsx/.xls) into one { headers, rows } entry per
 * non-empty sheet/tab, dynamically importing SheetJS (`xlsx`).
 * @param {ArrayBuffer} arrayBuffer - The raw workbook file contents.
 * @return {Promise<Record<string, { headers: string[], rows: Array<Object.<string, string>> }>>}
 */
export async function parseExcelWorkbook(arrayBuffer) {
  const XLSX = await import('xlsx');
  const workbook = XLSX.read(arrayBuffer, { type: 'array' });
  const result = {};

  workbook.SheetNames.forEach((sheetName) => {
    const sheet = workbook.Sheets[sheetName];
    const grid = XLSX.utils.sheet_to_json(sheet, { header: 1, raw: false, defval: '' });
    if (!grid || grid.length === 0) return;

    const headers = (grid[0] || []).map((h) => String(h ?? '').trim());
    if (headers.every((h) => !h)) return;

    const rows = [];
    for (let i = 1; i < grid.length; i++) {
      const values = grid[i] || [];
      if (values.every((v) => String(v ?? '').trim() === '')) continue;
      const rowObj = {};
      headers.forEach((header, idx) => {
        rowObj[header] = values[idx] !== undefined ? String(values[idx]).trim() : '';
      });
      rows.push(rowObj);
    }

    if (rows.length > 0) {
      result[sheetName] = { headers, rows };
    }
  });

  return result;
}

export function toCsv(rows) {
  if (!rows.length) return '';
  const headers = Object.keys(rows[0]);
  const escape = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [headers.join(','), ...rows.map((r) => headers.map((h) => escape(r[h])).join(','))].join('\n');
}

export function download(filename, text, mime = 'text/csv') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function buildBatchFormData(rows, name, jobDomain, jobTask) {
  const normalizedRows = rows.map((row) => ({
    name: String(row.name || ''),
    price: String(row.price || ''),
    description: String(row.description || ''),
    category: String(row.category || ''),
  }));
  const csvContent = normalizedRows.length > 0 ? toCsv(normalizedRows) : 'name,price,description,category';
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const fileToUpload = new File([blob], name, { type: 'text/csv' });

  const formData = new FormData();
  formData.append('file', fileToUpload);
  formData.append('domain', jobDomain);
  formData.append('task', jobTask);
  formData.append('created_by', 'admin');
  return formData;
}
