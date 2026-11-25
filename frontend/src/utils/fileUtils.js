export const SUPPORTED_CSV_EXTS = ['.csv', '.tsv', '.txt'];
export const SUPPORTED_BULK_EXTS = ['.csv', '.tsv', '.txt', '.xlsx', '.xls'];

export function isSupportedCsvFile(filename = '') {
  const lower = filename.toLowerCase();
  return SUPPORTED_CSV_EXTS.some((ext) => lower.endsWith(ext));
}

export function isSupportedBulkFile(filename = '') {
  const lower = filename.toLowerCase();
  return SUPPORTED_BULK_EXTS.some((ext) => lower.endsWith(ext));
}

export function stripFileExtension(filename = '') {
  return filename.replace(/\.[^/.]+$/, '');
}

export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

export function readFileAsArrayBuffer(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}
