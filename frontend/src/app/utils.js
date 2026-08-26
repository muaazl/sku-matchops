export function fmtEta(seconds) {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds <= 0) return 'done';
  const sec = Math.max(0, Math.round(Number(seconds)));
  if (Number.isNaN(sec)) return '—';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function fmtDuration(minutes) {
  if (minutes === null || minutes === undefined) return '—';
  const totalSeconds = Math.max(0, Math.round(Number(minutes) * 60));
  if (Number.isNaN(totalSeconds)) return '—';
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function fmtTime(str) {
  if (!str) return '—';
  const d = new Date(str.includes('T') ? str : str.replace(' ', 'T') + 'Z');
  if (Number.isNaN(d.getTime())) return str;
  return d.toLocaleString('en-US', {
    timeZone: 'Asia/Colombo',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function progressPct(job) {
  if (!job.total_items) return 0;
  return Math.min(100, Math.round((job.completed_items / job.total_items) * 100));
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

export function toCsv(rows) {
  if (!rows.length) return '';
  const headers = Object.keys(rows[0]);
  const escape = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [headers.join(','), ...rows.map((r) => headers.map((h) => escape(r[h])).join(','))].join('\n');
}

export function formatGk(gkJson, fallback = '—') {
  if (!gkJson || gkJson === 'None' || gkJson === '["None"]') return fallback;
  try {
    const parsed = typeof gkJson === 'string' ? JSON.parse(gkJson) : gkJson;
    if (Array.isArray(parsed)) {
      return parsed.filter(Boolean).join(', ') || fallback;
    }
    if (typeof parsed === 'object' && parsed !== null) {
      return Object.values(parsed).filter(Boolean).join(', ') || fallback;
    }
    return String(parsed) || fallback;
  } catch {
    return String(gkJson) || fallback;
  }
}

export function formatGkText(gkJson) {
  return formatGk(gkJson, '');
}
