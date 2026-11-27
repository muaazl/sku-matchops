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

export function parseUtcTimestamp(str) {
  if (!str) return null;
  const raw = String(str);
  // Treat naive timestamps (no explicit offset) as UTC because backend timestamps are always UTC.
  const hasExplicitOffset = /Z$|[+-]\d{2}:?\d{2}$/.test(raw);
  const isoLike = raw.includes('T') ? raw : raw.replace(' ', 'T');
  const d = new Date(hasExplicitOffset ? isoLike : `${isoLike}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function fmtTime(str) {
  if (!str) return '—';
  const d = parseUtcTimestamp(str);
  if (!d) return str;
  return d.toLocaleString('en-US', {
    timeZone: 'Asia/Colombo',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function computeElapsedSeconds(startedAt, nowMs = Date.now()) {
  const parsed = parseUtcTimestamp(startedAt);
  if (!parsed) return null;
  return Math.max(1, Math.floor((nowMs - parsed.getTime()) / 1000));
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

export function normalizeConfidencePct(value) {
  const parsed = typeof value === 'number' ? value : Number(value);
  const num = Number.isFinite(parsed) ? parsed : 0;
  const normalized = num > 1 ? num / 100 : num;
  return Math.round(Math.max(0, Math.min(1, normalized)) * 100);
}

export function formatConfidencePct(value) {
  if (value === null || value === undefined || value === '') return null;
  return `${normalizeConfidencePct(value)}%`;
}

export function normalizeMatchSources(rawSources = []) {
  const map = {};
  for (const item of rawSources) {
    const src = (item.source || '').trim().toLowerCase();
    const norm = src === 'classifier' ? 'Classifier' : 'Matcher';
    map[norm] = (map[norm] || 0) + (item.count || 0);
  }
  const total = Object.values(map).reduce((a, b) => a + b, 0);
  return Object.entries(map)
    .filter((entry) => entry[1] > 0)
    .map(([source, count]) => ({
      source,
      count,
      pct: total > 0 ? Math.round((count / total) * 1000) / 10 : 0,
    }));
}

export function safeJsonParse(value, fallback = null) {
  if (value === null || value === undefined) return fallback;
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

export function formatJsonPretty(value) {
  if (typeof value !== 'string') {
    return JSON.stringify(value, null, 2);
  }
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}
