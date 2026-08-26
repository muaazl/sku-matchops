import axios from 'axios';

// ── Dynamic Backend URL Resolution ──────────────────────────────────────────
// Locally: uses localhost:8000 (no tunnel needed).
// On Vercel: fetches the tunnel URL from the _config tab of the catalog sheet
//            (published as CSV). This updates automatically when start_tunnel.ps1 runs.

const LOCAL_BACKEND = 'http://localhost:8000';
const SHEET_ID = import.meta.env.VITE_GOOGLE_SHEET_ID || '';
const CONFIG_CSV_URL = SHEET_ID
  ? `https://docs.google.com/spreadsheets/d/${SHEET_ID}/gviz/tq?tqx=out:csv&sheet=_config`
  : '';

let _resolvedBaseUrl = import.meta.env.VITE_API_URL || null;

/**
 * Resolves the backend API base URL.
 * - If VITE_API_URL is set, uses that.
 * - If running on localhost, uses localhost:8000.
 * - Otherwise (Vercel), fetches from the Google Sheets _config CSV.
 */
async function resolveBaseUrl() {
  if (_resolvedBaseUrl) return _resolvedBaseUrl;

  // Running locally — connect directly
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    _resolvedBaseUrl = LOCAL_BACKEND;
    return _resolvedBaseUrl;
  }

  // Running on Vercel — fetch tunnel URL from published Google Sheet
  try {
    const resp = await fetch(CONFIG_CSV_URL);
    const csv = await resp.text();
    // CSV format: "Key","Value"\n"tunnel_url","https://xxx.trycloudflare.com"\n...
    const lines = csv.trim().split('\n');
    for (const line of lines) {
      if (line.includes('tunnel_url')) {
        // Extract the URL value from CSV (handles quoted values)
        const match = line.match(/tunnel_url[",\s]+["']?(https:\/\/[^"'\s,]+)/i);
        if (match) {
          _resolvedBaseUrl = match[1];
          console.log('[MatchOps] Resolved backend URL from Google Sheets:', _resolvedBaseUrl);
          return _resolvedBaseUrl;
        }
      }
    }
  } catch (err) {
    console.warn('[MatchOps] Failed to fetch tunnel URL from Google Sheets:', err.message);
  }

  // Final fallback
  _resolvedBaseUrl = LOCAL_BACKEND;
  console.warn('[MatchOps] Using fallback backend URL:', _resolvedBaseUrl);
  return _resolvedBaseUrl;
}

// Eagerly resolve on load (non-blocking)
resolveBaseUrl();

const api = axios.create({
  baseURL: LOCAL_BACKEND, // initial default, overridden by interceptor
});

// Request interceptor: ensures every request uses the resolved tunnel URL
api.interceptors.request.use(async (config) => {
  const base = await resolveBaseUrl();
  config.baseURL = base;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Import store here to avoid circular dependencies
      import('./store').then(({ useStore }) => {
        useStore.getState().clearToken();
      });
    }
    return Promise.reject(error);
  }
);

// Jobs
export const getJobs = async (params) => (await api.get('/jobs', { params })).data;
export const getJob = async (id) => (await api.get(`/jobs/${id}`)).data;
export const getDashboardStats = async (params) => (await api.get('/jobs/dashboard-stats', { params })).data;
export const cancelJob = async (id) => (await api.post(`/jobs/${id}/cancel`)).data;
export const retryJob = async (id) => (await api.post(`/jobs/${id}/retry`)).data;

// Batches
export const createBatch = async (formData) =>
  (await api.post('/batches', formData, { headers: { 'Content-Type': 'multipart/form-data' } })).data;
export const merchantFetch = async (payload) => (await api.post('/merchant-fetch', payload)).data;
export const getBatch = async (id) => (await api.get(`/batches/${id}`)).data;

// Qdrant
export const getCollections = async () => (await api.get('/vector-db/collections')).data;
export const getCollection = async (name) => (await api.get(`/vector-db/collections/${name}`)).data;
export const searchCollection = async ({ name, query, topK, scoreThreshold, filters }) =>
  (
    await api.post(`/vector-db/collections/${name}/search`, {
      query,
      top_k: topK,
      score_threshold: scoreThreshold,
      filters,
    })
  ).data;

// Rules
export const getRules = async (params) => (await api.get('/rules', { params })).data;
export const saveRule = async (rule) => {
  if (rule.rule_id.startsWith('new_rule_')) {
    return (await api.post('/rules', rule)).data;
  }
  return (await api.put(`/rules/${rule.rule_id}`, rule)).data;
};
export const deleteRule = async (id) => (await api.delete(`/rules/${id}`)).data;
export const reorderRules = async (payload) => (await api.put('/rules/reorder', payload)).data;
export const testRule = async ({ rule, sampleRecord }) =>
  (await api.post('/rules/test-draft', { rule, sample_record: sampleRecord })).data;

// Catalog
export const searchCatalog = async (params) => (await api.get('/catalog', { params })).data;
export const refreshCatalog = async () => (await api.post('/catalog/refresh')).data;
export const buildCatalogCache = async () => (await api.post('/catalog/build-cache')).data;
export const checkCatalogSync = async (params) => (await api.get('/catalog/check-sync', { params })).data;

// History / Processed SKUs
export const getProcessedSkus = async (params) => (await api.get('/processed-skus', { params })).data;

// API Requests
export const getApiRequests = async (params) => (await api.get('/api-requests', { params })).data;
export const getApiRequest = async (id) => (await api.get(`/api-requests/${id}`)).data;

// Models loading
export const getModelsStatus = async () => (await api.get('/models-status')).data;
export const loadModels = async () => (await api.post('/load-models')).data;

// Interactive Single SKU Execution
export const runInteractiveSingle = async (payload) => (await api.post('/interactive/run-single', payload)).data;
export const rerunInteractiveRules = async (payload) => (await api.post('/interactive/rerun-rules', payload)).data;
export const getTemplateSuggestions = async (payload) => (await api.post('/interactive/suggest', payload)).data;
export const runInteractiveAudit = async (payload) => (await api.post('/interactive/audit', payload)).data;

// Logs
export const getLogs = async (lines) => (await api.get('/logs', { params: { lines } })).data;

// Health check
export const checkHealth = async () => (await api.get('/health', { timeout: 4000 })).data;
