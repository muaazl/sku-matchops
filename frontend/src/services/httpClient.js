import axios from 'axios';
import { getStoredApiKey } from '../utils/authStorage';

// Use 127.0.0.1 rather than localhost because Windows delays ~2s on ::1 IPv6 resolution first.
const LOCAL_BACKEND = 'http://127.0.0.1:8000';
const SHEET_ID = import.meta.env.VITE_GOOGLE_SHEET_ID || '';
const CONFIG_CSV_URL = SHEET_ID
  ? `https://docs.google.com/spreadsheets/d/${SHEET_ID}/gviz/tq?tqx=out:csv&sheet=_config`
  : '';
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY || 'bismillah';

let resolvedBaseUrl = import.meta.env.VITE_API_URL || null;
let apiKeyProvider = null;

export function configureApiKeyProvider(provider) {
  apiKeyProvider = typeof provider === 'function' ? provider : null;
}

export function getActiveApiKey() {
  return apiKeyProvider?.() || getStoredApiKey() || DEFAULT_API_KEY;
}

/**
 * Resolves the backend API base URL from env, localhost detection, or published Google Sheet _config.
 * @return {Promise<string>}
 */
export async function resolveBaseUrl() {
  if (resolvedBaseUrl) return resolvedBaseUrl;

  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    resolvedBaseUrl = LOCAL_BACKEND;
    return resolvedBaseUrl;
  }

  try {
    const resp = await fetch(CONFIG_CSV_URL);
    const csv = await resp.text();
    const lines = csv.trim().split('\n');
    for (const line of lines) {
      if (line.includes('tunnel_url')) {
        const match = line.match(/tunnel_url[",\s]+["']?(https:\/\/[^"'\s,]+)/i);
        if (match) {
          resolvedBaseUrl = match[1];
          return resolvedBaseUrl;
        }
      }
    }
  } catch (err) {
    console.warn('[MatchOps] Failed to fetch tunnel URL from Google Sheets:', err.message);
  }

  resolvedBaseUrl = LOCAL_BACKEND;
  console.warn('[MatchOps] Using fallback backend URL:', resolvedBaseUrl);
  return resolvedBaseUrl;
}

// Eagerly resolve on load (non-blocking)
resolveBaseUrl();

export const httpClient = axios.create({
  baseURL: LOCAL_BACKEND,
});

httpClient.interceptors.request.use(async (config) => {
  config.baseURL = await resolveBaseUrl();
  config.headers['X-API-Key'] = getActiveApiKey();
  return config;
});

httpClient.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(error)
);
