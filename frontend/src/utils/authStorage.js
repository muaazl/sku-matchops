const BEARER_TOKEN_KEY = 'app_bearer_token';
const API_KEY_STORAGE_KEY = 'matchops_api_key';
const THEME_MODE_KEY = 'sku-matchops-theme';
const LEGACY_THEME_MODE_KEY = 'qdrant-web-ui-theme';
const VALID_THEME_MODES = ['dark', 'light', 'high-contrast'];

export function getStoredBearerToken() {
  return sessionStorage.getItem(BEARER_TOKEN_KEY) || '';
}

export function setStoredBearerToken(token) {
  if (token) {
    sessionStorage.setItem(BEARER_TOKEN_KEY, token);
  } else {
    sessionStorage.removeItem(BEARER_TOKEN_KEY);
  }
}

export function clearStoredBearerToken() {
  sessionStorage.removeItem(BEARER_TOKEN_KEY);
}

export function getStoredApiKey() {
  return sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';
}

export function setStoredApiKey(apiKey) {
  sessionStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
}

export function clearStoredApiKey() {
  sessionStorage.removeItem(API_KEY_STORAGE_KEY);
}

export function resolveInitialThemeMode({ prefersDarkMode = false, prefersHighContrast = false } = {}) {
  const storedMode = localStorage.getItem(THEME_MODE_KEY) || localStorage.getItem(LEGACY_THEME_MODE_KEY);
  if (VALID_THEME_MODES.includes(storedMode)) {
    return storedMode;
  }
  if (prefersHighContrast) return 'high-contrast';
  return prefersDarkMode ? 'dark' : 'light';
}

export function saveThemeMode(mode) {
  localStorage.setItem(THEME_MODE_KEY, mode);
  if (localStorage.getItem(LEGACY_THEME_MODE_KEY)) {
    localStorage.removeItem(LEGACY_THEME_MODE_KEY);
  }
}
