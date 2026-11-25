import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from './useAppStore';

describe('useAppStore Zustand store', () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    useAppStore.setState({
      merchantId: '',
      portalUrl: '',
      bearerToken: '',
      apiKey: '',
    });
  });

  it('manages merchant credentials and session storage', () => {
    useAppStore.getState().setMerchantCredentials('merchant-123', 'https://portal.example.com', 'bearer-abc');
    expect(useAppStore.getState().merchantId).toBe('merchant-123');
    expect(useAppStore.getState().portalUrl).toBe('https://portal.example.com');
    expect(useAppStore.getState().bearerToken).toBe('bearer-abc');
    expect(sessionStorage.getItem('app_bearer_token')).toBe('bearer-abc');

    useAppStore.getState().clearToken();
    expect(useAppStore.getState().bearerToken).toBe('');
    expect(sessionStorage.getItem('app_bearer_token')).toBeNull();
  });

  it('manages service API key and session storage', () => {
    useAppStore.getState().setApiKey('test-api-key');
    expect(useAppStore.getState().apiKey).toBe('test-api-key');
    expect(sessionStorage.getItem('matchops_api_key')).toBe('test-api-key');

    useAppStore.getState().clearSession();
    expect(useAppStore.getState().apiKey).toBe('');
    expect(sessionStorage.getItem('matchops_api_key')).toBeNull();
  });
});
