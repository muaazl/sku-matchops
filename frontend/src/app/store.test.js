import { describe, it, expect, beforeEach } from 'vitest';
import { useStore } from './store';

describe('useStore Zustand store', () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    useStore.setState({
      merchantId: '',
      portalUrl: '',
      bearerToken: '',
      apiKey: '',
    });
  });

  it('manages merchant credentials and session storage', () => {
    useStore.getState().setMerchantCredentials('merchant-123', 'https://portal.example.com', 'bearer-abc');
    expect(useStore.getState().merchantId).toBe('merchant-123');
    expect(useStore.getState().portalUrl).toBe('https://portal.example.com');
    expect(useStore.getState().bearerToken).toBe('bearer-abc');
    expect(sessionStorage.getItem('app_bearer_token')).toBe('bearer-abc');

    useStore.getState().clearToken();
    expect(useStore.getState().bearerToken).toBe('');
    expect(sessionStorage.getItem('app_bearer_token')).toBeNull();
  });

  it('manages service API key and session storage', () => {
    useStore.getState().setApiKey('test-api-key');
    expect(useStore.getState().apiKey).toBe('test-api-key');
    expect(sessionStorage.getItem('matchops_api_key')).toBe('test-api-key');

    useStore.getState().clearSession();
    expect(useStore.getState().apiKey).toBe('');
    expect(sessionStorage.getItem('matchops_api_key')).toBeNull();
  });
});
