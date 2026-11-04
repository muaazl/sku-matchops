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
      authToken: '',
      apiKey: '',
      loginModalOpen: false,
      loginPromptDismissed: false,
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

  it('manages login session and auth tokens', () => {
    useStore.getState().setSession({ token: 'jwt-xyz-789', apiKey: 'test-key' });
    expect(useStore.getState().authToken).toBe('jwt-xyz-789');
    expect(useStore.getState().apiKey).toBe('test-key');
    expect(useStore.getState().loginModalOpen).toBe(false);
    expect(useStore.getState().loginPromptDismissed).toBe(false);

    const savedAuth = JSON.parse(sessionStorage.getItem('matchops_auth') || '{}');
    expect(savedAuth.token).toBe('jwt-xyz-789');
    expect(savedAuth.apiKey).toBe('test-key');

    useStore.getState().clearSession();
    expect(useStore.getState().authToken).toBe('');
    expect(useStore.getState().apiKey).toBe('');
    expect(sessionStorage.getItem('matchops_auth')).toBeNull();
  });

  it('handles login modal open, close, and 401 prompt behavior', () => {
    useStore.getState().openLoginModal();
    expect(useStore.getState().loginModalOpen).toBe(true);

    useStore.getState().closeLoginModal();
    expect(useStore.getState().loginModalOpen).toBe(false);
    expect(useStore.getState().loginPromptDismissed).toBe(true);

    // If dismissed, promptLoginOn401 does not re-open the modal
    useStore.getState().promptLoginOn401();
    expect(useStore.getState().loginModalOpen).toBe(false);

    // Reset prompt allows it to open again
    useStore.getState().resetLoginPrompt();
    expect(useStore.getState().loginPromptDismissed).toBe(false);

    useStore.getState().promptLoginOn401();
    expect(useStore.getState().loginModalOpen).toBe(true);
  });

  it('evicts session on 401 if user previously had an active session', () => {
    useStore.getState().setSession({ token: 'active-token' });
    expect(useStore.getState().authToken).toBe('active-token');

    useStore.getState().promptLoginOn401();
    expect(useStore.getState().authToken).toBe('');
    expect(useStore.getState().loginModalOpen).toBe(true);
    expect(sessionStorage.getItem('matchops_auth')).toBeNull();
  });
});
