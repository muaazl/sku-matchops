import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export const useStore = create(
  persist(
    (set) => ({
      merchantId: '',
      portalUrl: '',
      bearerToken: '',
      setMerchantCredentials: (id, url, token) => {
        set({ merchantId: id, portalUrl: url, bearerToken: token });
        if (token) {
          sessionStorage.setItem('app_bearer_token', token);
        } else {
          sessionStorage.removeItem('app_bearer_token');
        }
      },
      clearToken: () => {
        set({ bearerToken: '' });
        sessionStorage.removeItem('app_bearer_token');
      },

      // MatchOps app session (separate from the merchant-portal token above). Credentials live
      // in sessionStorage only; the active tenant is remembered across sessions.
      authToken: '',
      apiKey: '',
      tenantId: 'default',
      setSession: ({ token = '', apiKey = '', tenantId }) => {
        set({ authToken: token, apiKey, ...(tenantId && { tenantId }) });
        sessionStorage.setItem('matchops_auth', JSON.stringify({ token, apiKey }));
      },
      clearSession: () => {
        set({ authToken: '', apiKey: '', tenantId: 'default' });
        sessionStorage.removeItem('matchops_auth');
      },
      setTenantId: (tenantId) => set({ tenantId: tenantId || 'default' }),
    }),
    {
      name: 'app-storage', // unique name
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        merchantId: state.merchantId,
        portalUrl: state.portalUrl,
        tenantId: state.tenantId,
        // Do not persist bearerToken / authToken / apiKey in localStorage
      }),
    }
  )
);

// Initialization to pick up tokens from sessionStorage on load
const storedToken = sessionStorage.getItem('app_bearer_token');
if (storedToken) {
  useStore.setState({ bearerToken: storedToken });
}
try {
  const { token = '', apiKey = '' } = JSON.parse(sessionStorage.getItem('matchops_auth') || '{}');
  useStore.setState({ authToken: token, apiKey });
} catch {
  sessionStorage.removeItem('matchops_auth');
}
