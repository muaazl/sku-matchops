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

      // MatchOps service API key (optional override over default environment key)
      apiKey: '',
      setApiKey: (apiKey = '') => {
        set({ apiKey });
        sessionStorage.setItem('matchops_api_key', apiKey);
      },
      clearSession: () => {
        set({ apiKey: '' });
        sessionStorage.removeItem('matchops_api_key');
      },
    }),
    {
      name: 'app-storage', // unique name
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        merchantId: state.merchantId,
        portalUrl: state.portalUrl,
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
