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
      // in sessionStorage only. The tenant comes from the signed-in account (server side).
      authToken: '',
      apiKey: '',
      loginModalOpen: false,
      loginPromptDismissed: false,
      openLoginModal: () => set({ loginModalOpen: true }),
      closeLoginModal: () => set({ loginModalOpen: false, loginPromptDismissed: true }),
      resetLoginPrompt: () => set({ loginPromptDismissed: false }),
      promptLoginOn401: () =>
        set((state) => {
          const hadSession = Boolean(state.authToken || state.apiKey);
          if (hadSession) {
            sessionStorage.removeItem('matchops_auth');
            return { authToken: '', apiKey: '', loginModalOpen: true, loginPromptDismissed: false };
          }
          if (state.loginPromptDismissed) return state;
          return { loginModalOpen: true };
        }),
      setSession: ({ token = '', apiKey = '' }) => {
        set({ authToken: token, apiKey, loginModalOpen: false, loginPromptDismissed: false });
        sessionStorage.setItem('matchops_auth', JSON.stringify({ token, apiKey }));
      },
      clearSession: () => {
        set({ authToken: '', apiKey: '', loginPromptDismissed: false });
        sessionStorage.removeItem('matchops_auth');
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
