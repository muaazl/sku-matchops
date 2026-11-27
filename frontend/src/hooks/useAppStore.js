import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import {
  clearStoredApiKey,
  clearStoredBearerToken,
  getStoredApiKey,
  getStoredBearerToken,
  setStoredApiKey,
  setStoredBearerToken,
} from '../utils/authStorage';

export const useAppStore = create(
  persist(
    (set) => ({
      merchantId: '',
      portalUrl: '',
      bearerToken: '',
      setMerchantCredentials: (id, url, token) => {
        set({ merchantId: id, portalUrl: url, bearerToken: token });
        setStoredBearerToken(token);
      },
      clearToken: () => {
        set({ bearerToken: '' });
        clearStoredBearerToken();
      },

      // MatchOps service API key (optional override over default environment key)
      apiKey: '',
      setApiKey: (apiKey = '') => {
        set({ apiKey });
        setStoredApiKey(apiKey);
      },
      clearSession: () => {
        set({ apiKey: '' });
        clearStoredApiKey();
      },
    }),
    {
      name: 'app-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        merchantId: state.merchantId,
        portalUrl: state.portalUrl,
      }),
    }
  )
);

const storedToken = getStoredBearerToken();
const storedApiKey = getStoredApiKey();
if (storedToken || storedApiKey) {
  useAppStore.setState({
    ...(storedToken && { bearerToken: storedToken }),
    ...(storedApiKey && { apiKey: storedApiKey }),
  });
}
