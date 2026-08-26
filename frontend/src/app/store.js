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
    }),
    {
      name: 'app-storage', // unique name
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        merchantId: state.merchantId,
        portalUrl: state.portalUrl,
        // Do not persist bearerToken in localStorage
      }),
    }
  )
);

// Initialization to pick up token from sessionStorage on load
const storedToken = sessionStorage.getItem('app_bearer_token');
if (storedToken) {
  useStore.setState({ bearerToken: storedToken });
}
