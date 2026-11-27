import { Button } from '@mui/material';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { SnackbarProvider, closeSnackbar } from 'notistack';
import App from './App';
import { useAppStore } from './hooks/useAppStore';
import './index.css';
import { configureApiKeyProvider } from './services/httpClient';
import reportWebVitals from './utils/reportWebVitals';

configureApiKeyProvider(() => useAppStore.getState().apiKey);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30 * 1000, // Avoid redundant refetches on page navigation.
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <SnackbarProvider
          anchorOrigin={{
            vertical: 'top',
            horizontal: 'center',
          }}
          style={{ flexWrap: 'nowrap' }}
          action={(snackbarId) => (
            <Button
              variant="outlined"
              color="inherit"
              onClick={() => {
                closeSnackbar(snackbarId);
              }}
            >
              Dismiss
            </Button>
          )}
        >
          <App />
        </SnackbarProvider>
      </HashRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>
);

reportWebVitals();
