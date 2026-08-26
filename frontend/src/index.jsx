import { Button } from '@mui/material';
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { HashRouter } from 'react-router-dom';
import { SnackbarProvider, closeSnackbar } from 'notistack';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30 * 1000, // 30s — avoid redundant refetches on page navigation
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
