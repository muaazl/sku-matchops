import React, { Suspense } from 'react';
import { useRoutes } from 'react-router-dom';
import routes from './routes';
import useTitle from './components/UseTitle';
import { CssBaseline, LinearProgress, Box } from '@mui/material';
import useMediaQuery from '@mui/material/useMediaQuery';
import StyledMain from './components/Common/StyledMain';
import { ColorModeProvider } from './context/color-context';
import ServerOfflineModal from './components/Common/ServerOfflineModal';

function NewApp() {
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const prefersHighContrast = useMediaQuery('(prefers-contrast: more)');
  const storedMode = localStorage.getItem('qdrant-web-ui-theme');

  const resolvedMode = ['dark', 'light', 'high-contrast'].includes(storedMode)
    ? storedMode
    : prefersHighContrast
    ? 'high-contrast'
    : prefersDarkMode
    ? 'dark'
    : 'light';

  const routing = useRoutes(routes());
  useTitle('SKU MatchOps');

  return (
    <ColorModeProvider initialMode={resolvedMode}>
      <CssBaseline />
      <ServerOfflineModal />
      <StyledMain>
        <Suspense
          fallback={
            <Box sx={{ width: '100%', position: 'fixed', top: 0, left: 0, zIndex: 9999 }}>
              <LinearProgress />
            </Box>
          }
        >
          {routing}
        </Suspense>
      </StyledMain>
    </ColorModeProvider>
  );
}

export default NewApp;
