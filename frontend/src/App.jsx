import React, { Suspense } from 'react';
import { useRoutes } from 'react-router-dom';
import { Box, CssBaseline, LinearProgress } from '@mui/material';
import useMediaQuery from '@mui/material/useMediaQuery';
import { ColorModeProvider } from './components/common/ColorModeProvider';
import ServerOfflineModal from './components/common/ServerOfflineModal';
import StyledMain from './components/common/StyledMain';
import { useTitle } from './hooks/useTitle';
import routes from './routes';
import { resolveInitialThemeMode } from './utils/authStorage';

function App() {
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const prefersHighContrast = useMediaQuery('(prefers-contrast: more)');
  const resolvedMode = resolveInitialThemeMode({ prefersDarkMode, prefersHighContrast });

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

export default App;
