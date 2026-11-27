import React, { useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { ThemeProvider } from '@mui/material/styles';
import { ColorModeContext } from '../../hooks/useColorMode';
import { createTheme } from '../../theme';
import { saveThemeMode } from '../../utils/authStorage';

export function ColorModeProvider({ children, initialMode }) {
  const [colorMode, setColorMode] = useState(initialMode);

  useEffect(() => {
    saveThemeMode(colorMode);
  }, [colorMode]);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: colorMode,
        },
      }),
    [colorMode]
  );

  useEffect(() => {
    const handleMessage = (event) => {
      let payload = event.data;
      if (typeof payload === 'string') {
        try {
          payload = JSON.parse(payload);
        } catch (err) {
          console.log('could not parse message payload as JSON', err);
        }
      }
      const mode = payload?.mode ?? payload?.qdrantTheme ?? payload;
      if (mode === 'light' || mode === 'dark' || mode === 'high-contrast') {
        setColorMode(mode);
        if (event.source) {
          event.source.postMessage(`{ status: 'success', message: 'Color mode changed to ${mode}' }`, event.origin);
        }
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  const value = useMemo(() => ({ colorMode, setColorMode }), [colorMode]);

  return (
    <ColorModeContext.Provider value={value}>
      <ThemeProvider theme={theme}>{children}</ThemeProvider>
    </ColorModeContext.Provider>
  );
}

ColorModeProvider.propTypes = {
  children: PropTypes.node.isRequired,
  initialMode: PropTypes.string.isRequired,
};
