import React, { useState, useEffect } from 'react';
import { useTheme } from '@mui/material/styles';
import { Box, Paper, Typography, Button, CircularProgress } from '@mui/material';
import { WifiOff, RefreshCw } from 'lucide-react';
import { checkHealth } from '../../app/api';

export default function ServerOfflineModal() {
  const theme = useTheme();
  const [isOffline, setIsOffline] = useState(false);
  const [checking, setChecking] = useState(false);

  const runHealthCheck = async () => {
    setChecking(true);
    try {
      await checkHealth();
      setIsOffline(false);
    } catch (err) {
      setIsOffline(true);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    runHealthCheck();
  }, []);

  if (!isOffline) {
    return null;
  }

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 99999,
        backdropFilter: 'blur(8px)',
        backgroundColor: theme.palette.mode === 'dark' ? 'rgba(0, 0, 0, 0.6)' : 'rgba(0, 0, 0, 0.3)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: 2,
        boxSizing: 'border-box',
      }}
    >
      <Paper
        elevation={8}
        sx={{
          maxWidth: 400,
          width: '100%',
          p: 4,
          borderRadius: 2,
          backgroundColor: theme.palette.background.paper,
          color: theme.palette.text.primary,
          textAlign: 'center',
          border: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Box
          sx={{
            display: 'inline-flex',
            p: 1.5,
            borderRadius: '50%',
            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
            mb: 2,
          }}
        >
          <WifiOff size={32} color={theme.palette.text.secondary} />
        </Box>

        <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
          Server is Offline
        </Typography>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.5 }}>
          Unable to connect to the backend server. Please check if the server is running and try again.
        </Typography>

        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={runHealthCheck}
          disabled={checking}
          startIcon={checking ? <CircularProgress size={16} color="inherit" /> : <RefreshCw size={16} />}
          sx={{ textTransform: 'none', fontWeight: 600, py: 1 }}
        >
          {checking ? 'Checking Connection…' : 'Retry Connection'}
        </Button>
      </Paper>
    </Box>
  );
}
