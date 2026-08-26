import React, { useEffect, useRef } from 'react';
import { useTheme } from '@mui/material/styles';
import { Box, Toolbar, CssBaseline, AppBar, Typography, Button, CircularProgress } from '@mui/material';
import { Outlet, useLocation, Link } from 'react-router-dom';
import ColorModeToggle from '../components/Common/ColorModeToggle';
import AccessibilityToggle from '../components/Common/AccessibilityToggle';
import SkuSidebar from './SkuSidebar';
import { DrawerHeader } from '../components/Sidebar/SidebarStyled';
import { Logo } from '../components/Logo';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getModelsStatus, loadModels } from './api';
import { Play, CheckCircle } from 'lucide-react';
import { useSnackbar } from 'notistack';

function getPageName(pathname) {
  if (pathname === '/dashboard') return 'Dashboard';
  if (pathname === '/jobs') return 'Jobs';
  if (pathname.startsWith('/jobs/')) return 'Job Detail';
  if (pathname === '/requests') return 'Requests';
  if (pathname === '/collections') return 'Collections';
  if (pathname === '/process-skus') return 'Process SKUs';
  if (pathname === '/interactive') return 'Interactive';
  if (pathname === '/sku-results') return 'SKU Results';
  if (pathname === '/rules') return 'Rules Engine';
  if (pathname === '/catalog') return 'Catalog Search';
  if (pathname === '/logs') return 'Logs';
  return 'SKU MatchOps';
}

export default function Layout() {
  const theme = useTheme();
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const location = useLocation();
  const wasLoadingRef = useRef(false);

  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['models-status'],
    queryFn: getModelsStatus,
    refetchInterval: (query) => {
      const d = query.state.data;
      if (d?.loading_in_progress) {
        return 2000;
      }
      return 10000;
    },
  });

  const loadMutation = useMutation({
    mutationFn: loadModels,
    onSuccess: (res) => {
      if (res?.status === 'success' || res?.message?.includes('already fully loaded')) {
        enqueueSnackbar('Models are already fully loaded!', { variant: 'success' });
      } else {
        wasLoadingRef.current = true;
        enqueueSnackbar('Model loading initiated in background...', { variant: 'info' });
        queryClient.setQueryData(['models-status'], (old) => ({
          ...old,
          loaded: false,
          loading_in_progress: true,
          status: 'loading',
        }));
        setTimeout(() => {
          refetchStatus();
        }, 500);
      }
    },
    onError: (e) => {
      enqueueSnackbar(`Failed to initiate model load: ${e.message}`, { variant: 'error' });
    },
  });

  useEffect(() => {
    if (statusData?.loaded && !statusData?.loading_in_progress) {
      if (wasLoadingRef.current) {
        enqueueSnackbar('Models loaded successfully!', { variant: 'success' });
        wasLoadingRef.current = false;
      }
    } else if (statusData?.loading_in_progress) {
      wasLoadingRef.current = true;
    }
  }, [statusData?.loaded, statusData?.loading_in_progress, enqueueSnackbar]);

  const isLoading = loadMutation.isPending || !!statusData?.loading_in_progress;
  const isLoaded = !!statusData?.loaded && !statusData?.loading_in_progress && !loadMutation.isPending;

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          zIndex: (t) => t.zIndex.drawer,
          background: theme.palette.background.paper,
          boxShadow: 'none',
          borderBottom: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Toolbar>
          <Logo width={120} />
          <Box sx={{ flexGrow: 1, pl: '120px', display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              component={Link}
              to="/dashboard"
              variant="body1"
              sx={{
                color: location.pathname === '/dashboard' ? theme.palette.text.primary : theme.palette.text.secondary,
                fontWeight: location.pathname === '/dashboard' ? 500 : 400,
                textDecoration: 'none',
                cursor: 'pointer',
                '&:hover': {
                  color: theme.palette.text.primary,
                  textDecoration: location.pathname === '/dashboard' ? 'none' : 'underline',
                },
              }}
            >
              Home
            </Typography>
            {location.pathname !== '/dashboard' && (
              <>
                <Typography variant="body1" sx={{ color: theme.palette.text.secondary }}>
                  /
                </Typography>
                <Typography variant="body1" sx={{ color: theme.palette.text.primary, fontWeight: 500 }}>
                  {getPageName(location.pathname)}
                </Typography>
              </>
            )}
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {isLoading ? (
              <Button
                variant="contained"
                size="small"
                color="primary"
                disabled
                startIcon={<CircularProgress size={14} color="inherit" />}
                sx={{
                  textTransform: 'none',
                  fontWeight: 600,
                  borderRadius: '6px',
                  boxShadow: 'none',
                }}
              >
                Loading Models…
              </Button>
            ) : isLoaded ? (
              <Button
                variant="outlined"
                size="small"
                color="success"
                disabled
                startIcon={<CheckCircle size={14} />}
                sx={{
                  textTransform: 'none',
                  fontWeight: 600,
                  borderRadius: '6px',
                  pointerEvents: 'none',
                  '&.Mui-disabled': {
                    borderColor: 'success.main',
                    color: 'success.main',
                    opacity: 1,
                  },
                }}
              >
                Models Loaded
              </Button>
            ) : (
              <Button
                variant="contained"
                size="small"
                color="primary"
                onClick={() => loadMutation.mutate()}
                startIcon={<Play size={14} />}
                sx={{
                  textTransform: 'none',
                  fontWeight: 600,
                  borderRadius: '6px',
                  boxShadow: 'none',
                  '&:hover': { boxShadow: 'none' },
                }}
              >
                Load Models
              </Button>
            )}
            <AccessibilityToggle />
            <ColorModeToggle />
          </Box>
        </Toolbar>
      </AppBar>
      <SkuSidebar />
      <Box component="main" sx={{ flexGrow: 1, minWidth: 0, overflowX: 'clip' }}>
        <DrawerHeader />
        <Outlet />
      </Box>
    </Box>
  );
}
