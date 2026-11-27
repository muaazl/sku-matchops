import React from 'react';
import { useTheme } from '@mui/material/styles';
import { AppBar, Box, CssBaseline, Toolbar, Typography } from '@mui/material';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useModelStatus } from '../../hooks/useModelStatus';
import { getPageName } from '../../utils/constants';
import ColorModeToggle from '../common/ColorModeToggle';
import { Logo } from '../common/Logo';
import { ModelStatusButton } from './ModelStatusButton';
import { DrawerHeader } from './SidebarStyled';
import SkuSidebar from './SkuSidebar';

export default function Layout() {
  const theme = useTheme();
  const location = useLocation();
  const { isLoading, isLoaded, triggerLoadModels } = useModelStatus();

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
            <ModelStatusButton isLoading={isLoading} isLoaded={isLoaded} onLoadModels={triggerLoadModels} />
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
