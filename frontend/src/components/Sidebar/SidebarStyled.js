import { styled } from '@mui/material/styles';
import MuiDrawer from '@mui/material/Drawer';
import { List, ListItem, ListItemButton, Typography } from '@mui/material';

const drawerWidth = 240;

export const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'flex-end',
  padding: theme.spacing(0, 1),
  // necessary for content to be below app bar
  ...theme.mixins.toolbar,
}));

export const Drawer = styled(MuiDrawer)(() => ({
  width: drawerWidth,
  flexShrink: 0,
  whiteSpace: 'nowrap',
  boxSizing: 'border-box',
  // Portal tokens (src/index.css): the dark navy sidebar is the same in every color mode
  '& .MuiDrawer-paper': {
    width: drawerWidth,
    overflowX: 'hidden',
    backgroundColor: 'var(--portal-sidebar-bg)',
    borderRight: '1px solid var(--portal-sidebar-border)',
    color: 'var(--portal-sidebar-text)',
  },
  '& .MuiDivider-root': {
    borderColor: 'var(--portal-sidebar-border)',
  },
}));

export const StyledListItemButton = styled(ListItemButton, {
  shouldForwardProp: (prop) => prop !== 'isActive',
})(({ isActive }) => ({
  display: 'flex',
  height: '36px',
  padding: '6px 12px',
  justifyContent: 'space-between',
  alignItems: 'center',
  alignSelf: 'stretch',
  borderRadius: 'var(--portal-radius)',
  '& .MuiListItemText-primary': {
    fontSize: '13px',
    fontWeight: 500,
  },
  '& .MuiListItemText-primary, & .MuiListItemIcon-root': {
    color: 'var(--portal-sidebar-text)',
  },
  '&:hover': {
    backgroundColor: 'var(--portal-sidebar-surface)',
    '& .MuiListItemText-primary, & .MuiListItemIcon-root': {
      color: 'var(--portal-sidebar-text-active)',
    },
  },
  ...(isActive && {
    backgroundColor: 'var(--portal-sidebar-surface)',
    // Crisp left accent bar in the signature portal amber
    boxShadow: 'inset var(--portal-accent-bar-width) 0 0 var(--portal-accent)',
    '& .MuiListItemIcon-root': {
      color: 'var(--portal-accent)',
    },
    '& .MuiListItemText-primary': {
      color: 'var(--portal-sidebar-text-active)',
      fontWeight: 600,
    },
  }),
}));

export const StyledList = styled(List)(() => ({
  display: 'flex',
  padding: '24px 12px 0 12px',
  flexDirection: 'column',
  alignItems: 'flex-start',
  flex: '1 0 0',
  alignSelf: 'stretch',
}));

export const StyledSidebarFooterListItem = styled(ListItem)(() => ({
  display: 'flex',
  padding: '0.5rem 0.75rem',
  justifyContent: 'flex-start',
  alignItems: 'center',
  gap: '0.75rem',
  alignSelf: 'stretch',
  '& a': {
    textDecoration: 'none',
    '&:hover': {
      textDecoration: 'underline',
      textDecorationThickness: '1px',
      textUnderlineOffset: '2px',
    },
  },
}));

export const StyledSidebarFooterText = styled(Typography)(() => ({
  color: 'var(--portal-sidebar-text)',
  fontFeatureSettings: "'ss01' on, 'ss05' on, 'ss06' on, 'liga' off, 'clig' off",
  fontFamily: 'Mona Sans, sans-serif',
  fontSize: '12px',
  fontStyle: 'normal',
  fontWeight: 400,
  lineHeight: '150%',
}));

export const StyledSidebarFooterList = styled(List)(() => ({
  marginTop: 'auto',
  borderTop: '1px solid var(--portal-sidebar-border)',
  padding: '1rem',
}));
