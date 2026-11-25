import React from 'react';
import PropTypes from 'prop-types';
import { Divider, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import {
  AutoFixHigh,
  Checklist,
  CloudUpload,
  Dashboard,
  Favorite,
  History,
  Inventory2,
  Search,
  Sensors,
  Terminal,
  Tune,
} from '@mui/icons-material';
import { Logo } from '../common/Logo';
import {
  Drawer,
  DrawerHeader,
  StyledList,
  StyledListItemButton,
  StyledSidebarFooterList,
  StyledSidebarFooterListItem,
  StyledSidebarFooterText,
} from './SidebarStyled';

export const NAV_ITEMS = [
  { title: 'Dashboard', icon: Dashboard, to: '/dashboard' },
  { title: 'Jobs', icon: Checklist, to: '/jobs' },
  { title: 'Requests', icon: Sensors, to: '/requests' },
  { title: 'Collections', icon: Inventory2, to: '/collections' },
  { title: 'Process SKUs', icon: CloudUpload, to: '/process-skus' },
  { title: 'Interactive', icon: AutoFixHigh, to: '/interactive' },
  { title: 'SKU Results', icon: History, to: '/sku-results' },
  { title: 'Rules Engine', icon: Tune, to: '/rules' },
  { title: 'Catalog Search', icon: Search, to: '/catalog' },
  { title: 'Logs', icon: Terminal, to: '/logs' },
];

function SidebarItem({ title, icon, linkTo, active = false }) {
  return (
    <ListItem disablePadding sx={{ display: 'block' }}>
      <StyledListItemButton component={Link} to={linkTo} isActive={active}>
        <ListItemIcon sx={{ minWidth: 0, mr: 3, justifyContent: 'center' }}>{icon}</ListItemIcon>
        <ListItemText primary={title} />
      </StyledListItemButton>
    </ListItem>
  );
}

SidebarItem.propTypes = {
  title: PropTypes.string.isRequired,
  icon: PropTypes.element.isRequired,
  linkTo: PropTypes.string.isRequired,
  active: PropTypes.bool,
};

export default function SkuSidebar() {
  const location = useLocation();
  const isActive = (linkTo) => location.pathname === linkTo || location.pathname.startsWith(`${linkTo}/`);

  return (
    <Drawer variant="permanent">
      <DrawerHeader sx={{ justifyContent: 'start', paddingLeft: '24px', paddingRight: '24px' }}>
        <Logo width={120} />
      </DrawerHeader>
      <Divider />
      <StyledList>
        {NAV_ITEMS.map(({ title, icon: Icon, to }) => (
          <SidebarItem key={to} title={title} icon={<Icon sx={{ fontSize: 18 }} />} linkTo={to} active={isActive(to)} />
        ))}
      </StyledList>
      <StyledSidebarFooterList>
        <StyledSidebarFooterListItem>
          <StyledSidebarFooterText
            variant="caption"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '13px',
              fontWeight: 500,
              color: 'text.secondary',
            }}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              Made with
              <Favorite sx={{ fontSize: 18, color: '#ef4444' }} />
              for
            </span>
            <img
              src="/pickme-logo.png"
              alt="PickMe"
              style={{
                height: '20px',
                width: '20px',
                display: 'block',
              }}
            />
          </StyledSidebarFooterText>
        </StyledSidebarFooterListItem>
      </StyledSidebarFooterList>
    </Drawer>
  );
}
