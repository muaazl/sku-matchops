import React from 'react';
import PropTypes from 'prop-types';
import { Divider, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  ListChecks,
  Radio,
  Boxes,
  Upload,
  Wand2,
  History,
  SlidersHorizontal,
  Search,
  Terminal,
  Heart,
} from 'lucide-react';
import {
  DrawerHeader,
  Drawer,
  StyledListItemButton,
  StyledList,
  StyledSidebarFooterListItem,
  StyledSidebarFooterText,
  StyledSidebarFooterList,
} from '../components/Sidebar/SidebarStyled';
import { Logo } from '../components/Logo';

export const NAV_ITEMS = [
  { title: 'Dashboard', icon: LayoutDashboard, to: '/dashboard' },
  { title: 'Jobs', icon: ListChecks, to: '/jobs' },
  { title: 'Requests', icon: Radio, to: '/requests' },
  { title: 'Collections', icon: Boxes, to: '/collections' },
  { title: 'Process SKUs', icon: Upload, to: '/process-skus' },
  { title: 'Interactive', icon: Wand2, to: '/interactive' },
  { title: 'SKU Results', icon: History, to: '/sku-results' },
  { title: 'Rules Engine', icon: SlidersHorizontal, to: '/rules' },
  { title: 'Catalog Search', icon: Search, to: '/catalog' },
  { title: 'Logs', icon: Terminal, to: '/logs' },
];

export default function SkuSidebar() {
  const location = useLocation();
  const isActive = (linkTo) => location.pathname === linkTo || location.pathname.startsWith(linkTo + '/');

  return (
    <Drawer variant="permanent">
      <DrawerHeader sx={{ justifyContent: 'start', paddingLeft: '24px', paddingRight: '24px' }}>
        <Logo width={120} />
      </DrawerHeader>
      <Divider />
      <StyledList>
        {NAV_ITEMS.map(({ title, icon: Icon, to }) => (
          <SidebarItem key={to} title={title} icon={<Icon size="16px" />} linkTo={to} active={isActive(to)} />
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
              <Heart size={20} strokeWidth={2} color="#ef4444" fill="#ef4444" />
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
