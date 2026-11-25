import React from 'react';
import PropTypes from 'prop-types';
import { Box, Drawer, IconButton, Typography } from '@mui/material';
import { Close } from '@mui/icons-material';

export function SideDrawer({ open, onClose, title, children, width = 480 }) {
  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: { xs: '100%', sm: width } } }}>
      {open && (
        <Box sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              {title}
            </Typography>
            <IconButton onClick={onClose} size="small" aria-label="Close drawer">
              <Close sx={{ fontSize: 20 }} />
            </IconButton>
          </Box>
          <Box sx={{ flexGrow: 1, overflowY: 'auto' }}>{children}</Box>
        </Box>
      )}
    </Drawer>
  );
}

SideDrawer.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  title: PropTypes.node.isRequired,
  children: PropTypes.node,
  width: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
};
