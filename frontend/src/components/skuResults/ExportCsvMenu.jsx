import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Button, Divider, ListItemIcon, ListItemText, Menu, MenuItem } from '@mui/material';
import { Download, KeyboardArrowDown, TableChart, VerifiedUser } from '@mui/icons-material';

export function ExportCsvMenu({ onExport }) {
  const [anchorEl, setAnchorEl] = useState(null);

  return (
    <>
      <Button
        variant="outlined"
        color="inherit"
        startIcon={<Download sx={{ fontSize: 15 }} />}
        endIcon={<KeyboardArrowDown sx={{ fontSize: 14 }} />}
        onClick={(e) => setAnchorEl(e.currentTarget)}
        aria-controls={anchorEl ? 'export-csv-menu' : undefined}
        aria-haspopup="true"
        aria-expanded={anchorEl ? 'true' : undefined}
      >
        Download CSV
      </Button>
      <Menu
        id="export-csv-menu"
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: { minWidth: 280, p: 0.5, mt: 0.5, boxShadow: 3 },
          },
        }}
      >
        <MenuItem
          onClick={() => {
            setAnchorEl(null);
            onExport('clean');
          }}
          sx={{ py: 1, borderRadius: 1 }}
        >
          <ListItemIcon sx={{ minWidth: 34, color: 'primary.main' }}>
            <TableChart sx={{ fontSize: 18 }} />
          </ListItemIcon>
          <ListItemText
            primary="Clean Version (CSV)"
            secondary="Table columns only (for Google Sheets)"
            primaryTypographyProps={{ variant: 'body2', fontWeight: 600 }}
            secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
          />
        </MenuItem>
        <Divider sx={{ my: 0.5 }} />
        <MenuItem
          onClick={() => {
            setAnchorEl(null);
            onExport('audit');
          }}
          sx={{ py: 1, borderRadius: 1 }}
        >
          <ListItemIcon sx={{ minWidth: 34, color: 'text.secondary' }}>
            <VerifiedUser sx={{ fontSize: 18 }} />
          </ListItemIcon>
          <ListItemText
            primary="Audit Version (CSV)"
            secondary="Full metadata, scores & confidence"
            primaryTypographyProps={{ variant: 'body2', fontWeight: 600 }}
            secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
          />
        </MenuItem>
      </Menu>
    </>
  );
}

ExportCsvMenu.propTypes = {
  onExport: PropTypes.func.isRequired,
};
