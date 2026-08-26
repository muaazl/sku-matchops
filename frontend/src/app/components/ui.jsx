import React from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Drawer,
  IconButton,
  Typography,
} from '@mui/material';
import { X } from 'lucide-react';

// Map a job/request status to an MUI chip color.
const STATUS_COLOR = {
  completed: 'success',
  done: 'success',
  running: 'info',
  queued: 'default',
  pending: 'warning',
  failed: 'error',
  cancelled: 'default',
  approved: 'success',
  rejected: 'error',
  matched: 'success',
  review: 'warning',
  confident: 'success',
};

export function StatusChip({ status, ...props }) {
  const color = STATUS_COLOR[status] ?? 'default';
  return (
    <Chip
      size="small"
      label={status}
      color={color}
      variant={color === 'default' ? 'outlined' : 'filled'}
      sx={{ textTransform: 'capitalize', fontWeight: 500, ...(props.sx || {}) }}
      {...props}
    />
  );
}

StatusChip.propTypes = {
  status: PropTypes.string.isRequired,
  sx: PropTypes.object,
};

export function HttpStatusChip({ code }) {
  const color = code >= 500 ? 'error' : code >= 400 ? 'warning' : code >= 200 && code < 300 ? 'success' : 'default';
  return <Chip size="small" label={code} color={color} sx={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }} />;
}

HttpStatusChip.propTypes = { code: PropTypes.number.isRequired };

export function PageHeader({ title, subtitle, actions }) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 2,
        mb: 3,
      }}
    >
      <Box>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {subtitle}
          </Typography>
        )}
      </Box>
      {actions && <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>{actions}</Box>}
    </Box>
  );
}

PageHeader.propTypes = {
  title: PropTypes.node.isRequired,
  subtitle: PropTypes.node,
  actions: PropTypes.node,
};

// Consistent page container that leaves room under the fixed AppBar.
export function PageContainer({ children }) {
  return <Box sx={{ p: { xs: 2, md: 4 }, maxWidth: 1400, mx: 'auto' }}>{children}</Box>;
}

PageContainer.propTypes = { children: PropTypes.node };

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
              <X size={20} />
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

export function ConfirmDialog({
  open,
  onClose,
  title,
  message,
  onConfirm,
  confirmText = 'Yes',
  cancelText = 'No',
  confirmColor = 'primary',
}) {
  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <DialogContentText sx={{ color: 'text.primary', mt: 1 }}>{message}</DialogContentText>
      </DialogContent>
      <DialogActions sx={{ p: 2, pt: 0 }}>
        <Button onClick={onClose} color="inherit">
          {cancelText}
        </Button>
        <Button variant="contained" color={confirmColor} onClick={onConfirm} autoFocus>
          {confirmText}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

ConfirmDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  title: PropTypes.string.isRequired,
  message: PropTypes.node.isRequired,
  onConfirm: PropTypes.func.isRequired,
  confirmText: PropTypes.string,
  cancelText: PropTypes.string,
  confirmColor: PropTypes.string,
};
