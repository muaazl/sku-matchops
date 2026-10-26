import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, Chip, Dialog, Drawer, IconButton, Stack, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { AlertTriangle, HelpCircle, X } from 'lucide-react';

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

// Shared visual language for "simple" modals across the app — an icon badge, a centered
// title/description, optional extra content, and a row of actions. Modeled after
// ServerOfflineModal's look (icon-in-circle, rounded bordered card, centered text) so
// every confirmation/prompt dialog in the app reads as the same component family, while
// still using MUI's Dialog underneath for proper focus-trap/ESC/backdrop-click behavior.
export function IconModal({
  open,
  onClose,
  icon: Icon,
  iconColor,
  title,
  description,
  children,
  actions,
  maxWidth = 420,
  hideCloseButton = false,
}) {
  const theme = useTheme();
  return (
    <Dialog
      open={open}
      onClose={onClose}
      slotProps={{
        backdrop: {
          sx: {
            backdropFilter: 'blur(8px)',
            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(0, 0, 0, 0.6)' : 'rgba(0, 0, 0, 0.3)',
          },
        },
        paper: {
          sx: {
            maxWidth,
            width: '100%',
            borderRadius: 2,
            border: `1px solid ${theme.palette.divider}`,
            p: 4,
            position: 'relative',
          },
        },
      }}
    >
      {!hideCloseButton && onClose && (
        <IconButton
          onClick={onClose}
          size="small"
          aria-label="Close dialog"
          sx={{ position: 'absolute', top: 12, right: 12, color: 'text.secondary' }}
        >
          <X size={18} />
        </IconButton>
      )}

      <Box sx={{ textAlign: 'center' }}>
        {Icon && (
          <Box
            sx={{
              display: 'inline-flex',
              p: 1.5,
              borderRadius: '50%',
              backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
              mb: 2,
            }}
          >
            <Icon size={32} color={iconColor || theme.palette.text.secondary} />
          </Box>
        )}

        <Typography variant="h6" sx={{ fontWeight: 600, mb: description ? 1 : 0 }}>
          {title}
        </Typography>

        {description && (
          <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.5 }}>
            {description}
          </Typography>
        )}
      </Box>

      {children && (
        <Box sx={{ textAlign: 'left', mt: 3 }}>{children}</Box>
      )}

      {actions && (
        <Stack direction="row" spacing={1.5} sx={{ mt: 3 }}>
          {actions}
        </Stack>
      )}
    </Dialog>
  );
}

IconModal.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func,
  icon: PropTypes.elementType,
  iconColor: PropTypes.string,
  title: PropTypes.node.isRequired,
  description: PropTypes.node,
  children: PropTypes.node,
  actions: PropTypes.node,
  maxWidth: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  hideCloseButton: PropTypes.bool,
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
  const theme = useTheme();
  const isDestructive = confirmColor === 'error';
  return (
    <IconModal
      open={open}
      onClose={onClose}
      icon={isDestructive ? AlertTriangle : HelpCircle}
      iconColor={isDestructive ? theme.palette.error.main : undefined}
      title={title}
      description={message}
      actions={
        <>
          <Button onClick={onClose} color="inherit" variant="outlined" fullWidth>
            {cancelText}
          </Button>
          <Button variant="contained" color={confirmColor} onClick={onConfirm} autoFocus fullWidth>
            {confirmText}
          </Button>
        </>
      }
    />
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
