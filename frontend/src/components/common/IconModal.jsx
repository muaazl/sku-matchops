import React from 'react';
import PropTypes from 'prop-types';
import { Box, Dialog, IconButton, Stack, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Close } from '@mui/icons-material';

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
          <Close sx={{ fontSize: 18 }} />
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
            {React.isValidElement(Icon) ? (
              Icon
            ) : (
              <Icon sx={{ fontSize: 32, color: iconColor || theme.palette.text.secondary }} />
            )}
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

      {children && <Box sx={{ textAlign: 'left', mt: 3 }}>{children}</Box>}

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
  icon: PropTypes.oneOfType([PropTypes.elementType, PropTypes.node]),
  iconColor: PropTypes.string,
  title: PropTypes.node.isRequired,
  description: PropTypes.node,
  children: PropTypes.node,
  actions: PropTypes.node,
  maxWidth: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  hideCloseButton: PropTypes.bool,
};
