import React from 'react';
import PropTypes from 'prop-types';
import { Button } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { HelpOutline, WarningAmber } from '@mui/icons-material';
import { IconModal } from './IconModal';

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
      icon={isDestructive ? WarningAmber : HelpOutline}
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
