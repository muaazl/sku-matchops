import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Button, IconButton, InputAdornment, TextField } from '@mui/material';
import { VerifiedUser, Visibility, VisibilityOff } from '@mui/icons-material';
import { IconModal } from '../common/IconModal';

export function BearerTokenModal({ open, onClose, token, onTokenChange, onSubmit }) {
  const [showToken, setShowToken] = useState(false);

  return (
    <IconModal
      open={open}
      onClose={onClose}
      icon={VerifiedUser}
      title="Enter Bearer Token"
      description={
        "The bearer token is kept only for this browser tab's session (never written to disk) " +
        'and is cleared automatically when you close the tab, or immediately if the portal reports it ' +
        'as invalid, blacklisted, or expired.'
      }
      actions={
        <>
          <Button onClick={onClose} color="inherit" variant="outlined" fullWidth>
            Cancel
          </Button>
          <Button variant="contained" fullWidth onClick={onSubmit} disabled={!token.trim()}>
            Confirm &amp; Fetch
          </Button>
        </>
      }
    >
      <TextField
        autoFocus
        label="Bearer token"
        type={showToken ? 'text' : 'password'}
        value={token}
        onChange={(e) => onTokenChange(e.target.value)}
        placeholder="••••••••••••"
        fullWidth
        InputProps={{
          endAdornment: (
            <InputAdornment position="end">
              <IconButton
                onClick={() => setShowToken((s) => !s)}
                edge="end"
                size="small"
                aria-label="Toggle token visibility"
              >
                {showToken ? <VisibilityOff sx={{ fontSize: 17 }} /> : <Visibility sx={{ fontSize: 17 }} />}
              </IconButton>
            </InputAdornment>
          ),
        }}
      />
    </IconModal>
  );
}

BearerTokenModal.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  token: PropTypes.string.isRequired,
  onTokenChange: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
};
