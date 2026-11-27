import React from 'react';
import PropTypes from 'prop-types';
import { Button, CircularProgress } from '@mui/material';
import { CheckCircle, PlayArrow } from '@mui/icons-material';

export function ModelStatusButton({ isLoading, isLoaded, onLoadModels }) {
  if (isLoading) {
    return (
      <Button
        variant="contained"
        size="small"
        color="primary"
        disabled
        startIcon={<CircularProgress size={14} color="inherit" />}
        sx={{
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: '6px',
          boxShadow: 'none',
        }}
      >
        Loading Models…
      </Button>
    );
  }

  if (isLoaded) {
    return (
      <Button
        variant="outlined"
        size="small"
        color="success"
        disabled
        startIcon={<CheckCircle sx={{ fontSize: 14 }} />}
        sx={{
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: '6px',
          pointerEvents: 'none',
          '&.Mui-disabled': {
            borderColor: 'success.main',
            color: 'success.main',
            opacity: 1,
          },
        }}
      >
        Models Loaded
      </Button>
    );
  }

  return (
    <Button
      variant="contained"
      size="small"
      color="primary"
      onClick={onLoadModels}
      startIcon={<PlayArrow sx={{ fontSize: 14 }} />}
      sx={{
        textTransform: 'none',
        fontWeight: 600,
        borderRadius: '6px',
        boxShadow: 'none',
        '&:hover': { boxShadow: 'none' },
      }}
    >
      Load Models
    </Button>
  );
}

ModelStatusButton.propTypes = {
  isLoading: PropTypes.bool.isRequired,
  isLoaded: PropTypes.bool.isRequired,
  onLoadModels: PropTypes.func.isRequired,
};
