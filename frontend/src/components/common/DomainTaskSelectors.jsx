import React from 'react';
import PropTypes from 'prop-types';
import { Box, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { DOMAINS } from '../../utils/constants';

export function DomainSelector({ value, onChange, ariaLabel = 'Select domain' }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
        Domain
      </Typography>
      <ToggleButtonGroup
        size="small"
        exclusive
        value={value}
        onChange={(_, v) => v && onChange(v)}
        aria-label={ariaLabel}
        fullWidth
      >
        {DOMAINS.map((d) => (
          <ToggleButton key={d} value={d} sx={{ textTransform: 'capitalize' }}>
            {d}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
}

DomainSelector.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  ariaLabel: PropTypes.string,
};

export function TaskSelector({ value, onChange, ariaLabel = 'Select task' }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
        Task
      </Typography>
      <ToggleButtonGroup
        size="small"
        exclusive
        value={value}
        onChange={(_, v) => v && onChange(v)}
        aria-label={ariaLabel}
        fullWidth
      >
        <ToggleButton value="matcher" sx={{ textTransform: 'none' }}>
          Matcher
        </ToggleButton>
        <ToggleButton value="classifier" sx={{ textTransform: 'none' }}>
          Classifier
        </ToggleButton>
        <ToggleButton value="pipeline" sx={{ textTransform: 'none' }}>
          Pipeline
        </ToggleButton>
      </ToggleButtonGroup>
    </Box>
  );
}

TaskSelector.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  ariaLabel: PropTypes.string,
};
