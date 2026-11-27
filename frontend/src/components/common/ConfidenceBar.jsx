import React from 'react';
import PropTypes from 'prop-types';
import { Box, LinearProgress, Typography } from '@mui/material';
import { normalizeConfidencePct } from '../../utils/formatters';
import { StatusChip } from './StatusChip';

export function ConfidenceBar({ label, value, status, source, colorize = true }) {
  const pct = normalizeConfidencePct(value);
  const progressColor = colorize ? (pct >= 85 ? 'success' : pct >= 60 ? 'warning' : 'error') : 'primary';

  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.3, alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {source && (
            <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
              ({source})
            </Typography>
          )}
          <Typography variant="body2">{pct}%</Typography>
          {status && <StatusChip status={status} />}
        </Box>
      </Box>
      <LinearProgress variant="determinate" value={pct} color={progressColor} sx={{ height: 6, borderRadius: 3 }} />
    </Box>
  );
}

ConfidenceBar.propTypes = {
  label: PropTypes.node.isRequired,
  value: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  status: PropTypes.string,
  source: PropTypes.string,
  colorize: PropTypes.bool,
};
