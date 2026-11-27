import React from 'react';
import PropTypes from 'prop-types';
import { Chip } from '@mui/material';
import { STATUS_COLOR_MAP } from '../../utils/constants';

export function StatusChip({ status, ...props }) {
  const color = STATUS_COLOR_MAP[status] ?? 'default';
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
