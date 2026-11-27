import React from 'react';
import PropTypes from 'prop-types';
import { Chip } from '@mui/material';

export function HttpStatusChip({ code }) {
  const color = code >= 500 ? 'error' : code >= 400 ? 'warning' : code >= 200 && code < 300 ? 'success' : 'default';
  return <Chip size="small" label={code} color={color} sx={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }} />;
}

HttpStatusChip.propTypes = {
  code: PropTypes.number.isRequired,
};
