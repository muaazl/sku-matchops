import React from 'react';
import PropTypes from 'prop-types';
import { Skeleton } from '@mui/material';

export function CardSkeleton({ height = 120, sx = {} }) {
  return <Skeleton variant="rounded" height={height} sx={{ borderRadius: 2, ...sx }} />;
}

CardSkeleton.propTypes = {
  height: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  sx: PropTypes.object,
};
