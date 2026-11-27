import React from 'react';
import PropTypes from 'prop-types';
import { Box } from '@mui/material';

export function PageContainer({ children }) {
  return <Box sx={{ p: { xs: 2, md: 4 }, maxWidth: 1400, mx: 'auto' }}>{children}</Box>;
}

PageContainer.propTypes = {
  children: PropTypes.node,
};
