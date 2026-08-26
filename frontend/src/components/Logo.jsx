import React from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { getFullPath } from '../lib/common-helpers';

export const Logo = ({ width = '100px', ...props }) => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const logoFile = isDark ? 'logo-red-white.svg' : 'logo-red-black.svg';
  const logoUrl = getFullPath(logoFile);

  return <img src={logoUrl} alt="logo" width={width} {...props} />;
};

Logo.propTypes = {
  width: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
};
