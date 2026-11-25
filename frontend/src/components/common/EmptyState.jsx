import React from 'react';
import PropTypes from 'prop-types';
import { Box, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Inbox } from '@mui/icons-material';

export function EmptyState({ icon: Icon = Inbox, title, description, action, minHeight = 220, sx = {} }) {
  const theme = useTheme();

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        p: 3,
        minHeight,
        ...sx,
      }}
    >
      <Box
        sx={{
          width: 52,
          height: 52,
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.04)',
          color: 'text.secondary',
          mb: 2,
        }}
      >
        {React.isValidElement(Icon) ? Icon : <Icon sx={{ fontSize: 26 }} />}
      </Box>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: description ? 0.75 : 0 }}>
        {title}
      </Typography>
      {description && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ maxWidth: 440, mb: action ? 2.5 : 0, lineHeight: 1.5 }}
        >
          {description}
        </Typography>
      )}
      {action && <Box sx={{ mt: description ? 0 : 2 }}>{action}</Box>}
    </Box>
  );
}

EmptyState.propTypes = {
  icon: PropTypes.oneOfType([PropTypes.elementType, PropTypes.node]),
  title: PropTypes.node.isRequired,
  description: PropTypes.node,
  action: PropTypes.node,
  minHeight: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  sx: PropTypes.object,
};
