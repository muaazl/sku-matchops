import React from 'react';
import PropTypes from 'prop-types';
import { Box } from '@mui/material';

// Lightweight, theme-aware JSON/code viewer.
export function JsonBlock({ value }) {
  let text = value;
  if (typeof value !== 'string') {
    text = JSON.stringify(value, null, 2);
  } else {
    try {
      text = JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      /* leave as-is if not valid JSON */
    }
  }
  return (
    <Box
      component="pre"
      sx={{
        m: 0,
        p: 2,
        borderRadius: 2,
        border: (t) => `1px solid ${t.palette.divider}`,
        bgcolor: (t) => t.palette.background.code || t.palette.background.default,
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: 13,
        lineHeight: 1.5,
        overflowX: 'auto',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}
    >
      {text}
    </Box>
  );
}

JsonBlock.propTypes = {
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.object, PropTypes.array]),
};
