import React from 'react';
import PropTypes from 'prop-types';
import { Box } from '@mui/material';
import { formatJsonPretty } from '../../utils/formatters';

export function JsonBlock({ value }) {
  const text = formatJsonPretty(value);
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
