import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, Chip, TextField, Typography } from '@mui/material';
import { AutoAwesome, PlayArrow, Science } from '@mui/icons-material';
import { JsonBlock } from '../common/JsonBlock';

export function RuleTesterPanel({ sample, onSampleChange, onAutoGenerate, onRunTest, isTesting, testResult }) {
  return (
    <>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Science sx={{ fontSize: 16 }} /> Test against sample
        </Typography>
        <Button
          size="small"
          variant="text"
          startIcon={<AutoAwesome sx={{ fontSize: 14 }} />}
          onClick={onAutoGenerate}
          sx={{ textTransform: 'none', fontWeight: 600 }}
        >
          Auto-generate sample
        </Button>
      </Box>
      <TextField
        multiline
        minRows={3}
        fullWidth
        size="small"
        value={sample}
        onChange={(e) => onSampleChange(e.target.value)}
        sx={{ fontFamily: 'monospace', '& textarea': { fontFamily: 'monospace', fontSize: 13 } }}
      />
      <Button
        variant="outlined"
        color="inherit"
        disabled={isTesting}
        startIcon={<PlayArrow sx={{ fontSize: 14 }} />}
        onClick={onRunTest}
        sx={{ mt: 1.5 }}
      >
        Run test
      </Button>
      {testResult && (
        <Box sx={{ mt: 2 }}>
          <Chip
            size="small"
            label={testResult.fires ? 'Rule fires ✓' : 'Rule does not fire'}
            color={testResult.fires ? 'success' : 'default'}
            sx={{ mb: 1 }}
          />
          <JsonBlock value={testResult} />
        </Box>
      )}
    </>
  );
}

RuleTesterPanel.propTypes = {
  sample: PropTypes.string.isRequired,
  onSampleChange: PropTypes.func.isRequired,
  onAutoGenerate: PropTypes.func.isRequired,
  onRunTest: PropTypes.func.isRequired,
  isTesting: PropTypes.bool.isRequired,
  testResult: PropTypes.object,
};
