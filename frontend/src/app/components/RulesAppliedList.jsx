import React from 'react';
import PropTypes from 'prop-types';
import { Box, Chip, Stack, Typography } from '@mui/material';

export default function RulesAppliedList({ rules }) {
  if (!rules) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
        No rules were applied.
      </Typography>
    );
  }

  let parsed = rules;
  if (typeof rules === 'string') {
    try {
      parsed = JSON.parse(rules);
    } catch {
      return (
        <Box
          component="pre"
          sx={{
            m: 0,
            p: 1.5,
            borderRadius: 1,
            border: (t) => `1px solid ${t.palette.divider}`,
            bgcolor: (t) => t.palette.background.default,
            fontFamily: 'monospace',
            fontSize: 13,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {String(rules)}
        </Box>
      );
    }
  }

  if (!Array.isArray(parsed) || parsed.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
        No rules were applied.
      </Typography>
    );
  }

  return (
    <Stack spacing={1.5}>
      {parsed.map((rule, idx) => (
        <Box
          key={rule.rule_id || idx}
          sx={{
            pl: 1.5,
            borderLeft: '3px solid',
            borderColor: 'primary.main',
            py: 0.5,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.3 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {rule.rule_id}
            </Typography>
            {rule.module && (
              <Chip label={rule.module} size="small" variant="outlined" sx={{ height: 20, fontSize: '0.7rem' }} />
            )}
          </Box>
          {rule.description && (
            <Typography variant="body2" color="text.secondary">
              {rule.description}
            </Typography>
          )}
          {rule.change && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
              Changed:{' '}
              {typeof rule.change === 'string'
                ? rule.change
                : Object.entries(rule.change)
                    .map(([k, v]) => `${k} \u2192 ${typeof v === 'string' ? v : JSON.stringify(v)}`)
                    .join(', ')}
            </Typography>
          )}
          {rule.reasoning && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: 'block', fontStyle: 'italic', mt: 0.3 }}
            >
              {rule.reasoning}
            </Typography>
          )}
        </Box>
      ))}
    </Stack>
  );
}

RulesAppliedList.propTypes = {
  rules: PropTypes.oneOfType([PropTypes.string, PropTypes.array, PropTypes.object]),
};
