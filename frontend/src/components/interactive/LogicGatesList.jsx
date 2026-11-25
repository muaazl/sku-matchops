import React from 'react';
import PropTypes from 'prop-types';
import { Paper, Stack, Typography } from '@mui/material';

export function LogicGatesList({ gates }) {
  return (
    <Stack spacing={2}>
      {(gates || []).map((g, idx) => (
        <Paper key={idx} variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" fontWeight={700}>
            Rank #{g.rank}: {g.candidate_name} ({g.status})
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Rationale: {g.reasons}
          </Typography>
        </Paper>
      ))}
    </Stack>
  );
}

LogicGatesList.propTypes = {
  gates: PropTypes.array,
};
