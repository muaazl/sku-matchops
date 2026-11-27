import React from 'react';
import PropTypes from 'prop-types';
import { Box, Divider, Paper, Stack, Typography } from '@mui/material';
import { JsonBlock } from '../common/JsonBlock';
import { CandidatePoolTable } from './CandidatePoolTable';
import { LogicGatesList } from './LogicGatesList';
import { StageNlpSummary } from './StageNlpSummary';

export function MatcherAuditTabs({ auditTrace, activeTab }) {
  if (activeTab === 0) {
    const retrieval = auditTrace.stage2_candidate_retrieval || {};
    return (
      <Stack spacing={3}>
        <StageNlpSummary auditTrace={auditTrace} />
        <Divider />
        <Box>
          <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
            Matcher Search Strategy
          </Typography>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="body2">
              <strong>Strategy:</strong> {retrieval.search_strategy || 'Hybrid Query'} |{' '}
              <strong>Candidates Retrieved:</strong> {retrieval.total_candidates_found || 0}
            </Typography>
          </Paper>
        </Box>
      </Stack>
    );
  }

  if (activeTab === 1) {
    return <CandidatePoolTable candidates={auditTrace.stage3_cross_encoder?.top_candidates} showRawScore />;
  }

  if (activeTab === 2) {
    return <LogicGatesList gates={auditTrace.stage4_logic_gates} />;
  }

  return <JsonBlock value={auditTrace} />;
}

MatcherAuditTabs.propTypes = {
  auditTrace: PropTypes.object.isRequired,
  activeTab: PropTypes.number.isRequired,
};
