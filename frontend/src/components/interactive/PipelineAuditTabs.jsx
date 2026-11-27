import React from 'react';
import PropTypes from 'prop-types';
import { Box, Divider, Paper, Stack, Typography } from '@mui/material';
import { JsonBlock } from '../common/JsonBlock';
import RulesAppliedList from '../common/RulesAppliedList';
import { CandidatePoolTable } from './CandidatePoolTable';
import { LogicGatesList } from './LogicGatesList';
import { StageNlpSummary } from './StageNlpSummary';

export function PipelineAuditTabs({ auditTrace, activeTab }) {
  if (activeTab === 0) {
    return (
      <Stack spacing={3}>
        <StageNlpSummary auditTrace={auditTrace} />
        <Divider />
        <Box>
          <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
            Stage 6: Pipeline Escalation Decision
          </Typography>
          <Paper variant="outlined" sx={{ p: 2 }}>
            {auditTrace.stage6_escalation?.escalated ? (
              <Stack spacing={1}>
                <Typography variant="body2" color="warning.main" fontWeight={600}>
                  Escalated to Classifier: {auditTrace.stage6_escalation?.escalation_reason}
                </Typography>
                <Typography variant="body2">
                  <strong>Winner:</strong> {auditTrace.stage6_escalation?.escalation_winner} |{' '}
                  <strong>Classifier BT:</strong> {auditTrace.stage6_escalation?.classifier_bt} (
                  {Math.round((auditTrace.stage6_escalation?.classifier_bt_confidence || 0) * 100)}%)
                </Typography>
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No escalation required ({auditTrace.stage6_escalation?.note || 'Normal matching'}).
              </Typography>
            )}
          </Paper>
        </Box>
      </Stack>
    );
  }

  if (activeTab === 1) {
    return <CandidatePoolTable candidates={auditTrace.stage3_cross_encoder?.top_candidates} />;
  }

  if (activeTab === 2) {
    return <LogicGatesList gates={auditTrace.stage4_logic_gates} />;
  }

  if (activeTab === 3) {
    return (
      <Paper variant="outlined" sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1.5 }}>
          Classifier Fallback Telemetry
        </Typography>
        {auditTrace.stage6_escalation?.escalated ? (
          <Stack spacing={1}>
            <Typography variant="body2">
              <strong>Predicted BT:</strong> {auditTrace.stage6_escalation.classifier_bt} (Conf:{' '}
              {auditTrace.stage6_escalation.classifier_bt_confidence?.toFixed(4)})
            </Typography>
            <Typography variant="body2">
              <strong>Predicted GK:</strong> {auditTrace.stage6_escalation.classifier_gk}
            </Typography>
            <Typography variant="body2">
              <strong>Predicted Region:</strong> {auditTrace.stage6_escalation.classifier_region}
            </Typography>
          </Stack>
        ) : (
          <Typography variant="body2" color="text.secondary">
            Classifier fallback was not triggered for this SKU item.
          </Typography>
        )}
      </Paper>
    );
  }

  if (activeTab === 4) {
    return (
      <Stack spacing={2}>
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
            Stage 7: Template Tag Enrichment
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {auditTrace.stage7_template_enrichment?.notes ||
              auditTrace.stage7_template_enrichment?.note ||
              'No template rule matched.'}
          </Typography>
        </Paper>
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
            Stage 8: Business Rules Engine
          </Typography>
          <RulesAppliedList rules={auditTrace.stage8_rules_engine?.rules_applied} />
        </Paper>
      </Stack>
    );
  }

  return <JsonBlock value={auditTrace} />;
}

PipelineAuditTabs.propTypes = {
  auditTrace: PropTypes.object.isRequired,
  activeTab: PropTypes.number.isRequired,
};
