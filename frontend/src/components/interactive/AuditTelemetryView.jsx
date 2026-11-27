import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Box, Button, Paper, Stack, Tab, Tabs } from '@mui/material';
import { ArrowBack } from '@mui/icons-material';
import { PageContainer } from '../common/PageContainer';
import { PageHeader } from '../common/PageHeader';
import { AuditSummaryCard } from './AuditSummaryCard';
import { ClassifierAuditTabs } from './ClassifierAuditTabs';
import { MatcherAuditTabs } from './MatcherAuditTabs';
import { PipelineAuditTabs } from './PipelineAuditTabs';

const CLASSIFIER_TABS = [
  '1. Classification Summary & NLP',
  '2. Keywords Considered Pool',
  '3. Candidate Basic Types & Region Scoring',
  '4. Raw API JSON Inspector',
];

const MATCHER_TABS = [
  '1. Matcher Overview & NLP',
  '2. Candidates & Cross-Encoder Pool',
  '3. Logic Gates & Winner Evaluation',
  '4. Raw API JSON Inspector',
];

const PIPELINE_TABS = [
  '1. Pipeline Overview & Decisions',
  '2. Candidates & Scoring Pool',
  '3. Logic Gates & Rejection Audit',
  '4. Classifier Fallback',
  '5. Template & Rules Engine',
  '6. Raw API JSON Inspector',
];

export default function AuditTelemetryView({ auditTrace, sku, domain, task, resultTask, onBack }) {
  const [auditTab, setAuditTab] = useState(0);

  if (!auditTrace) return null;

  const auditTask = auditTrace?.input?.task || resultTask || task;
  const isClassifierAudit = auditTask === 'classifier';
  const isMatcherAudit = auditTask === 'matcher';
  const isPipelineAudit = auditTask === 'pipeline';

  const tabLabels = isClassifierAudit ? CLASSIFIER_TABS : isMatcherAudit ? MATCHER_TABS : PIPELINE_TABS;

  return (
    <PageContainer>
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
        <Button startIcon={<ArrowBack sx={{ fontSize: 16 }} />} onClick={onBack} color="inherit" size="small">
          Back to Interactive Execution
        </Button>
      </Box>

      <PageHeader
        title="SKU Diagnostic Audit & Pipeline Telemetry"
        subtitle={
          'Deep multi-stage inspection of candidates considered, scoring, logic gates, ' +
          'template enrichment, and business rules.'
        }
      />

      <Stack spacing={3}>
        <AuditSummaryCard
          auditTrace={auditTrace}
          sku={sku}
          domain={domain}
          auditTask={auditTask}
          isClassifierAudit={isClassifierAudit}
        />

        <Paper variant="outlined">
          <Tabs
            value={auditTab}
            onChange={(e, val) => setAuditTab(val)}
            indicatorColor="primary"
            textColor="primary"
            variant="scrollable"
            scrollButtons="auto"
          >
            {tabLabels.map((label) => (
              <Tab key={label} label={label} />
            ))}
          </Tabs>

          <Box sx={{ p: 3 }}>
            {isClassifierAudit && <ClassifierAuditTabs auditTrace={auditTrace} activeTab={auditTab} />}
            {isMatcherAudit && <MatcherAuditTabs auditTrace={auditTrace} activeTab={auditTab} />}
            {isPipelineAudit && <PipelineAuditTabs auditTrace={auditTrace} activeTab={auditTab} />}
          </Box>
        </Paper>
      </Stack>
    </PageContainer>
  );
}

AuditTelemetryView.propTypes = {
  auditTrace: PropTypes.object,
  sku: PropTypes.string,
  domain: PropTypes.string,
  task: PropTypes.string,
  resultTask: PropTypes.string,
  onBack: PropTypes.func.isRequired,
};
