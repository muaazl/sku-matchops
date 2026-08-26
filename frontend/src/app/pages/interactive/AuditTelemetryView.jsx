import React, { useState, useMemo } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Divider,
  Grid,
  InputAdornment,
  LinearProgress,
  Paper,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import { ArrowLeft, Search } from 'lucide-react';
import { PageContainer, PageHeader, StatusChip } from '../../components/ui';
import RulesAppliedList from '../../components/RulesAppliedList';
import { JsonBlock } from '../../components/JsonBlock';

function Confidence({ label, value, status, source }) {
  const num = typeof value === 'number' ? value : 0;
  const normalized = num > 1 ? num / 100 : num;
  const pct = Math.round(Math.max(0, Math.min(1, normalized)) * 100);
  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.3, alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {source && (
            <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
              ({source})
            </Typography>
          )}
          <Typography variant="body2">{pct}%</Typography>
          {status && <StatusChip status={status} />}
        </Box>
      </Box>
      <LinearProgress variant="determinate" value={pct} sx={{ height: 6, borderRadius: 3 }} />
    </Box>
  );
}

Confidence.propTypes = {
  label: PropTypes.node.isRequired,
  value: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  status: PropTypes.string,
  source: PropTypes.string,
};

export default function AuditTelemetryView({ auditTrace, sku, domain, task, resultTask, onBack }) {
  const [auditTab, setAuditTab] = useState(0);
  const [keywordSearch, setKeywordSearch] = useState('');

  const auditTask = auditTrace?.input?.task || resultTask || task;
  const isClassifierAudit = auditTask === 'classifier';
  const isMatcherAudit = auditTask === 'matcher';
  const isPipelineAudit = auditTask === 'pipeline';

  const filteredKeywords = useMemo(() => {
    const list = auditTrace?.stage3_gk_classification?.considered_keywords || [];
    if (!keywordSearch.trim()) return list;
    const q = keywordSearch.toLowerCase().trim();
    return list.filter(
      (k) =>
        k.tag?.toLowerCase().includes(q) ||
        k.source?.toLowerCase().includes(q) ||
        k.prune_reason?.toLowerCase().includes(q)
    );
  }, [auditTrace, keywordSearch]);

  if (!auditTrace) return null;

  return (
    <PageContainer>
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
        <Button startIcon={<ArrowLeft size={16} />} onClick={onBack} color="inherit" size="small">
          Back to Interactive Execution
        </Button>
      </Box>

      <PageHeader
        title="SKU Diagnostic Audit & Pipeline Telemetry"
        subtitle="Deep multi-stage inspection of candidates considered, scoring, logic gates, template enrichment, and business rules."
      />

      <Stack spacing={3}>
        <Card sx={{ borderLeft: '4px solid', borderColor: 'primary.main' }}>
          <CardContent sx={{ py: 2.5 }}>
            <Grid container spacing={3} alignItems="center">
              <Grid size={{ xs: 12, md: 4 }}>
                <Typography variant="caption" color="text.secondary" fontWeight={600}>
                  TARGET SKU NAME
                </Typography>
                <Typography variant="h6" fontWeight={700}>
                  {auditTrace.input?.sku_name || sku}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Chip
                    label={`Domain: ${(auditTrace.input?.domain || domain).toUpperCase()}`}
                    size="small"
                    color="primary"
                  />
                  <Chip label={`Task: ${auditTask}`} size="small" variant="outlined" />
                </Box>
              </Grid>

              {isClassifierAudit ? (
                <Grid size={{ xs: 12, md: 4 }}>
                  <Typography variant="caption" color="text.secondary" fontWeight={600}>
                    PREDICTED BASIC TYPE & REGION
                  </Typography>
                  <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body1" fontWeight={600}>
                        {auditTrace.final_output?.suggested_bt || '(None)'}
                      </Typography>
                      <StatusChip status={auditTrace.final_output?.bt_status || 'LOW'} />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      <strong>{auditTrace.stage4_third_tag_classification?.tag_label || 'Category'}:</strong>{' '}
                      {auditTrace.final_output?.suggested_region || '-'}
                    </Typography>
                  </Stack>
                </Grid>
              ) : (
                <Grid size={{ xs: 12, md: 4 }}>
                  <Typography variant="caption" color="text.secondary" fontWeight={600}>
                    MATCHED CATALOG ITEM
                  </Typography>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {auditTrace.final_output?.matched_catalog_name || '(No Match)'}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.5 }}>
                    <StatusChip status={auditTrace.final_output?.status || 'Low / Rejected'} />
                    <Typography variant="caption" color="text.secondary">
                      Score: {auditTrace.final_output?.score != null ? auditTrace.final_output.score.toFixed(2) : '0.0'}
                    </Typography>
                  </Box>
                </Grid>
              )}

              <Grid size={{ xs: 12, md: 4 }}>
                <Typography variant="caption" color="text.secondary" fontWeight={600}>
                  FINAL TAGS PRODUCED
                </Typography>
                <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                  <Typography variant="body2">
                    <strong>BT:</strong> {auditTrace.final_output?.suggested_bt || '-'}
                  </Typography>
                  <Typography variant="body2">
                    <strong>GK:</strong> {auditTrace.final_output?.suggested_gk || '-'}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Region/Cat:</strong> {auditTrace.final_output?.suggested_region || '-'}
                  </Typography>
                </Stack>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        <Paper variant="outlined">
          <Tabs
            value={auditTab}
            onChange={(e, val) => setAuditTab(val)}
            indicatorColor="primary"
            textColor="primary"
            variant="scrollable"
            scrollButtons="auto"
          >
            {isClassifierAudit && [
              <Tab key="c0" label="1. Classification Summary & NLP" />,
              <Tab key="c1" label="2. Keywords Considered Pool" />,
              <Tab key="c2" label="3. Candidate Basic Types & Region Scoring" />,
              <Tab key="c3" label="4. Raw API JSON Inspector" />,
            ]}

            {isMatcherAudit && [
              <Tab key="m0" label="1. Matcher Overview & NLP" />,
              <Tab key="m1" label="2. Candidates & Cross-Encoder Pool" />,
              <Tab key="m2" label="3. Logic Gates & Winner Evaluation" />,
              <Tab key="m3" label="4. Raw API JSON Inspector" />,
            ]}

            {isPipelineAudit && [
              <Tab key="p0" label="1. Pipeline Overview & Decisions" />,
              <Tab key="p1" label="2. Candidates & Scoring Pool" />,
              <Tab key="p2" label="3. Logic Gates & Rejection Audit" />,
              <Tab key="p3" label="4. Classifier Fallback" />,
              <Tab key="p4" label="5. Template & Rules Engine" />,
              <Tab key="p5" label="6. Raw API JSON Inspector" />,
            ]}
          </Tabs>

          <Box sx={{ p: 3 }}>
            {isClassifierAudit && (
              <>
                {auditTab === 0 && (
                  <Stack spacing={3}>
                    <Box>
                      <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1.5 }}>
                        Stage 1: NLP Normalization & Entity Extraction
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid size={{ xs: 12, sm: 6 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Clean Input
                            </Typography>
                            <Typography variant="body1" fontWeight={600}>
                              {auditTrace.stage1_nlp?.clean_input || '-'}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid size={{ xs: 12, sm: 6 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Weight Stripped Input
                            </Typography>
                            <Typography variant="body1" fontWeight={600}>
                              {auditTrace.stage1_nlp?.weight_stripped_input || '-'}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid size={{ xs: 12, sm: 4 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Extracted Weight Value
                            </Typography>
                            <Typography variant="body1" fontWeight={600}>
                              {auditTrace.stage1_nlp?.extracted_weights?.value != null
                                ? `${auditTrace.stage1_nlp.extracted_weights.value} ${auditTrace.stage1_nlp.extracted_weights.unit || ''
                                }`
                                : 'None'}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid size={{ xs: 12, sm: 8 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              NER Extracted Entities
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 0.5 }}>
                              {Object.entries(auditTrace.stage1_nlp?.ner_entities || {}).flatMap(([k, v]) =>
                                Array.isArray(v) && v.length > 0
                                  ? v.map((item, idx) => (
                                    <Chip
                                      key={`${k}-${idx}`}
                                      label={`${k}: ${item}`}
                                      size="small"
                                      variant="outlined"
                                    />
                                  ))
                                  : null
                              )}
                              {Object.values(auditTrace.stage1_nlp?.ner_entities || {}).every(
                                (v) => !Array.isArray(v) || v.length === 0
                              ) && (
                                  <Typography variant="body2" color="text.secondary">
                                    No entity tokens detected
                                  </Typography>
                                )}
                            </Box>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Box>

                    <Divider />

                    <Box>
                      <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
                        Stage 5: Business Rules Engine Post-Processing
                      </Typography>
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Rules Triggered:</strong> {auditTrace.stage5_rules_engine?.rules_applied_count || 0}
                        </Typography>
                        <RulesAppliedList rules={auditTrace.stage5_rules_engine?.rules_applied} />
                      </Paper>
                    </Box>
                  </Stack>
                )}

                {auditTab === 1 && (
                  <Stack spacing={3}>
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: 2,
                      }}
                    >
                      <Box>
                        <Typography variant="subtitle1" fontWeight={700} color="primary">
                          Generic Keywords Considered Pool
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Candidate keywords evaluated across schema, ML classifiers, and vector search.
                        </Typography>
                      </Box>
                      <TextField
                        size="small"
                        placeholder="Filter keywords..."
                        value={keywordSearch}
                        onChange={(e) => setKeywordSearch(e.target.value)}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <Search size={16} />
                            </InputAdornment>
                          ),
                        }}
                        sx={{ width: 220 }}
                      />
                    </Box>

                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 700 }}>Status</TableCell>
                            <TableCell sx={{ fontWeight: 700 }}>Keyword Tag</TableCell>
                            <TableCell sx={{ fontWeight: 700 }}>Source</TableCell>
                            <TableCell sx={{ fontWeight: 700 }} align="right">
                              Base Score
                            </TableCell>
                            <TableCell sx={{ fontWeight: 700 }} align="right">
                              Reranker Score
                            </TableCell>
                            <TableCell sx={{ fontWeight: 700 }}>Flavor Filter</TableCell>
                            <TableCell sx={{ fontWeight: 700 }}>Selection Rationale</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {filteredKeywords.map((k, idx) => {
                            const isSelected = k.is_selected;
                            return (
                              <TableRow key={idx}>
                                <TableCell>
                                  {isSelected ? (
                                    <Chip
                                      label="SELECTED"
                                      size="small"
                                      color="success"
                                      variant="outlined"
                                      sx={{ height: 20, fontSize: '0.65rem' }}
                                    />
                                  ) : (
                                    <Chip
                                      label="REJECTED"
                                      size="small"
                                      variant="outlined"
                                      sx={{ height: 20, fontSize: '0.65rem', opacity: 0.7 }}
                                    />
                                  )}
                                </TableCell>
                                <TableCell sx={{ fontWeight: isSelected ? 600 : 400 }}>{k.tag}</TableCell>
                                <TableCell>
                                  <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                                    {k.source}
                                  </Typography>
                                </TableCell>
                                <TableCell align="right">
                                  {k.base_score != null ? k.base_score.toFixed(4) : '-'}
                                </TableCell>
                                <TableCell align="right">
                                  {k.reranker_score != null ? k.reranker_score.toFixed(4) : '-'}
                                </TableCell>
                                <TableCell>
                                  <Typography
                                    variant="caption"
                                    color={k.flavor_filter_passed ? 'text.secondary' : 'error.main'}
                                  >
                                    {k.flavor_filter_passed ? 'Passed' : 'Pruned'}
                                  </Typography>
                                </TableCell>
                                <TableCell>
                                  <Typography variant="caption" color="text.secondary">
                                    {k.prune_reason || (isSelected ? 'Selected' : '-')}
                                  </Typography>
                                </TableCell>
                              </TableRow>
                            );
                          })}
                          {filteredKeywords.length === 0 && (
                            <TableRow>
                              <TableCell colSpan={7} align="center" sx={{ py: 3, color: 'text.secondary' }}>
                                No keywords found
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Stack>
                )}

                {auditTab === 2 && (
                  <Grid container spacing={3}>
                    <Grid size={{ xs: 12, md: 6 }}>
                      <Card variant="outlined">
                        <CardHeader
                          title="Basic Type (BT) Candidates"
                          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
                        />
                        <CardContent>
                          <Confidence
                            label={`Chosen: ${auditTrace.stage2_bt_classification?.predicted_bt || '(None)'}`}
                            value={auditTrace.stage2_bt_classification?.confidence || 0}
                            status={auditTrace.stage2_bt_classification?.status}
                            source={auditTrace.stage2_bt_classification?.source}
                          />
                          <Divider sx={{ my: 2 }} />
                          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
                            Candidate Basic Types:
                          </Typography>
                          <Stack spacing={1}>
                            {(auditTrace.stage2_bt_classification?.top_candidates || []).map((cand, idx) => (
                              <Box key={idx} sx={{ p: 1, borderRadius: 1, bgcolor: 'action.hover' }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.3 }}>
                                  <Typography variant="body2">{cand.bt}</Typography>
                                  <Typography variant="body2">{Math.round(cand.score * 100)}%</Typography>
                                </Box>
                                <LinearProgress
                                  variant="determinate"
                                  value={cand.score * 100}
                                  sx={{ height: 4, borderRadius: 2 }}
                                />
                              </Box>
                            ))}
                          </Stack>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                      <Card variant="outlined">
                        <CardHeader
                          title={`${auditTrace.stage4_third_tag_classification?.tag_label || 'Category'} Candidates`}
                          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
                        />
                        <CardContent>
                          <Confidence
                            label={`Chosen: ${auditTrace.stage4_third_tag_classification?.suggested_tag || '(None)'}`}
                            value={auditTrace.stage4_third_tag_classification?.confidence || 0}
                            status={auditTrace.stage4_third_tag_classification?.status}
                            source={auditTrace.stage4_third_tag_classification?.source}
                          />
                          <Divider sx={{ my: 2 }} />
                          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
                            Candidate Scores:
                          </Typography>
                          <Stack spacing={1}>
                            {(auditTrace.stage4_third_tag_classification?.top_candidates || []).map((cand, idx) => (
                              <Box key={idx} sx={{ p: 1, borderRadius: 1, bgcolor: 'action.hover' }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.3 }}>
                                  <Typography variant="body2">{cand.tag}</Typography>
                                  <Typography variant="body2">{Math.round(cand.score * 100)}%</Typography>
                                </Box>
                                <LinearProgress
                                  variant="determinate"
                                  value={cand.score * 100}
                                  sx={{ height: 4, borderRadius: 2 }}
                                />
                              </Box>
                            ))}
                          </Stack>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                )}

                {auditTab === 3 && <JsonBlock value={auditTrace} />}
              </>
            )}

            {isMatcherAudit && (
              <>
                {auditTab === 0 && (
                  <Stack spacing={3}>
                    <Box>
                      <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
                        Stage 1: NLP Normalization & Entity Extraction
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid size={{ xs: 12, sm: 6 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Clean Input
                            </Typography>
                            <Typography variant="body1" fontWeight={600}>
                              {auditTrace.stage1_nlp?.clean_input || '-'}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid size={{ xs: 12, sm: 6 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Weight Stripped Input
                            </Typography>
                            <Typography variant="body1" fontWeight={600}>
                              {auditTrace.stage1_nlp?.weight_stripped_input || '-'}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Box>

                    <Divider />

                    <Box>
                      <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
                        Matcher Search Strategy
                      </Typography>
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="body2">
                          <strong>Strategy:</strong>{' '}
                          {auditTrace.stage2_candidate_retrieval?.search_strategy || 'Hybrid Query'} |{' '}
                          <strong>Candidates Retrieved:</strong>{' '}
                          {auditTrace.stage2_candidate_retrieval?.total_candidates_found || 0}
                        </Typography>
                      </Paper>
                    </Box>
                  </Stack>
                )}

                {auditTab === 1 && (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ fontWeight: 700 }}>#</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>Catalog Name</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>Brand</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>BasicType</TableCell>
                          <TableCell sx={{ fontWeight: 700 }} align="right">
                            Raw CE Score
                          </TableCell>
                          <TableCell sx={{ fontWeight: 700 }} align="right">
                            Final CE Score
                          </TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(auditTrace.stage3_cross_encoder?.top_candidates || []).map((c, idx) => (
                          <TableRow key={idx}>
                            <TableCell>{idx + 1}</TableCell>
                            <TableCell sx={{ fontWeight: 600 }}>{c.catalog_name}</TableCell>
                            <TableCell>{c.brand || '-'}</TableCell>
                            <TableCell>{c.basic_type || '-'}</TableCell>
                            <TableCell align="right">{c.raw_cross_score?.toFixed(4)}</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600 }}>
                              {c.final_cross_score?.toFixed(4)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}

                {auditTab === 2 && (
                  <Stack spacing={2}>
                    {(auditTrace.stage4_logic_gates || []).map((g, idx) => (
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
                )}

                {auditTab === 3 && <JsonBlock value={auditTrace} />}
              </>
            )}

            {isPipelineAudit && (
              <>
                {auditTab === 0 && (
                  <Stack spacing={3}>
                    <Box>
                      <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1 }}>
                        Stage 1: NLP Normalization & Entity Extraction
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid size={{ xs: 12, sm: 6 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Clean Input
                            </Typography>
                            <Typography variant="body1" fontWeight={600}>
                              {auditTrace.stage1_nlp?.clean_input || '-'}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid size={{ xs: 12, sm: 6 }}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Weight Stripped Input
                            </Typography>
                            <Typography variant="body1" fontWeight={600}>
                              {auditTrace.stage1_nlp?.weight_stripped_input || '-'}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Box>

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
                )}

                {auditTab === 1 && (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ fontWeight: 700 }}>#</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>Catalog Name</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>Brand</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>BasicType</TableCell>
                          <TableCell sx={{ fontWeight: 700 }} align="right">
                            Final CE Score
                          </TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(auditTrace.stage3_cross_encoder?.top_candidates || []).map((c, idx) => (
                          <TableRow key={idx}>
                            <TableCell>{idx + 1}</TableCell>
                            <TableCell sx={{ fontWeight: 600 }}>{c.catalog_name}</TableCell>
                            <TableCell>{c.brand || '-'}</TableCell>
                            <TableCell>{c.basic_type || '-'}</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600 }}>
                              {c.final_cross_score?.toFixed(4)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}

                {auditTab === 2 && (
                  <Stack spacing={2}>
                    {(auditTrace.stage4_logic_gates || []).map((g, idx) => (
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
                )}

                {auditTab === 3 && (
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
                )}

                {auditTab === 4 && (
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
                )}

                {auditTab === 5 && <JsonBlock value={auditTrace} />}
              </>
            )}
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
