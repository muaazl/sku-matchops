import React, { useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { Search } from '@mui/icons-material';
import { ConfidenceBar } from '../common/ConfidenceBar';
import { JsonBlock } from '../common/JsonBlock';
import RulesAppliedList from '../common/RulesAppliedList';
import { StageNlpSummary } from './StageNlpSummary';

export function ClassifierAuditTabs({ auditTrace, activeTab }) {
  const [keywordSearch, setKeywordSearch] = useState('');

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

  if (activeTab === 0) {
    return (
      <Stack spacing={3}>
        <StageNlpSummary auditTrace={auditTrace} />
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
    );
  }

  if (activeTab === 1) {
    return (
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
                  <Search sx={{ fontSize: 16 }} />
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
                    <TableCell align="right">{k.base_score != null ? k.base_score.toFixed(4) : '-'}</TableCell>
                    <TableCell align="right">{k.reranker_score != null ? k.reranker_score.toFixed(4) : '-'}</TableCell>
                    <TableCell>
                      <Typography variant="caption" color={k.flavor_filter_passed ? 'text.secondary' : 'error.main'}>
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
    );
  }

  if (activeTab === 2) {
    return (
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardHeader
              title="Basic Type (BT) Candidates"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
            />
            <CardContent>
              <ConfidenceBar
                label={`Chosen: ${auditTrace.stage2_bt_classification?.predicted_bt || '(None)'}`}
                value={auditTrace.stage2_bt_classification?.confidence || 0}
                status={auditTrace.stage2_bt_classification?.status}
                source={auditTrace.stage2_bt_classification?.source}
                colorize={false}
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
              <ConfidenceBar
                label={`Chosen: ${auditTrace.stage4_third_tag_classification?.suggested_tag || '(None)'}`}
                value={auditTrace.stage4_third_tag_classification?.confidence || 0}
                status={auditTrace.stage4_third_tag_classification?.status}
                source={auditTrace.stage4_third_tag_classification?.source}
                colorize={false}
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
    );
  }

  return <JsonBlock value={auditTrace} />;
}

ClassifierAuditTabs.propTypes = {
  auditTrace: PropTypes.object.isRequired,
  activeTab: PropTypes.number.isRequired,
};
