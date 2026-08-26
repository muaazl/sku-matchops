import React, { useState } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  TextField,
  Button,
  ToggleButtonGroup,
  ToggleButton,
  Typography,
  Divider,
  Stack,
  LinearProgress,
  CircularProgress,
  Chip,
  Tooltip,
  FormControlLabel,
  Switch,
} from '@mui/material';
import { Play, Sparkles, ArrowRight, Cpu } from 'lucide-react';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader, StatusChip } from '../components/ui';
import RulesAppliedList from '../components/RulesAppliedList';
import { DOMAINS } from '../constants';
import { runInteractiveSingle, getTemplateSuggestions, runInteractiveAudit } from '../api';
import AuditTelemetryView from './interactive/AuditTelemetryView';

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

export default function Interactive() {
  const { enqueueSnackbar } = useSnackbar();

  const [viewMode, setViewMode] = useState('interactive');
  const [auditTrace, setAuditTrace] = useState(null);
  const [auditLoading, setAuditLoading] = useState(false);

  const [sku, setSku] = useState('Red Bull Energy Drink 250ml');
  const [price, setPrice] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [task, setTask] = useState('pipeline');
  const [domain, setDomain] = useState(DOMAINS[0]);

  const [result, setResult] = useState(null);
  const [resultTask, setResultTask] = useState('pipeline');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState(null);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [enableSuggestions, setEnableSuggestions] = useState(true);

  const handleRunAudit = async () => {
    if (!sku) {
      enqueueSnackbar('Please enter a SKU name', { variant: 'warning' });
      return;
    }
    setAuditLoading(true);
    try {
      const auditTaskToRun = resultTask || task;
      const payload = {
        sku_name: sku,
        domain: result?.domain || domain,
        task: auditTaskToRun,
        price: price ? parseFloat(price) : 0.0,
        description,
        category,
      };
      const auditRes = await runInteractiveAudit(payload);
      setAuditTrace(auditRes);
      setViewMode('audit');
      enqueueSnackbar('Audit telemetry loaded successfully', { variant: 'success' });
    } catch (err) {
      console.error('Audit fetch error:', err);
      enqueueSnackbar(`Failed to load audit: ${err.response?.data?.detail || err.message}`, { variant: 'error' });
    } finally {
      setAuditLoading(false);
    }
  };

  const run = async () => {
    if (!sku) {
      enqueueSnackbar('Please enter a SKU name', { variant: 'warning' });
      return;
    }
    setLoading(true);
    setResult(null);
    setSuggestions(null);
    try {
      const payload = {
        sku_name: sku,
        domain,
        task,
        price: price ? parseFloat(price) : 0.0,
        description,
        category,
      };
      const res = await runInteractiveSingle(payload);
      setResult(res);
      setResultTask(task);
      const r = res.results?.[0] || {};
      enqueueSnackbar('Execution completed successfully', { variant: 'success' });

      if (enableSuggestions) {
        setSuggestionsLoading(true);
        getTemplateSuggestions({
          sku_name: sku,
          domain,
          exclude_bt: r.suggested_bt || '',
          exclude_gk: r.suggested_gk || '',
        })
          .then((suggRes) => setSuggestions(suggRes))
          .catch((err) => console.error('Failed to fetch template suggestions:', err))
          .finally(() => setSuggestionsLoading(false));
      }
    } catch (err) {
      console.error(err);
      enqueueSnackbar(`Execution failed: ${err.response?.data?.detail || err.message}`, { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const r = result?.results?.[0];

  if (viewMode === 'audit') {
    return (
      <AuditTelemetryView
        auditTrace={auditTrace}
        sku={sku}
        domain={result?.domain || domain}
        task={task}
        resultTask={resultTask}
        onBack={() => setViewMode('interactive')}
      />
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Interactive"
        subtitle="Run a single SKU through the matcher, classifier or full pipeline."
      />

      <Grid container spacing={3} alignItems="flex-start">
        <Grid size={{ xs: 12, md: 5 }}>
          <Card>
            <CardHeader
              title="Input Details"
              titleTypographyProps={{ variant: 'subtitle1', sx: { fontWeight: 600 } }}
            />
            <CardContent>
              <Stack spacing={2.5}>
                <TextField
                  label="SKU name"
                  value={sku}
                  onChange={(e) => setSku(e.target.value)}
                  fullWidth
                  onKeyDown={(e) => e.key === 'Enter' && run()}
                />
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                  <TextField label="Price" value={price} onChange={(e) => setPrice(e.target.value)} fullWidth />
                  <TextField
                    label="Seller Category"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    fullWidth
                  />
                </Stack>
                <TextField
                  label="Description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  fullWidth
                  multiline
                  rows={2}
                />

                <Stack spacing={2}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                      Domain
                    </Typography>
                    <ToggleButtonGroup
                      size="small"
                      exclusive
                      value={domain}
                      onChange={(e, v) => v && setDomain(v)}
                      fullWidth
                    >
                      {DOMAINS.map((d) => (
                        <ToggleButton key={d} value={d} sx={{ textTransform: 'capitalize' }}>
                          {d}
                        </ToggleButton>
                      ))}
                    </ToggleButtonGroup>
                  </Box>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                      Task
                    </Typography>
                    <ToggleButtonGroup
                      size="small"
                      exclusive
                      value={task}
                      onChange={(e, v) => v && setTask(v)}
                      fullWidth
                    >
                      <ToggleButton value="matcher" sx={{ textTransform: 'none' }}>
                        Matcher
                      </ToggleButton>
                      <ToggleButton value="classifier" sx={{ textTransform: 'none' }}>
                        Classifier
                      </ToggleButton>
                      <ToggleButton value="pipeline" sx={{ textTransform: 'none' }}>
                        Pipeline
                      </ToggleButton>
                    </ToggleButtonGroup>
                  </Box>

                  <FormControlLabel
                    control={
                      <Switch
                        checked={enableSuggestions}
                        onChange={(e) => setEnableSuggestions(e.target.checked)}
                        color="primary"
                        size="small"
                      />
                    }
                    label={
                      <Typography variant="body2" color="text.secondary" fontWeight={500}>
                        Enable Tag Suggestions (Experimental)
                      </Typography>
                    }
                    sx={{ mt: 0.5 }}
                  />
                </Stack>

                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <Play size={16} />}
                  onClick={run}
                  disabled={loading}
                  fullWidth
                  size="large"
                  sx={{ mt: 1, height: 42, fontWeight: 600 }}
                >
                  {loading ? 'Running...' : 'Run'}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 7 }}>
          {loading && (
            <Card
              sx={{
                p: 6,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 300,
              }}
            >
              <CircularProgress sx={{ mb: 2 }} />
              <Typography variant="body2" color="text.secondary">
                Running pipeline tasks...
              </Typography>
            </Card>
          )}

          {!loading && !r && (
            <Card
              sx={{
                p: 6,
                textAlign: 'center',
                border: '2px dashed',
                borderColor: 'divider',
                backgroundColor: 'action.hover',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 400,
              }}
            >
              <Box sx={{ color: 'text.secondary', mb: 2 }}>
                <Sparkles size={48} strokeWidth={1.5} />
              </Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                Ready to Process
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 360, mb: 3 }}>
                Enter the SKU details on the left and choose a task. The results, confidence metrics, and matching logic
                will appear here.
              </Typography>
            </Card>
          )}

          {r && !loading && (
            <Stack spacing={3}>
              <Card>
                <CardHeader
                  title="Result"
                  titleTypographyProps={{ variant: 'subtitle1', sx: { fontWeight: 600 } }}
                  action={
                    <Button
                      variant="contained"
                      color="secondary"
                      size="small"
                      startIcon={auditLoading ? <CircularProgress size={14} color="inherit" /> : <Cpu size={14} />}
                      onClick={handleRunAudit}
                      disabled={auditLoading}
                      sx={{ fontWeight: 600, textTransform: 'none', borderRadius: 2 }}
                    >
                      {auditLoading ? 'Auditing...' : 'Audit Result'}
                    </Button>
                  }
                />
                <CardContent>
                  {r.matched_catalog_name && (
                    <>
                      <Typography variant="caption" color="text.secondary">
                        Matched catalog item
                      </Typography>
                      <Typography variant="body1" sx={{ fontWeight: 500, mb: 1 }}>
                        {r.matched_catalog_name}
                      </Typography>
                      {r.score != null && <Confidence label="Match score" value={r.score} status={r.status} />}
                      {r.logic_notes && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          {r.logic_notes}
                        </Typography>
                      )}
                      <Divider sx={{ my: 2 }} />
                    </>
                  )}
                  {r.suggested_bt && (
                    <Confidence
                      label={`Basic Type · ${r.suggested_bt}`}
                      value={r.bt_confidence}
                      status={r.bt_status}
                      source={r.bt_source}
                    />
                  )}
                  {r.suggested_gk && (
                    <Confidence
                      label={`Generic Keywords · ${r.suggested_gk}`}
                      value={r.gk_confidence}
                      status={r.gk_status}
                    />
                  )}
                  {r.suggested_region && (
                    <Confidence
                      label={`${(result?.domain || domain) === 'food' ? 'Region' : 'Categories'} · ${
                        r.suggested_region
                      }`}
                      value={r.region_confidence}
                      status={r.region_status}
                      source={r.region_source}
                    />
                  )}
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Rules applied
                    </Typography>
                    <RulesAppliedList rules={r.rules_applied} />
                  </Box>
                </CardContent>
              </Card>

              {suggestionsLoading && (
                <Card>
                  <CardContent sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <CircularProgress size={20} />
                    <Typography variant="body2" color="text.secondary">
                      Searching for matching template SKU in catalog...
                    </Typography>
                  </CardContent>
                </Card>
              )}

              {suggestions && suggestions.matched && (
                <Card sx={{ borderLeft: '4px solid', borderColor: 'info.main' }}>
                  <CardHeader
                    title={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Sparkles size={18} style={{ color: '#0288d1' }} />
                        <Typography variant="subtitle1" fontWeight={600}>
                          Template-Based Suggestions
                        </Typography>
                      </Box>
                    }
                  />
                  <CardContent sx={{ pt: 0 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Found a similar base item in the catalog. Swapped the flavor/brand to generate matched tag
                      suggestions.
                    </Typography>

                    <Box sx={{ p: 1.5, bgcolor: 'action.hover', borderRadius: 1, mb: 2 }}>
                      <Typography variant="caption" color="text.secondary" display="block">
                        Base Catalog Item Match
                      </Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {suggestions.base_sku}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.8, mt: 0.5 }}>
                        <Typography
                          variant="caption"
                          sx={{ px: 0.8, py: 0.2, bgcolor: 'action.selected', borderRadius: 0.5, fontWeight: 500 }}
                        >
                          {suggestions.base_entity}
                        </Typography>
                        <ArrowRight size={12} style={{ color: 'text.secondary' }} />
                        <Typography
                          variant="caption"
                          sx={{
                            px: 0.8,
                            py: 0.2,
                            bgcolor: 'info.light',
                            color: 'info.contrastText',
                            borderRadius: 0.5,
                            fontWeight: 500,
                          }}
                        >
                          {suggestions.new_entity}
                        </Typography>
                      </Box>
                    </Box>

                    <Stack spacing={1.5}>
                      {suggestions.suggested_bt && (
                        <Box>
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                            Suggested Basic Type
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2" fontWeight={500}>
                              {suggestions.suggested_bt}
                            </Typography>
                            {suggestions.bt_info && (
                              <Chip
                                label={
                                  suggestions.bt_info.status === 'exact_dictionary_match'
                                    ? 'Exact Dictionary'
                                    : suggestions.bt_info.status === 'fuzzy_snapped'
                                    ? `Fuzzy Snapped (from "${suggestions.bt_info.original}")`
                                    : 'New / Unregistered'
                                }
                                size="small"
                                color={
                                  suggestions.bt_info.status === 'exact_dictionary_match'
                                    ? 'success'
                                    : suggestions.bt_info.status === 'fuzzy_snapped'
                                    ? 'info'
                                    : 'warning'
                                }
                                variant="outlined"
                                sx={{ height: 20, fontSize: '0.65rem' }}
                              />
                            )}
                          </Box>
                        </Box>
                      )}

                      {suggestions.gk_info && suggestions.gk_info.length > 0 && (
                        <Box>
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                            Suggested Generic Keywords
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                            {suggestions.gk_info.map((info, idx) => (
                              <Tooltip
                                key={idx}
                                title={
                                  info.status === 'exact_dictionary_match'
                                    ? 'Exact match in dictionary'
                                    : info.status === 'fuzzy_snapped'
                                    ? `Fuzzy snapped to dictionary from: ${info.original} (${Math.round(
                                        info.snap_score
                                      )}% similarity)`
                                    : `New unregistered tag: ${info.original} (capitalized according to casing exceptions)`
                                }
                              >
                                <Chip
                                  label={info.suggested}
                                  size="small"
                                  color={
                                    info.status === 'exact_dictionary_match'
                                      ? 'success'
                                      : info.status === 'fuzzy_snapped'
                                      ? 'info'
                                      : 'default'
                                  }
                                  variant="outlined"
                                  sx={{
                                    fontSize: '0.75rem',
                                    borderStyle: info.status === 'new_unregistered' ? 'dashed' : 'solid',
                                    borderColor: info.status === 'new_unregistered' ? 'warning.main' : undefined,
                                  }}
                                />
                              </Tooltip>
                            ))}
                          </Box>
                        </Box>
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              )}
            </Stack>
          )}
        </Grid>
      </Grid>
    </PageContainer>
  );
}

