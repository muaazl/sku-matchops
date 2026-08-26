import React, { useMemo, useState } from 'react';
import {
  Box,
  Card,
  IconButton,
  TextField,
  MenuItem,
  Typography,
  Divider,
  Button,
  Stack,
  Switch,
  FormControlLabel,
  Chip,
  Grid,
  Drawer,
  Table,
  TableCell,
  TableRow,
  TablePagination,
} from '@mui/material';
import { Plus, Trash2, Play, FlaskConical, GripVertical, Pencil, X, Sparkles } from 'lucide-react';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader, StatusChip, ConfirmDialog } from '../components/ui';
import { RULE_MODULES, CONDITION_TYPES, ACTION_TYPES, DOMAINS } from '../constants';
import { JsonBlock } from '../components/JsonBlock';
import TagAutocomplete from '../components/rules/TagAutocomplete';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getRules, saveRule, reorderRules, testRule, deleteRule } from '../api';
import {
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledHeaderCell,
  StyledTableRow,
  TableSkeleton,
} from '../../components/Common/StyledTable';

const emptyRule = (currentRules = []) => {
  const maxPriority = currentRules.length > 0 ? Math.max(...currentRules.map((r) => r.priority || 0)) : 0;
  return {
    rule_id: `new_rule_${Math.floor(Math.random() * 900 + 100)}`,
    domain: 'market',
    module: 'bt_override',
    priority: maxPriority + 10,
    description: '',
    reasoning: '',
    condition_logic: 'AND',
    is_active: 1,
    conditions: [{ condition_group: 1, condition_type: 'sku_contains', value: '', negate: 0 }],
    actions: [{ action_type: 'set_bt', value: '' }],
  };
};

export default function RulesEngine() {
  const { enqueueSnackbar } = useSnackbar();
  const queryClient = useQueryClient();
  const [domain, setDomain] = useState('all');
  const [module, setModule] = useState('all');
  const [editing, setEditing] = useState(null);
  const [sample, setSample] = useState('{\n  "sku_name": "Red Bull Energy Drink 250ml",\n  "domain": "market"\n}');
  const [testResult, setTestResult] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Confirmation Modal State
  const [confirm, setConfirm] = useState({ open: false, title: '', message: '', onConfirm: null });

  // Reset page when filter changes
  React.useEffect(() => {
    setPage(0);
  }, [domain, module]);

  const { data: serverRules = [], isLoading } = useQuery({
    queryKey: ['rules', domain, module],
    queryFn: () =>
      getRules({
        ...(domain !== 'all' && { domain }),
        ...(module !== 'all' && { module }),
      }),
  });

  const filtered = useMemo(() => {
    const res = [...serverRules];
    res.sort((a, b) => a.priority - b.priority);
    return res;
  }, [serverRules]);

  const [localRules, setLocalRules] = useState([]);
  const [draggedIndex, setDraggedIndex] = useState(null);

  React.useEffect(() => {
    setLocalRules(filtered);
  }, [filtered]);

  const reorderMutation = useMutation({
    mutationFn: reorderRules,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rules'] });
      enqueueSnackbar('Rules reordered successfully', { variant: 'success' });
    },
    onError: (e) => enqueueSnackbar(`Reorder failed: ${e.message}`, { variant: 'error' }),
  });

  const handleDragStart = (e, index) => {
    const actualIndex = page * rowsPerPage + index;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', actualIndex);

    const rowEl = e.currentTarget.closest('tr');
    if (rowEl) {
      e.dataTransfer.setDragImage(rowEl, 20, 20);
    }

    setDraggedIndex(actualIndex);
  };

  const handleDragOver = (e, index) => {
    e.preventDefault();
    const actualIndex = page * rowsPerPage + index;
    if (draggedIndex === null || draggedIndex === actualIndex) return;

    const updated = [...localRules];
    const [draggedItem] = updated.splice(draggedIndex, 1);
    updated.splice(actualIndex, 0, draggedItem);

    setLocalRules(updated);
    setDraggedIndex(actualIndex);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
    const initialIds = filtered.map((r) => r.rule_id).join(',');
    const newIds = localRules.map((r) => r.rule_id).join(',');
    if (initialIds !== newIds) {
      reorderMutation.mutate({
        module: module,
        ordered_rule_ids: localRules.map((r) => r.rule_id),
      });
    }
  };

  const saveMutation = useMutation({
    mutationFn: saveRule,
    onSuccess: (data) => {
      const savedId = data?.rule_id || editing?.rule_id;
      enqueueSnackbar(`Rule ${savedId} saved`, { variant: 'success' });
      setEditing(null);
      queryClient.invalidateQueries({ queryKey: ['rules'] });
    },
    onError: (e) => enqueueSnackbar(`Save failed: ${e.message}`, { variant: 'error' }),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteRule,
    onSuccess: (_, deletedId) => {
      enqueueSnackbar(`Rule ${deletedId} deleted`, { variant: 'success' });
      setEditing(null);
      queryClient.invalidateQueries({ queryKey: ['rules'] });
    },
    onError: (e) => enqueueSnackbar(`Delete failed: ${e.message}`, { variant: 'error' }),
  });

  const save = () => {
    saveMutation.mutate(editing);
  };

  const testMutation = useMutation({
    mutationFn: testRule,
    onSuccess: (data) => {
      setTestResult(data);
    },
    onError: (e) => enqueueSnackbar(`Test failed: ${e.message}`, { variant: 'error' }),
  });

  const runTest = () => {
    let parsed;
    try {
      parsed = JSON.parse(sample);
    } catch {
      enqueueSnackbar('Sample is not valid JSON', { variant: 'error' });
      return;
    }
    testMutation.mutate({ rule: editing, sampleRecord: parsed });
  };

  const generateSampleForRule = (rule) => {
    if (!rule) return;
    const sampleObj = {
      sku_name: 'Sample Product 250ml',
      domain: rule.domain || 'market',
      price: 10.0,
      category: 'General',
      gk: [],
      bt: '',
      region: '',
    };

    if (rule.conditions && rule.conditions.length > 0) {
      const skuKeywords = [];
      const excludedKeywords = [];

      rule.conditions.forEach((c) => {
        const val = (c.value || '').trim();
        if (!val) return;

        if (c.condition_type === 'sku_contains') {
          if (!c.negate) {
            skuKeywords.push(val);
          } else {
            excludedKeywords.push(val.toLowerCase());
          }
        } else if (c.condition_type === 'bt_is') {
          sampleObj.bt = c.negate ? 'Other Type' : val;
        } else if (c.condition_type === 'gk_contains') {
          if (!c.negate) sampleObj.gk.push(val);
        } else if (c.condition_type === 'category_contains') {
          sampleObj.category = c.negate ? 'Other Category' : val;
        } else if (c.condition_type === 'region_is') {
          sampleObj.region = c.negate ? 'Other Region' : val;
        } else if (c.condition_type === 'price_below') {
          const p = parseFloat(val);
          if (!isNaN(p)) sampleObj.price = Math.max(0, p - 1);
        } else if (c.condition_type === 'price_above') {
          const p = parseFloat(val);
          if (!isNaN(p)) sampleObj.price = p + 5;
        } else if (c.condition_type === 'flavor_contains' || c.condition_type === 'flavor_is') {
          if (!c.negate) {
            if (val.toLowerCase() === 'seafood') {
              skuKeywords.push('Prawn Curry');
            } else if (val.toLowerCase() === 'meat') {
              skuKeywords.push('Chicken');
            } else if (val.toLowerCase() === 'vegetable' || val.toLowerCase() === 'veg') {
              skuKeywords.push('Potato');
            } else {
              skuKeywords.push(val);
            }
          }
        }
      });

      // Filter out any keyword that is explicitly negated by a NOT sku_contains condition
      const cleanKeywords = skuKeywords.filter((kw) => !excludedKeywords.some((ex) => kw.toLowerCase().includes(ex)));

      if (cleanKeywords.length > 0) {
        sampleObj.sku_name = `${cleanKeywords.join(' ')} 250ml`;
      }
    }

    setSample(JSON.stringify(sampleObj, null, 2));
    enqueueSnackbar('Auto-generated test sample matching rule conditions', { variant: 'info' });
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const paginatedRows = useMemo(() => {
    return localRules.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [localRules, page, rowsPerPage]);

  return (
    <PageContainer>
      <PageHeader
        title="Rules Engine"
        subtitle="Domain- and module-scoped rules. Edit conditions/actions and test against a sample."
        actions={
          <Button
            variant="contained"
            startIcon={<Plus size={16} />}
            onClick={() => {
              setEditing(emptyRule(filtered));
              setTestResult(null);
            }}
          >
            New rule
          </Button>
        }
      />

      <Card sx={{ p: 2, mb: 3 }}>
        <Stack direction="row" spacing={2}>
          <TextField
            select
            size="small"
            label="Domain"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            sx={{ width: 160 }}
          >
            <MenuItem value="all">All domains</MenuItem>
            {DOMAINS.map((d) => (
              <MenuItem key={d} value={d} sx={{ textTransform: 'capitalize' }}>
                {d}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            select
            size="small"
            label="Module"
            value={module}
            onChange={(e) => setModule(e.target.value)}
            sx={{ width: 180 }}
          >
            <MenuItem value="all">All modules</MenuItem>
            {RULE_MODULES.map((m) => (
              <MenuItem key={m} value={m}>
                {m}
              </MenuItem>
            ))}
          </TextField>
        </Stack>
      </Card>

      <StyledTableContainer>
        <Table aria-label="rules table">
          <StyledTableHead>
            <TableRow>
              <StyledHeaderCell width="50px" />
              <StyledHeaderCell width="15%">Rule ID</StyledHeaderCell>
              <StyledHeaderCell width="12%">Domain</StyledHeaderCell>
              <StyledHeaderCell width="15%">Module</StyledHeaderCell>
              <StyledHeaderCell>Description</StyledHeaderCell>
              <StyledHeaderCell width="12%">Status</StyledHeaderCell>
              <StyledHeaderCell align="right" width="10%">
                Actions
              </StyledHeaderCell>
            </TableRow>
          </StyledTableHead>
          <StyledTableBody>
            {isLoading ? (
              <TableSkeleton columns={7} rows={5} />
            ) : paginatedRows.length === 0 ? (
              <StyledTableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No rules found.
                  </Typography>
                </TableCell>
              </StyledTableRow>
            ) : (
              paginatedRows.map((row, index) => {
                const isItemDragged = draggedIndex === page * rowsPerPage + index;
                return (
                  <StyledTableRow
                    key={row.rule_id}
                    hover
                    onDragOver={(e) => handleDragOver(e, index)}
                    sx={{
                      cursor: 'default',
                      opacity: isItemDragged ? 0.4 : 1,
                      backgroundColor: isItemDragged ? 'action.hover' : 'inherit',
                      transition: 'opacity 0.2s ease, background-color 0.2s ease',
                    }}
                  >
                    <TableCell onClick={(e) => e.stopPropagation()} sx={{ width: '50px' }}>
                      {module !== 'all' ? (
                        <Box
                          draggable
                          onDragStart={(e) => handleDragStart(e, index)}
                          onDragEnd={handleDragEnd}
                          sx={{
                            cursor: 'grab',
                            '&:active': { cursor: 'grabbing' },
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            p: 0.5,
                            color: 'text.secondary',
                          }}
                          title="Drag to reorder"
                        >
                          <GripVertical size={16} />
                        </Box>
                      ) : (
                        <Box
                          sx={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            p: 0.5,
                            color: 'text.disabled',
                            opacity: 0.5,
                          }}
                          title="Select a specific module to enable reordering"
                        >
                          <GripVertical size={16} />
                        </Box>
                      )}
                    </TableCell>
                    <TableCell>{row.rule_id}</TableCell>
                    <TableCell sx={{ textTransform: 'capitalize' }}>{row.domain}</TableCell>
                    <TableCell>{row.module}</TableCell>
                    <TableCell>{row.description}</TableCell>
                    <TableCell>
                      <StatusChip status={row.is_active ? 'active' : 'disabled'} />
                    </TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            const ruleCopy = JSON.parse(JSON.stringify(row));
                            if (!ruleCopy.conditions) ruleCopy.conditions = [];
                            if (!ruleCopy.actions) ruleCopy.actions = [];
                            setEditing(ruleCopy);
                            setTestResult(null);
                          }}
                          aria-label="Edit rule"
                          title="Edit rule"
                        >
                          <Pencil size={18} />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="error"
                          onClick={(e) => {
                            e.stopPropagation();
                            setConfirm({
                              open: true,
                              title: 'Delete Rule',
                              message: `Are you sure you want to delete rule ${row.rule_id}?`,
                              onConfirm: () => deleteMutation.mutate(row.rule_id),
                            });
                          }}
                          aria-label="Delete rule"
                          title="Delete rule"
                        >
                          <Trash2 size={18} />
                        </IconButton>
                      </Stack>
                    </TableCell>
                  </StyledTableRow>
                );
              })
            )}
          </StyledTableBody>
        </Table>
        <TablePagination
          rowsPerPageOptions={[10, 25]}
          component="div"
          count={filtered.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
          sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
        />
      </StyledTableContainer>

      <Drawer
        anchor="right"
        open={!!editing}
        onClose={() => setEditing(null)}
        PaperProps={{ sx: { width: { xs: '100%', sm: 620 } } }}
      >
        {editing && (
          <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Edit rule</Typography>
              <IconButton aria-label="Close edit rule" onClick={() => setEditing(null)} size="small">
                <X size={18} />
              </IconButton>
            </Box>

            <Stack spacing={2}>
              <TextField
                label="Rule ID"
                value={editing.rule_id.startsWith('new_rule_') ? '(Autogenerated)' : editing.rule_id}
                disabled
                size="small"
              />
              <TextField
                label="Description"
                value={editing.description}
                onChange={(e) => setEditing({ ...editing, description: e.target.value })}
                size="small"
              />
              <Grid container spacing={2}>
                <Grid size={6}>
                  <TextField
                    select
                    label="Domain"
                    value={editing.domain}
                    onChange={(e) => setEditing({ ...editing, domain: e.target.value })}
                    size="small"
                    fullWidth
                  >
                    {DOMAINS.map((d) => (
                      <MenuItem key={d} value={d}>
                        {d}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid size={6}>
                  <TextField
                    select
                    label="Module"
                    value={editing.module}
                    onChange={(e) => setEditing({ ...editing, module: e.target.value })}
                    size="small"
                    fullWidth
                  >
                    {RULE_MODULES.map((m) => (
                      <MenuItem key={m} value={m}>
                        {m}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
              </Grid>
              <FormControlLabel
                control={
                  <Switch
                    checked={!!editing.is_active}
                    onChange={(e) => setEditing({ ...editing, is_active: e.target.checked ? 1 : 0 })}
                  />
                }
                label="Active"
              />
            </Stack>

            <Divider sx={{ my: 2.5 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2">Conditions ({editing.condition_logic})</Typography>
              <TextField
                select
                size="small"
                value={editing.condition_logic}
                onChange={(e) => setEditing({ ...editing, condition_logic: e.target.value })}
                sx={{ width: 90 }}
              >
                <MenuItem value="AND">AND</MenuItem>
                <MenuItem value="OR">OR</MenuItem>
              </TextField>
            </Box>
            <Stack spacing={1.5}>
              {editing.conditions.map((c, i) => (
                <Stack key={i} direction="row" spacing={1} alignItems="center">
                  <TextField
                    select
                    size="small"
                    value={c.condition_type}
                    onChange={(e) => {
                      const conditions = [...editing.conditions];
                      conditions[i] = { ...c, condition_type: e.target.value };
                      setEditing({ ...editing, conditions });
                    }}
                    sx={{ width: 170 }}
                  >
                    {CONDITION_TYPES.map((t) => (
                      <MenuItem key={t} value={t}>
                        {t}
                      </MenuItem>
                    ))}
                  </TextField>
                  <TagAutocomplete
                    value={c.value}
                    onChange={(newValue) => {
                      const conditions = [...editing.conditions];
                      conditions[i] = { ...c, value: newValue };
                      setEditing({ ...editing, conditions });
                    }}
                    type={c.condition_type}
                    ruleDomain={editing.domain}
                    placeholder="value"
                  />
                  <Button
                    variant={c.negate ? 'contained' : 'outlined'}
                    color={c.negate ? 'error' : 'inherit'}
                    size="small"
                    onClick={() => {
                      const conditions = [...editing.conditions];
                      conditions[i] = { ...c, negate: c.negate ? 0 : 1 };
                      setEditing({ ...editing, conditions });
                    }}
                    sx={{ minWidth: 60, height: 40 }}
                  >
                    NOT
                  </Button>
                  <IconButton
                    aria-label="Delete condition"
                    size="small"
                    onClick={() => setEditing({ ...editing, conditions: editing.conditions.filter((_, j) => j !== i) })}
                  >
                    <Trash2 size={15} />
                  </IconButton>
                </Stack>
              ))}
              <Button
                size="small"
                startIcon={<Plus size={14} />}
                sx={{ alignSelf: 'flex-start' }}
                onClick={() =>
                  setEditing({
                    ...editing,
                    conditions: [
                      ...editing.conditions,
                      {
                        condition_group: editing.conditions.length + 1,
                        condition_type: 'sku_contains',
                        value: '',
                        negate: 0,
                      },
                    ],
                  })
                }
              >
                Add condition
              </Button>
            </Stack>

            <Divider sx={{ my: 2.5 }} />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Actions
            </Typography>
            <Stack spacing={1.5}>
              {editing.actions.map((a, i) => (
                <Stack key={i} direction="row" spacing={1} alignItems="center">
                  <TextField
                    select
                    size="small"
                    value={a.action_type}
                    onChange={(e) => {
                      const actions = [...editing.actions];
                      actions[i] = { ...a, action_type: e.target.value };
                      setEditing({ ...editing, actions });
                    }}
                    sx={{ width: 170 }}
                  >
                    {ACTION_TYPES.map((t) => (
                      <MenuItem key={t} value={t}>
                        {t}
                      </MenuItem>
                    ))}
                  </TextField>
                  <TagAutocomplete
                    value={a.value}
                    onChange={(newValue) => {
                      const actions = [...editing.actions];
                      actions[i] = { ...a, value: newValue };
                      setEditing({ ...editing, actions });
                    }}
                    type={a.action_type}
                    ruleDomain={editing.domain}
                    placeholder="value"
                  />
                  <IconButton
                    aria-label="Delete action"
                    size="small"
                    onClick={() => setEditing({ ...editing, actions: editing.actions.filter((_, j) => j !== i) })}
                  >
                    <Trash2 size={15} />
                  </IconButton>
                </Stack>
              ))}
              <Button
                size="small"
                startIcon={<Plus size={14} />}
                sx={{ alignSelf: 'flex-start' }}
                onClick={() =>
                  setEditing({ ...editing, actions: [...editing.actions, { action_type: 'set_bt', value: '' }] })
                }
              >
                Add action
              </Button>
            </Stack>

            <Divider sx={{ my: 2.5 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <FlaskConical size={16} /> Test against sample
              </Typography>
              <Button
                size="small"
                variant="text"
                startIcon={<Sparkles size={14} />}
                onClick={() => generateSampleForRule(editing)}
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
              onChange={(e) => setSample(e.target.value)}
              sx={{ fontFamily: 'monospace', '& textarea': { fontFamily: 'monospace', fontSize: 13 } }}
            />
            <Button
              variant="outlined"
              color="inherit"
              disabled={testMutation.isPending}
              startIcon={<Play size={14} />}
              onClick={runTest}
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

            <Stack direction="row" spacing={1.5} sx={{ mt: 3 }}>
              <Button variant="contained" disabled={saveMutation.isPending} onClick={save}>
                Save rule
              </Button>
              {!editing.rule_id.startsWith('new_rule_') && (
                <Button
                  variant="outlined"
                  color="error"
                  disabled={deleteMutation.isPending}
                  onClick={() => {
                    setConfirm({
                      open: true,
                      title: 'Delete Rule',
                      message: `Are you sure you want to delete rule ${editing.rule_id}?`,
                      onConfirm: () => deleteMutation.mutate(editing.rule_id),
                    });
                  }}
                >
                  Delete rule
                </Button>
              )}
              <Button color="inherit" onClick={() => setEditing(null)}>
                Cancel
              </Button>
            </Stack>
          </Box>
        )}
      </Drawer>

      <ConfirmDialog
        open={confirm.open}
        onClose={() => setConfirm((prev) => ({ ...prev, open: false }))}
        title={confirm.title}
        message={confirm.message}
        onConfirm={() => {
          if (confirm.onConfirm) confirm.onConfirm();
          setConfirm((prev) => ({ ...prev, open: false }));
        }}
        confirmText="Yes, delete"
        cancelText="Cancel"
        confirmColor="error"
      />
    </PageContainer>
  );
}
