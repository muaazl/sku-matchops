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
  Drawer,
  Table,
  TableCell,
  TableRow,
  TablePagination,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  Add,
  DeleteOutline,
  PlayArrow,
  Science,
  DragIndicator,
  Edit,
  Close,
  AutoAwesome,
  Balance,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader, StatusChip, ConfirmDialog, EmptyState } from '../components/ui';
import { CONDITION_TYPES, CONDITION_LABELS, ACTION_TYPES, ACTION_LABELS, DOMAINS } from '../constants';
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

const normalizeConditions = (conditions = [], fallbackLogic = 'AND', fromServer = false) => {
  let group = 1;
  const fallback = String(fallbackLogic || 'AND').toUpperCase();
  return conditions.map((c, idx) => {
    let rawOp = c.operator || (idx > 0 ? fallback : 'AND') || 'AND';
    if (
      fromServer &&
      idx > 0 &&
      rawOp.toUpperCase() === 'AND' &&
      fallback === 'OR' &&
      c.condition_group !== conditions[idx - 1]?.condition_group
    ) {
      rawOp = 'OR';
    }
    const operator = idx === 0 ? 'AND' : rawOp.toUpperCase() === 'OR' ? 'OR' : 'AND';
    if (idx > 0 && operator === 'OR') {
      group += 1;
    }
    return {
      ...c,
      operator,
      condition_group: group,
      negate: c.negate ? 1 : 0,
    };
  });
};

const buildConditionGroups = (conditions = []) => {
  const groups = [];
  conditions.forEach((c, index) => {
    const op = index === 0 ? 'AND' : (c.operator || 'AND').toUpperCase();
    if (index === 0 || op === 'OR') {
      groups.push([{ cond: c, index }]);
    } else {
      groups[groups.length - 1].push({ cond: c, index });
    }
  });
  return groups;
};

const getConditionPlaceholder = (conditionType) => {
  if (conditionType === 'price_below' || conditionType === 'price_above') return 'e.g. 250';
  if (conditionType === 'flavor_is') return 'Select flavor type...';
  return 'Type or select value...';
};

const emptyRule = (currentRules = []) => {
  const maxPriority = currentRules.length > 0 ? Math.max(...currentRules.map((r) => r.priority || 0)) : 0;
  return {
    rule_id: `new_rule_${Math.floor(Math.random() * 900 + 100)}`,
    domain: 'market',
    priority: maxPriority + 10,
    description: '',
    reasoning: '',
    condition_logic: 'AND',
    is_active: 1,
    conditions: [{ condition_group: 1, operator: 'AND', condition_type: 'sku_contains', value: '', negate: 0 }],
    actions: [{ action_type: 'set_bt', value: '' }],
  };
};

// Stable fallback: a fresh [] per render would re-trigger the localRules sync effect forever
// while the rules query is loading (an infinite render loop).
const NO_RULES = [];

export default function RulesEngine() {
  const { enqueueSnackbar } = useSnackbar();
  const queryClient = useQueryClient();
  const [domain, setDomain] = useState('all');
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
  }, [domain]);

  const { data: serverRules = NO_RULES, isLoading } = useQuery({
    queryKey: ['rules', domain],
    queryFn: () => getRules({ ...(domain !== 'all' && { domain }) }),
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
      const groups = buildConditionGroups(rule.conditions);
      const activeGroup = groups.find((grp) => grp.some(({ cond }) => (cond.value || '').trim() !== '')) || groups[0];

      activeGroup.forEach(({ cond: c }) => {
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

  const updateConditions = (nextRaw) => {
    setEditing((prev) => {
      if (!prev) return prev;
      const normalized = normalizeConditions(nextRaw, prev.condition_logic);
      const hasOr = normalized.slice(1).some((cond) => cond.operator === 'OR');
      return {
        ...prev,
        condition_logic: hasOr ? 'OR' : 'AND',
        conditions: normalized,
      };
    });
  };

  const conditionGroups = useMemo(() => buildConditionGroups(editing?.conditions || []), [editing?.conditions]);

  const canEditRules = true;
  const lockedHint = '';

  const paginatedRows = useMemo(() => {
    return localRules.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [localRules, page, rowsPerPage]);

  return (
    <PageContainer>
      <PageHeader
        title="Rules Engine"
        subtitle="Domain-scoped rules. Edit conditions/actions and test against a sample."
        actions={
          <>
            <Tooltip title={lockedHint}>
              <span>
                <Button
                  variant="contained"
                  startIcon={<Add sx={{ fontSize: 16 }} />}
                  disabled={!canEditRules}
                  onClick={() => {
                    setEditing(emptyRule(filtered));
                    setTestResult(null);
                  }}
                >
                  New rule
                </Button>
              </span>
            </Tooltip>
          </>
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
        </Stack>
      </Card>

      <StyledTableContainer>
        <Table aria-label="rules table">
          <StyledTableHead>
            <TableRow>
              <StyledHeaderCell width="50px" />
              <StyledHeaderCell width="15%">Rule ID</StyledHeaderCell>
              <StyledHeaderCell width="12%">Domain</StyledHeaderCell>
              <StyledHeaderCell>Description</StyledHeaderCell>
              <StyledHeaderCell width="12%">Status</StyledHeaderCell>
              <StyledHeaderCell align="right" width="10%">
                Actions
              </StyledHeaderCell>
            </TableRow>
          </StyledTableHead>
          <StyledTableBody>
            {isLoading ? (
              <TableSkeleton columns={6} rows={5} />
            ) : paginatedRows.length === 0 ? (
              <StyledTableRow>
                <TableCell colSpan={6} sx={{ p: 0, border: 'none' }}>
                  <EmptyState
                    icon={Balance}
                    title="No rules found"
                    description={
                      domain !== 'all'
                        ? `No rules found for the "${domain}" domain. Try selecting "All Domain".`
                        : 'Define automated business rules to apply specific tags, categories, or actions to incoming SKUs.'
                    }
                    action={
                      canEditRules && domain === 'all' ? (
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<Add sx={{ fontSize: 16 }} />}
                          onClick={() => {
                            setEditing(emptyRule(filtered));
                            setTestResult(null);
                          }}
                          sx={{ textTransform: 'none', fontWeight: 600 }}
                        >
                          New rule
                        </Button>
                      ) : null
                    }
                    minHeight={240}
                  />
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
                      {domain !== 'all' && canEditRules ? (
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
                          <DragIndicator sx={{ fontSize: 16 }} />
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
                          title={canEditRules ? 'Select a specific domain to enable reordering' : lockedHint}
                        >
                          <DragIndicator sx={{ fontSize: 16 }} />
                        </Box>
                      )}
                    </TableCell>
                    <TableCell>{row.rule_id}</TableCell>
                    <TableCell sx={{ textTransform: 'capitalize' }}>{row.domain}</TableCell>
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
                            ruleCopy.conditions = normalizeConditions(
                              ruleCopy.conditions || [],
                              ruleCopy.condition_logic || 'AND',
                              true
                            );
                            ruleCopy.actions = (ruleCopy.actions || []).filter((a) =>
                              ACTION_TYPES.includes(a.action_type)
                            );
                            setEditing(ruleCopy);
                            setTestResult(null);
                          }}
                          aria-label="Edit rule"
                          title="Edit rule"
                        >
                          <Edit sx={{ fontSize: 18 }} />
                        </IconButton>
                        <Tooltip title={lockedHint || 'Delete rule'}>
                          <span>
                            <IconButton
                              size="small"
                              color="error"
                              disabled={!canEditRules}
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
                            >
                              <DeleteOutline sx={{ fontSize: 18 }} />
                            </IconButton>
                          </span>
                        </Tooltip>
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
        PaperProps={{ sx: { width: { xs: '100%', sm: 720 } } }}
      >
        {editing && (
          <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                {editing.rule_id.startsWith('new_rule_') ? 'Create rule' : 'Edit rule'}
              </Typography>
              <IconButton aria-label="Close edit rule" onClick={() => setEditing(null)} size="small">
                <Close sx={{ fontSize: 18 }} />
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

            <Box
              sx={{
                mb: 2,
                p: 1.25,
                borderRadius: 1.5,
                border: '1px solid',
                borderColor: 'divider',
                bgcolor: (theme) => alpha(theme.palette.primary.main, 0.04),
              }}
            >
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: 'block', fontWeight: 600, mb: 0.5, textTransform: 'uppercase', letterSpacing: 0.5 }}
              >
                Rule summary
              </Typography>
              {editing.conditions.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No conditions — matches every SKU in this domain.
                </Typography>
              ) : (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 0.6, lineHeight: 1.8 }}>
                  <Chip
                    label="IF"
                    size="small"
                    sx={{ height: 20, fontSize: '0.68rem', fontWeight: 700, borderRadius: 1 }}
                  />
                  {conditionGroups.map((group, gIdx) => (
                    <React.Fragment key={`preview-grp-${gIdx}`}>
                      {gIdx > 0 && (
                        <Chip
                          label="OR"
                          size="small"
                          color="warning"
                          sx={{ height: 20, fontSize: '0.68rem', fontWeight: 700, borderRadius: 1 }}
                        />
                      )}
                      {conditionGroups.length > 1 && group.length > 1 && (
                        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 700 }}>
                          (
                        </Typography>
                      )}
                      {group.map(({ cond, index }, itemIdx) => (
                        <React.Fragment key={`preview-cond-${index}`}>
                          {itemIdx > 0 && (
                            <Chip
                              label="AND"
                              size="small"
                              color="primary"
                              variant="outlined"
                              sx={{ height: 20, fontSize: '0.65rem', fontWeight: 700, borderRadius: 1 }}
                            />
                          )}
                          <Typography component="span" variant="body2" sx={{ fontSize: '0.8rem' }}>
                            {!!cond.negate && (
                              <Box component="span" sx={{ color: 'error.main', fontWeight: 700, mr: 0.5 }}>
                                NOT
                              </Box>
                            )}
                            <Box component="span" sx={{ fontWeight: 500 }}>
                              {CONDITION_LABELS[cond.condition_type] || cond.condition_type}
                            </Box>{' '}
                            <Box
                              component="span"
                              sx={{
                                fontWeight: 600,
                                color: cond.value ? 'text.primary' : 'text.disabled',
                                fontStyle: cond.value ? 'normal' : 'italic',
                              }}
                            >
                              {cond.value ? `"${cond.value}"` : '(empty)'}
                            </Box>
                          </Typography>
                        </React.Fragment>
                      ))}
                      {conditionGroups.length > 1 && group.length > 1 && (
                        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 700 }}>
                          )
                        </Typography>
                      )}
                    </React.Fragment>
                  ))}
                </Box>
              )}
            </Box>

            <Stack spacing={1.5}>
              {conditionGroups.map((group, gIdx) => (
                <React.Fragment key={`cond-group-${gIdx}`}>
                  {gIdx > 0 && (
                    <Divider sx={{ my: 0.5 }}>
                      <Chip
                        label="OR — alternative match"
                        size="small"
                        color="warning"
                        variant="outlined"
                        sx={{ fontWeight: 700, fontSize: '0.7rem', height: 22 }}
                      />
                    </Divider>
                  )}
                  <Stack
                    spacing={1}
                    sx={{
                      p: 1.25,
                      borderRadius: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                      bgcolor: (theme) => alpha(theme.palette.action.hover, 0.03),
                    }}
                  >
                    {group.map(({ cond: c, index: i }) => {
                      const isNegated = !!c.negate;
                      return (
                        <Box
                          key={i}
                          sx={{
                            p: 1,
                            borderRadius: 1.5,
                            border: '1px solid',
                            borderColor: (theme) =>
                              isNegated ? alpha(theme.palette.error.main, 0.4) : theme.palette.divider,
                            bgcolor: (theme) =>
                              isNegated ? alpha(theme.palette.error.main, 0.06) : theme.palette.background.paper,
                            transition: 'background-color 150ms ease, border-color 150ms ease',
                          }}
                        >
                          <Stack
                            direction={{ xs: 'column', sm: 'row' }}
                            spacing={1}
                            alignItems={{ xs: 'stretch', sm: 'center' }}
                          >
                            <Stack direction="row" spacing={0.75} alignItems="center" sx={{ flexShrink: 0 }}>
                              {i === 0 ? (
                                <Box
                                  sx={{
                                    width: 82,
                                    height: 32,
                                    borderRadius: 1,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '0.72rem',
                                    fontWeight: 700,
                                    letterSpacing: 0.5,
                                    color: 'primary.main',
                                    bgcolor: (theme) => alpha(theme.palette.primary.main, 0.12),
                                    border: (theme) => `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                                  }}
                                >
                                  IF
                                </Box>
                              ) : (
                                <Tooltip title="AND: must also match • OR: start alternative match">
                                  <ToggleButtonGroup
                                    exclusive
                                    size="small"
                                    value={c.operator || 'AND'}
                                    onChange={(_, nextOp) => {
                                      if (!nextOp) return;
                                      const next = [...editing.conditions];
                                      next[i] = { ...c, operator: nextOp };
                                      updateConditions(next);
                                    }}
                                    aria-label={`Condition ${i + 1} operator`}
                                    sx={{ height: 32, width: 82 }}
                                  >
                                    <ToggleButton
                                      value="AND"
                                      sx={{
                                        flex: 1,
                                        px: 0.5,
                                        fontSize: '0.68rem',
                                        fontWeight: 700,
                                        '&.Mui-selected': {
                                          bgcolor: (theme) => alpha(theme.palette.primary.main, 0.16),
                                          color: 'primary.main',
                                        },
                                      }}
                                    >
                                      AND
                                    </ToggleButton>
                                    <ToggleButton
                                      value="OR"
                                      sx={{
                                        flex: 1,
                                        px: 0.5,
                                        fontSize: '0.68rem',
                                        fontWeight: 700,
                                        '&.Mui-selected': {
                                          bgcolor: (theme) => alpha(theme.palette.warning.main, 0.2),
                                          color: 'warning.main',
                                        },
                                      }}
                                    >
                                      OR
                                    </ToggleButton>
                                  </ToggleButtonGroup>
                                </Tooltip>
                              )}

                              <Tooltip
                                title={
                                  isNegated
                                    ? 'Negated: must NOT match (click to turn off)'
                                    : 'Click to invert (must NOT match)'
                                }
                              >
                                <Button
                                  variant={isNegated ? 'contained' : 'outlined'}
                                  color={isNegated ? 'error' : 'inherit'}
                                  size="small"
                                  aria-pressed={isNegated}
                                  onClick={() => {
                                    const next = [...editing.conditions];
                                    next[i] = { ...c, negate: isNegated ? 0 : 1 };
                                    updateConditions(next);
                                  }}
                                  sx={{
                                    minWidth: 50,
                                    height: 32,
                                    px: 1,
                                    fontSize: '0.7rem',
                                    fontWeight: 700,
                                    borderStyle: isNegated ? 'solid' : 'dashed',
                                    color: isNegated ? undefined : 'text.secondary',
                                  }}
                                >
                                  NOT
                                </Button>
                              </Tooltip>
                            </Stack>

                            <TextField
                              select
                              size="small"
                              value={c.condition_type}
                              onChange={(e) => {
                                const next = [...editing.conditions];
                                next[i] = { ...c, condition_type: e.target.value };
                                updateConditions(next);
                              }}
                              sx={{ width: { xs: '100%', sm: 200 }, flexShrink: 0 }}
                            >
                              {CONDITION_TYPES.map((t) => (
                                <MenuItem key={t} value={t}>
                                  {CONDITION_LABELS[t] || t}
                                </MenuItem>
                              ))}
                            </TextField>

                            <Box sx={{ flex: 1, minWidth: 140, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <TagAutocomplete
                                value={c.value}
                                onChange={(newValue) => {
                                  const next = [...editing.conditions];
                                  next[i] = { ...c, value: newValue };
                                  updateConditions(next);
                                }}
                                type={c.condition_type}
                                ruleDomain={editing.domain}
                                placeholder={getConditionPlaceholder(c.condition_type)}
                              />
                              <Tooltip title="Remove condition">
                                <IconButton
                                  aria-label="Delete condition"
                                  size="small"
                                  onClick={() => updateConditions(editing.conditions.filter((_, j) => j !== i))}
                                >
                                  <DeleteOutline sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </Stack>
                        </Box>
                      );
                    })}
                  </Stack>
                </React.Fragment>
              ))}

              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<Add sx={{ fontSize: 14 }} />}
                  onClick={() =>
                    updateConditions([
                      ...editing.conditions,
                      {
                        condition_group: 1,
                        operator: 'AND',
                        condition_type: 'sku_contains',
                        value: '',
                        negate: 0,
                      },
                    ])
                  }
                >
                  {editing.conditions.length === 0 ? 'Add condition' : 'Add condition (AND)'}
                </Button>
                {editing.conditions.length > 0 && (
                  <Button
                    size="small"
                    variant="outlined"
                    color="warning"
                    startIcon={<Add sx={{ fontSize: 14 }} />}
                    onClick={() =>
                      updateConditions([
                        ...editing.conditions,
                        {
                          condition_group: 1,
                          operator: 'OR',
                          condition_type: 'sku_contains',
                          value: '',
                          negate: 0,
                        },
                      ])
                    }
                  >
                    Add alternative (OR)
                  </Button>
                )}
              </Stack>
            </Stack>

            <Divider sx={{ my: 2.5 }} />
            <Stack spacing={1.25}>
              {editing.actions.map((a, i) => (
                <Box
                  key={i}
                  sx={{
                    p: 1,
                    borderRadius: 1.5,
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: (theme) => theme.palette.background.paper,
                  }}
                >
                  <Stack
                    direction={{ xs: 'column', sm: 'row' }}
                    spacing={1}
                    alignItems={{ xs: 'stretch', sm: 'center' }}
                  >
                    <Box
                      sx={{
                        width: 56,
                        height: 32,
                        borderRadius: 1,
                        flexShrink: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        letterSpacing: 0.5,
                        color: 'success.main',
                        bgcolor: (theme) => alpha(theme.palette.success.main, 0.12),
                        border: (theme) => `1px solid ${alpha(theme.palette.success.main, 0.3)}`,
                      }}
                    >
                      {i === 0 ? 'THEN' : 'AND'}
                    </Box>
                    <TextField
                      select
                      size="small"
                      value={a.action_type}
                      onChange={(e) => {
                        const actions = [...editing.actions];
                        actions[i] = { ...a, action_type: e.target.value };
                        setEditing({ ...editing, actions });
                      }}
                      sx={{ width: { xs: '100%', sm: 215 }, flexShrink: 0 }}
                    >
                      {ACTION_TYPES.map((t) => (
                        <MenuItem key={t} value={t}>
                          {ACTION_LABELS[t] || t}
                        </MenuItem>
                      ))}
                    </TextField>
                    <Box sx={{ flex: 1, minWidth: 140, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <TagAutocomplete
                        value={a.value}
                        onChange={(newValue) => {
                          const actions = [...editing.actions];
                          actions[i] = { ...a, value: newValue };
                          setEditing({ ...editing, actions });
                        }}
                        type={a.action_type}
                        ruleDomain={editing.domain}
                        placeholder="Select or enter value..."
                      />
                      <Tooltip title="Remove action">
                        <IconButton
                          aria-label="Delete action"
                          size="small"
                          onClick={() => setEditing({ ...editing, actions: editing.actions.filter((_, j) => j !== i) })}
                        >
                          <DeleteOutline sx={{ fontSize: 16 }} />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Stack>
                </Box>
              ))}
              <Button
                size="small"
                variant="outlined"
                startIcon={<Add sx={{ fontSize: 14 }} />}
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
                <Science sx={{ fontSize: 16 }} /> Test against sample
              </Typography>
              <Button
                size="small"
                variant="text"
                startIcon={<AutoAwesome sx={{ fontSize: 14 }} />}
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
              startIcon={<PlayArrow sx={{ fontSize: 14 }} />}
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
              <Tooltip title={lockedHint}>
                <span>
                  <Button variant="contained" disabled={!canEditRules || saveMutation.isPending} onClick={save}>
                    Save rule
                  </Button>
                </span>
              </Tooltip>
              {!editing.rule_id.startsWith('new_rule_') && (
                <Button
                  variant="outlined"
                  color="error"
                  disabled={!canEditRules || deleteMutation.isPending}
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
