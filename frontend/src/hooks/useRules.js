import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { deleteRule, getRules, reorderRules, saveRule, testRule } from '../services/ruleApiService';
import { EMPTY_RULES, generateSampleForRule, normalizeConditions } from '../utils/ruleUtils';

export function useRules({ domain, page, rowsPerPage }) {
  const { enqueueSnackbar } = useSnackbar();
  const queryClient = useQueryClient();

  const [editing, setEditing] = useState(null);
  const [sample, setSample] = useState('{\n  "sku_name": "Red Bull Energy Drink 250ml",\n  "domain": "market"\n}');
  const [testResult, setTestResult] = useState(null);
  const [localRules, setLocalRules] = useState([]);
  const [draggedIndex, setDraggedIndex] = useState(null);

  const { data: serverRules = EMPTY_RULES, isLoading } = useQuery({
    queryKey: ['rules', domain],
    queryFn: () => getRules({ ...(domain !== 'all' && { domain }) }),
  });

  const filtered = useMemo(() => {
    const res = [...serverRules];
    res.sort((a, b) => a.priority - b.priority);
    return res;
  }, [serverRules]);

  useEffect(() => {
    setLocalRules(filtered);
  }, [filtered]);

  const reorderMutation = useMutation({
    mutationFn: reorderRules,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rules'] });
      enqueueSnackbar('Rules reordered successfully', { variant: 'success' });
    },
    onError: (err) => enqueueSnackbar(`Reorder failed: ${err.message}`, { variant: 'error' }),
  });

  const saveMutation = useMutation({
    mutationFn: saveRule,
    onSuccess: (data) => {
      const savedId = data?.rule_id || editing?.rule_id;
      enqueueSnackbar(`Rule ${savedId} saved`, { variant: 'success' });
      setEditing(null);
      queryClient.invalidateQueries({ queryKey: ['rules'] });
    },
    onError: (err) => enqueueSnackbar(`Save failed: ${err.message}`, { variant: 'error' }),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteRule,
    onSuccess: (_, deletedId) => {
      enqueueSnackbar(`Rule ${deletedId} deleted`, { variant: 'success' });
      setEditing(null);
      queryClient.invalidateQueries({ queryKey: ['rules'] });
    },
    onError: (err) => enqueueSnackbar(`Delete failed: ${err.message}`, { variant: 'error' }),
  });

  const testMutation = useMutation({
    mutationFn: testRule,
    onSuccess: (data) => {
      setTestResult(data);
    },
    onError: (err) => enqueueSnackbar(`Test failed: ${err.message}`, { variant: 'error' }),
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

  const runRuleTest = () => {
    let parsed;
    try {
      parsed = JSON.parse(sample);
    } catch {
      enqueueSnackbar('Sample is not valid JSON', { variant: 'error' });
      return;
    }
    testMutation.mutate({ rule: editing, sampleRecord: parsed });
  };

  const autoGenerateSample = (rule) => {
    const sampleObj = generateSampleForRule(rule);
    if (!sampleObj) return;
    setSample(JSON.stringify(sampleObj, null, 2));
    enqueueSnackbar('Auto-generated test sample matching rule conditions', { variant: 'info' });
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

  return {
    filtered,
    localRules,
    isLoading,
    draggedIndex,
    editing,
    setEditing,
    sample,
    setSample,
    testResult,
    setTestResult,
    isSaving: saveMutation.isPending,
    isDeleting: deleteMutation.isPending,
    isTesting: testMutation.isPending,
    saveEditedRule: () => saveMutation.mutate(editing),
    deleteRuleById: (id) => deleteMutation.mutate(id),
    runRuleTest,
    autoGenerateSample,
    updateConditions,
    handleDragStart,
    handleDragOver,
    handleDragEnd,
  };
}
