import React, { useEffect, useMemo, useState } from 'react';
import { Button, Card, MenuItem, Stack, TextField, Tooltip } from '@mui/material';
import { Add } from '@mui/icons-material';
import { ConfirmDialog } from '../components/common/ConfirmDialog';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { RuleEditorDrawer } from '../components/rules/RuleEditorDrawer';
import { RulesTable } from '../components/rules/RulesTable';
import { useRules } from '../hooks/useRules';
import { ACTION_TYPES, DOMAINS } from '../utils/constants';
import { createEmptyRule, normalizeConditions } from '../utils/ruleUtils';

export default function RulesEngine() {
  const [domain, setDomain] = useState('all');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [confirm, setConfirm] = useState({ open: false, title: '', message: '', onConfirm: null });

  useEffect(() => {
    setPage(0);
  }, [domain]);

  const {
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
    isSaving,
    isDeleting,
    isTesting,
    saveEditedRule,
    deleteRuleById,
    runRuleTest,
    autoGenerateSample,
    updateConditions,
    handleDragStart,
    handleDragOver,
    handleDragEnd,
  } = useRules({ domain, page, rowsPerPage });

  const canEditRules = true;
  const lockedHint = '';

  const paginatedRows = useMemo(
    () => localRules.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage),
    [localRules, page, rowsPerPage]
  );

  const handleOpenNewRule = () => {
    setEditing(createEmptyRule(filtered));
    setTestResult(null);
  };

  const handleEditRule = (row) => {
    const ruleCopy = JSON.parse(JSON.stringify(row));
    ruleCopy.conditions = normalizeConditions(ruleCopy.conditions || [], ruleCopy.condition_logic || 'AND', true);
    ruleCopy.actions = (ruleCopy.actions || []).filter((a) => ACTION_TYPES.includes(a.action_type));
    setEditing(ruleCopy);
    setTestResult(null);
  };

  const handleRequestDelete = (ruleId) => {
    setConfirm({
      open: true,
      title: 'Delete Rule',
      message: `Are you sure you want to delete rule ${ruleId}?`,
      onConfirm: () => deleteRuleById(ruleId),
    });
  };

  return (
    <PageContainer>
      <PageHeader
        title="Rules Engine"
        subtitle="Domain-scoped rules. Edit conditions/actions and test against a sample."
        actions={
          <Tooltip title={lockedHint}>
            <span>
              <Button
                variant="contained"
                startIcon={<Add sx={{ fontSize: 16 }} />}
                disabled={!canEditRules}
                onClick={handleOpenNewRule}
              >
                New rule
              </Button>
            </span>
          </Tooltip>
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

      <RulesTable
        rows={paginatedRows}
        totalCount={filtered.length}
        isLoading={isLoading}
        domain={domain}
        canEditRules={canEditRules}
        lockedHint={lockedHint}
        page={page}
        rowsPerPage={rowsPerPage}
        draggedIndex={draggedIndex}
        onDragStart={handleDragStart}
        onDragOver={handleDragOver}
        onDragEnd={handleDragEnd}
        onEditRule={handleEditRule}
        onDeleteRule={handleRequestDelete}
        onNewRule={handleOpenNewRule}
        onPageChange={(event, newPage) => setPage(newPage)}
        onRowsPerPageChange={(event) => {
          setRowsPerPage(Number.parseInt(event.target.value, 10));
          setPage(0);
        }}
      />

      <RuleEditorDrawer
        editing={editing}
        onChangeEditing={setEditing}
        onClose={() => setEditing(null)}
        onUpdateConditions={updateConditions}
        canEditRules={canEditRules}
        lockedHint={lockedHint}
        sample={sample}
        onSampleChange={setSample}
        onAutoGenerateSample={autoGenerateSample}
        onRunTest={runRuleTest}
        isTesting={isTesting}
        testResult={testResult}
        isSaving={isSaving}
        isDeleting={isDeleting}
        onSave={saveEditedRule}
        onRequestDelete={handleRequestDelete}
      />

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
