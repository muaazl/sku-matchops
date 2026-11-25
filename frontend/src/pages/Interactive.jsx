import React from 'react';
import { Card, CircularProgress, Grid, Stack, Typography } from '@mui/material';
import { AutoAwesome } from '@mui/icons-material';
import { EmptyState } from '../components/common/EmptyState';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import AuditTelemetryView from '../components/interactive/AuditTelemetryView';
import { InteractiveInputForm } from '../components/interactive/InteractiveInputForm';
import { InteractiveResultCard } from '../components/interactive/InteractiveResultCard';
import { TemplateSuggestionsCard } from '../components/interactive/TemplateSuggestionsCard';
import { useSkuMatch } from '../hooks/useSkuMatch';

export default function Interactive() {
  const {
    viewMode,
    setViewMode,
    auditTrace,
    auditLoading,
    sku,
    setSku,
    price,
    setPrice,
    description,
    setDescription,
    category,
    setCategory,
    task,
    setTask,
    domain,
    setDomain,
    result,
    resultTask,
    loading,
    suggestions,
    suggestionsLoading,
    enableSuggestions,
    setEnableSuggestions,
    runSingleMatch,
    runAudit,
  } = useSkuMatch();

  const item = result?.results?.[0];

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
      <PageHeader title="Interactive" subtitle="Run a single SKU through the matcher, classifier or full pipeline." />

      <Grid container spacing={3} alignItems="flex-start">
        <Grid size={{ xs: 12, md: 5 }}>
          <InteractiveInputForm
            sku={sku}
            onSkuChange={setSku}
            price={price}
            onPriceChange={setPrice}
            category={category}
            onCategoryChange={setCategory}
            description={description}
            onDescriptionChange={setDescription}
            domain={domain}
            onDomainChange={setDomain}
            task={task}
            onTaskChange={setTask}
            enableSuggestions={enableSuggestions}
            onToggleSuggestions={setEnableSuggestions}
            loading={loading}
            onRun={runSingleMatch}
          />
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

          {!loading && !item && (
            <Card
              sx={{
                p: 4,
                border: '2px dashed',
                borderColor: 'divider',
                borderRadius: 2,
                backgroundColor: 'action.hover',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 400,
              }}
            >
              <EmptyState
                icon={AutoAwesome}
                title="Ready to Process"
                description={
                  'Enter the SKU details on the left and choose a task. The results, confidence metrics, ' +
                  'and matching logic will appear here.'
                }
                minHeight={320}
              />
            </Card>
          )}

          {item && !loading && (
            <Stack spacing={3}>
              <InteractiveResultCard
                item={item}
                resolvedDomain={result?.domain || domain}
                auditLoading={auditLoading}
                onAudit={runAudit}
              />
              <TemplateSuggestionsCard suggestions={suggestions} loading={suggestionsLoading} />
            </Stack>
          )}
        </Grid>
      </Grid>
    </PageContainer>
  );
}
