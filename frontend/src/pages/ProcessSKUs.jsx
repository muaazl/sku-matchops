import React from 'react';
import { Grid } from '@mui/material';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { BearerTokenModal } from '../components/process/BearerTokenModal';
import { BulkBatchQueue } from '../components/process/BulkBatchQueue';
import { CsvImportDialog } from '../components/process/CsvImportDialog';
import { CsvIntakeCalloutCard } from '../components/process/CsvIntakeCalloutCard';
import { MerchantPortalCard } from '../components/process/MerchantPortalCard';
import { SingleBatchPreview } from '../components/process/SingleBatchPreview';
import { useBatchProcess } from '../hooks/useBatchProcess';

export default function ProcessSKUs() {
  const {
    csvModalOpen,
    setCsvModalOpen,
    uploadMode,
    setUploadMode,
    domain,
    setDomain,
    task,
    setTask,
    file,
    setFile,
    pastedText,
    setPastedText,
    bulkFiles,
    setBulkFiles,
    pendingJobs,
    localMerchantId,
    setLocalMerchantId,
    tokenModalOpen,
    setTokenModalOpen,
    localToken,
    setLocalToken,
    fetchingPortal,
    previewData,
    jobName,
    setJobName,
    isCreatingBatch,
    triggerPortalFetch,
    submitTokenAndFetch,
    pickSingleFile,
    confirmCsvImport,
    startSingleJob,
    cancelSinglePreview,
    pickBulkFiles,
    confirmBulkImport,
    updatePendingJob,
    startPendingJob,
    startAllPendingJobs,
    removePendingJob,
    navigateToJobs,
  } = useBatchProcess();

  if (pendingJobs) {
    return (
      <BulkBatchQueue
        pendingJobs={pendingJobs}
        onUpdateJob={updatePendingJob}
        onStartJob={startPendingJob}
        onRemoveJob={removePendingJob}
        onStartAll={startAllPendingJobs}
        onDone={navigateToJobs}
      />
    );
  }

  if (previewData) {
    return (
      <SingleBatchPreview
        previewData={previewData}
        jobName={jobName}
        onJobNameChange={setJobName}
        isCreatingBatch={isCreatingBatch}
        onCancel={cancelSinglePreview}
        onStart={startSingleJob}
      />
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Process SKUs"
        subtitle={
          'Fetch product listings from a merchant portal or upload custom CSV data ' +
          'to kick off a SKU processing job.'
        }
      />

      <Grid container spacing={3} alignItems="flex-start" sx={{ mt: 1 }}>
        <Grid size={{ xs: 12, md: 5 }}>
          <MerchantPortalCard
            merchantId={localMerchantId}
            onMerchantIdChange={setLocalMerchantId}
            domain={domain}
            onDomainChange={setDomain}
            task={task}
            onTaskChange={setTask}
            fetchingPortal={fetchingPortal}
            onFetch={triggerPortalFetch}
          />
        </Grid>

        <Grid size={{ xs: 12, md: 7 }}>
          <CsvIntakeCalloutCard
            file={file}
            pastedText={pastedText}
            onClearFile={() => setFile(null)}
            onClearPastedText={() => setPastedText('')}
            onOpenModal={() => setCsvModalOpen(true)}
          />
        </Grid>
      </Grid>

      <CsvImportDialog
        open={csvModalOpen}
        onClose={() => setCsvModalOpen(false)}
        uploadMode={uploadMode}
        onUploadModeChange={setUploadMode}
        file={file}
        onPickSingleFile={pickSingleFile}
        onClearFile={() => setFile(null)}
        pastedText={pastedText}
        onPastedTextChange={setPastedText}
        bulkFiles={bulkFiles}
        onPickBulkFiles={pickBulkFiles}
        onRemoveBulkFile={(idx) => setBulkFiles((prev) => prev.filter((_, i) => i !== idx))}
        domain={domain}
        onDomainChange={setDomain}
        task={task}
        onTaskChange={setTask}
        isSubmitting={isCreatingBatch}
        onConfirmSingle={confirmCsvImport}
        onConfirmBulk={confirmBulkImport}
      />

      <BearerTokenModal
        open={tokenModalOpen}
        onClose={() => setTokenModalOpen(false)}
        token={localToken}
        onTokenChange={setLocalToken}
        onSubmit={submitTokenAndFetch}
      />
    </PageContainer>
  );
}
