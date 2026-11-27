import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Button, Card, MenuItem, Stack, TextField } from '@mui/material';
import { ArrowBack } from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { ConfirmDialog } from '../components/common/ConfirmDialog';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { CompletedJobsTable } from '../components/skuResults/CompletedJobsTable';
import { ExportCsvMenu } from '../components/skuResults/ExportCsvMenu';
import { ProcessedSkusTable } from '../components/skuResults/ProcessedSkusTable';
import { SkuDetailDrawer } from '../components/skuResults/SkuDetailDrawer';
import { useJobs } from '../hooks/useJobs';
import { useSkuResults } from '../hooks/useSkuResults';
import { DOMAINS } from '../utils/constants';

export default function SkuResults() {
  const { enqueueSnackbar } = useSnackbar();
  const location = useLocation();
  const navigate = useNavigate();
  const navigatedJobId = location.state?.jobId;

  const [selectedJob, setSelectedJob] = useState(null);
  const [domainFilter, setDomainFilter] = useState('all');
  const [sheetFilter, setSheetFilter] = useState('');
  const [taskFilter, setTaskFilter] = useState('all');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const [selectedSku, setSelectedSku] = useState(null);
  const [rerunJob, setRerunJob] = useState(null);

  const [skuPage, setSkuPage] = useState(0);
  const [skuRowsPerPage, setSkuRowsPerPage] = useState(100);
  const [skuSearch, setSkuSearch] = useState('');
  const [skuBtFilter, setSkuBtFilter] = useState('');
  const [skuGkFilter, setSkuGkFilter] = useState('');
  const [skuSourceFilter, setSkuSourceFilter] = useState('all');

  useEffect(() => {
    setSkuPage(0);
    setSkuSearch('');
    setSkuBtFilter('');
    setSkuGkFilter('');
    setSkuSourceFilter('all');
  }, [selectedJob]);

  const {
    jobs,
    totalJobs,
    isLoading: isLoadingJobs,
  } = useJobs({
    page,
    limit: rowsPerPage,
    status: 'completed',
    domain: domainFilter,
    type: taskFilter,
    sheetName: sheetFilter,
    queryKeyPrefix: 'jobs_processed',
  });

  useEffect(() => {
    if (navigatedJobId && jobs && jobs.length > 0) {
      const job = jobs.find((j) => String(j.id) === String(navigatedJobId));
      if (job) {
        if (job.status !== 'completed') {
          setRerunJob(job);
        } else {
          setSelectedJob(job);
        }
        navigate(location.pathname, { replace: true, state: {} });
      }
    }
  }, [navigatedJobId, jobs, navigate, location.pathname]);

  const { skus, totalSkus, isLoadingSkus, exportRows, rerunIncompleteJob } = useSkuResults({
    selectedJob,
    page: skuPage,
    rowsPerPage: skuRowsPerPage,
    skuSearch,
    skuBtFilter,
    skuGkFilter,
    skuSourceFilter,
  });

  if (!selectedJob) {
    return (
      <PageContainer>
        <PageHeader title="SKU Results" subtitle="Select a job to view its SKU results." />

        <Card sx={{ p: 2, mb: 3 }}>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems={{ md: 'center' }}>
            <TextField
              size="small"
              label="Sheet Name"
              value={sheetFilter}
              onChange={(e) => {
                setSheetFilter(e.target.value);
                setPage(0);
              }}
              sx={{ width: 200 }}
              placeholder="Search sheets..."
            />
            <TextField
              select
              size="small"
              label="Domain"
              value={domainFilter}
              onChange={(e) => {
                setDomainFilter(e.target.value);
                setPage(0);
              }}
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
              label="Task"
              value={taskFilter}
              onChange={(e) => {
                setTaskFilter(e.target.value);
                setPage(0);
              }}
              sx={{ width: 160 }}
            >
              <MenuItem value="all">All tasks</MenuItem>
              <MenuItem value="matcher">Matcher</MenuItem>
              <MenuItem value="classifier">Classifier</MenuItem>
              <MenuItem value="pipeline">Pipeline</MenuItem>
            </TextField>
          </Stack>
        </Card>

        <CompletedJobsTable
          jobs={jobs}
          totalJobs={totalJobs}
          isLoading={isLoadingJobs}
          hasActiveFilters={Boolean(sheetFilter || domainFilter !== 'all' || taskFilter !== 'all')}
          page={page}
          rowsPerPage={rowsPerPage}
          onSelectJob={setSelectedJob}
          onResetFilters={() => {
            setSheetFilter('');
            setDomainFilter('all');
            setTaskFilter('all');
            setPage(0);
          }}
          onNavigateProcess={() => navigate('/process-skus')}
          onPageChange={(e, newPage) => setPage(newPage)}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(Number.parseInt(e.target.value, 10));
            setPage(0);
          }}
        />
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <Button
        startIcon={<ArrowBack sx={{ fontSize: 16 }} />}
        onClick={() => setSelectedJob(null)}
        sx={{ mb: 2 }}
        color="inherit"
      >
        Back to SKU Results
      </Button>
      <PageHeader
        title={selectedJob.target_sheet || selectedJob.sheet_name || `Job ${selectedJob.id}`}
        subtitle={`Domain: ${selectedJob.domain} | Total SKUs: ${selectedJob.total_items}`}
        actions={<ExportCsvMenu onExport={exportRows} />}
      />

      <Card sx={{ p: 2, mb: 3 }}>
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          spacing={2}
          alignItems={{ md: 'center' }}
          flexWrap="wrap"
          useFlexGap
        >
          <TextField
            size="small"
            label="Search SKU Name"
            value={skuSearch}
            onChange={(e) => {
              setSkuSearch(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 220 }}
            placeholder="Search SKU name..."
          />
          <TextField
            size="small"
            label="Basic Type (BT)"
            value={skuBtFilter}
            onChange={(e) => {
              setSkuBtFilter(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 180 }}
            placeholder="Filter BT..."
          />
          <TextField
            size="small"
            label="Generic Keywords (GK)"
            value={skuGkFilter}
            onChange={(e) => {
              setSkuGkFilter(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 180 }}
            placeholder="Filter GK..."
          />
          <TextField
            select
            size="small"
            label="Source"
            value={skuSourceFilter}
            onChange={(e) => {
              setSkuSourceFilter(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 160 }}
          >
            <MenuItem value="all">All Sources</MenuItem>
            <MenuItem value="matcher">Matcher</MenuItem>
            <MenuItem value="classifier">Classifier</MenuItem>
          </TextField>
        </Stack>
      </Card>

      <ProcessedSkusTable
        domain={selectedJob.domain}
        skus={skus}
        totalSkus={totalSkus}
        isLoading={isLoadingSkus}
        hasActiveFilters={Boolean(skuSearch || skuBtFilter || skuGkFilter || skuSourceFilter !== 'all')}
        page={skuPage}
        rowsPerPage={skuRowsPerPage}
        onSelectSku={setSelectedSku}
        onClearFilters={() => {
          setSkuSearch('');
          setSkuBtFilter('');
          setSkuGkFilter('');
          setSkuSourceFilter('all');
          setSkuPage(0);
        }}
        onPageChange={(e, newPage) => setSkuPage(newPage)}
        onRowsPerPageChange={(e) => {
          setSkuRowsPerPage(Number.parseInt(e.target.value, 10));
          setSkuPage(0);
        }}
      />

      <SkuDetailDrawer selectedSku={selectedSku} selectedJob={selectedJob} onClose={() => setSelectedSku(null)} />

      <ConfirmDialog
        open={Boolean(rerunJob)}
        onClose={() => setRerunJob(null)}
        title="Job Not Completed Fully"
        message={
          rerunJob
            ? `Job ${rerunJob.id} (${rerunJob.target_sheet || rerunJob.sheet_name || 'N/A'}) had a status of '${
                rerunJob.status
              }' and was not processed fully. Would you like to rerun it?`
            : ''
        }
        onConfirm={async () => {
          if (!rerunJob) return;
          try {
            await rerunIncompleteJob(rerunJob);
            setRerunJob(null);
            navigate('/jobs');
          } catch (err) {
            enqueueSnackbar(err.response?.data?.detail || 'Failed to rerun the job', { variant: 'error' });
            setRerunJob(null);
          }
        }}
        confirmText="Yes, Rerun"
        cancelText="No, Close"
      />
    </PageContainer>
  );
}
