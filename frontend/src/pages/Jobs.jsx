import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Alert, Box, Button, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { ConfirmDialog } from '../components/common/ConfirmDialog';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import JobDetailModal from '../components/jobs/JobDetailModal';
import { JobsTable } from '../components/jobs/JobsTable';
import { useJobs } from '../hooks/useJobs';
import { JOB_STATUS_FILTERS } from '../utils/constants';

export default function Jobs() {
  const navigate = useNavigate();
  const { jobId } = useParams();
  const [filter, setFilter] = useState('all');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [modalJobId, setModalJobId] = useState(jobId || null);
  const [confirm, setConfirm] = useState({ open: false, title: '', message: '', onConfirm: null });

  useEffect(() => {
    if (jobId) {
      setModalJobId(jobId);
    }
  }, [jobId]);

  useEffect(() => {
    setPage(0);
  }, [filter]);

  const {
    jobs,
    totalJobs,
    isLoading,
    isFetching,
    isError,
    error,
    refetch,
    cancelJob,
    isCancelling,
    retryJob,
    isRetrying,
  } = useJobs({
    page,
    limit: rowsPerPage,
    status: filter,
    enableStream: true,
  });

  const handleRowClick = (row, isUpload) => {
    if (isUpload && row.status === 'failed') return;

    if (['running', 'queued'].includes(row.status)) {
      setModalJobId(row.id);
    } else if (row.status === 'completed') {
      navigate('/sku-results', { state: { jobId: row.id } });
    } else {
      setConfirm({
        open: true,
        title: 'Job Not Completed Fully',
        message: `Job ${row.id} (${row.target_sheet || row.sheet_name || 'N/A'}) had a status of '${
          row.status
        }' and was not processed fully. Would you like to rerun it?`,
        onConfirm: () => retryJob(row.id),
      });
    }
  };

  return (
    <PageContainer>
      <PageHeader
        title="Jobs"
        subtitle="Batch runs, CSV uploads, and retries. Click a row for live detail."
        actions={
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
            <ToggleButtonGroup
              size="small"
              value={filter}
              exclusive
              onChange={(e, v) => v && setFilter(v)}
              aria-label="Filter jobs by status"
              sx={{ flexWrap: 'wrap' }}
            >
              {JOB_STATUS_FILTERS.map((s) => (
                <ToggleButton key={s} value={s} sx={{ textTransform: 'capitalize', px: 1.5, height: 40 }}>
                  {s}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            <Button
              variant="outlined"
              color="inherit"
              size="small"
              aria-label="Refresh Jobs"
              onClick={() => refetch()}
              disabled={isLoading || isFetching}
              startIcon={
                <Refresh
                  sx={{
                    fontSize: 16,
                    ...(isFetching && {
                      '@keyframes spin': {
                        from: { transform: 'rotate(0deg)' },
                        to: { transform: 'rotate(360deg)' },
                      },
                      animation: 'spin 0.9s linear infinite',
                    }),
                  }}
                />
              }
              sx={{ height: 40, px: 1.75, flexShrink: 0 }}
            >
              Refresh
            </Button>
          </Box>
        }
      />
      <Box sx={{ mb: 3 }}>
        {isError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error.message}
          </Alert>
        )}
        <JobsTable
          jobs={jobs}
          totalJobs={totalJobs}
          isLoading={isLoading}
          filter={filter}
          page={page}
          rowsPerPage={rowsPerPage}
          isCancelling={isCancelling}
          isRetrying={isRetrying}
          onRowClick={handleRowClick}
          onCancelClick={(rowId) =>
            setConfirm({
              open: true,
              title: 'Cancel Job',
              message: `Are you sure you want to cancel Job ${rowId}?`,
              onConfirm: () => cancelJob(rowId),
            })
          }
          onRetryClick={(rowId) =>
            setConfirm({
              open: true,
              title: 'Retry Job',
              message: `Are you sure you want to retry Job ${rowId}?`,
              onConfirm: () => retryJob(rowId),
            })
          }
          onResetFilter={() => setFilter('all')}
          onNavigateProcess={() => navigate('/process-skus')}
          onPageChange={(event, newPage) => setPage(newPage)}
          onRowsPerPageChange={(event) => {
            setRowsPerPage(Number.parseInt(event.target.value, 10));
            setPage(0);
          }}
        />
      </Box>

      <ConfirmDialog
        open={confirm.open}
        onClose={() => setConfirm((prev) => ({ ...prev, open: false }))}
        title={confirm.title}
        message={confirm.message}
        onConfirm={() => {
          if (confirm.onConfirm) confirm.onConfirm();
          setConfirm((prev) => ({ ...prev, open: false }));
        }}
      />

      <JobDetailModal
        jobId={modalJobId}
        open={!!modalJobId}
        onClose={() => {
          setModalJobId(null);
          if (jobId) {
            navigate('/jobs', { replace: true });
          }
        }}
      />
    </PageContainer>
  );
}
