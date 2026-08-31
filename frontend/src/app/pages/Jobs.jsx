import React, { useMemo, useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  Box,
  IconButton,
  Tooltip,
  ToggleButtonGroup,
  ToggleButton,
  Alert,
  Table,
  TableCell,
  TableRow,
  TablePagination,
  Typography,
} from '@mui/material';
import { XCircle, RotateCw, RefreshCw } from 'lucide-react';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader, StatusChip, ConfirmDialog } from '../components/ui';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getJobs, cancelJob, retryJob } from '../api';
import JobDetailModal from './JobDetailModal';
import { fmtDuration } from '../utils';
import {
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledHeaderCell,
  StyledTableRow,
  TableSkeleton,
} from '../../components/Common/StyledTable';

const STATUS_FILTERS = ['all', 'running', 'queued', 'completed', 'failed', 'cancelled'];

export default function Jobs() {
  const navigate = useNavigate();
  const { jobId } = useParams();
  const { enqueueSnackbar } = useSnackbar();
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState('all');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [modalJobId, setModalJobId] = useState(jobId || null);

  useEffect(() => {
    if (jobId) {
      setModalJobId(jobId);
    }
  }, [jobId]);

  // Confirmation Modal State
  const [confirm, setConfirm] = useState({ open: false, title: '', message: '', onConfirm: null });

  const {
    data: serverJobs,
    isLoading,
    isFetching,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => getJobs({}),
    refetchInterval: (query) => {
      // Poll if any jobs are running or queued
      const d = query.state.data;
      if (d && d.some((j) => j.status === 'running' || j.status === 'queued')) {
        return 3000;
      }
      return false;
    },
  });

  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      enqueueSnackbar('Job cancellation requested', { variant: 'info' });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
    onError: (e) => enqueueSnackbar(`Cancel failed: ${e.message}`, { variant: 'error' }),
  });

  const retryMutation = useMutation({
    mutationFn: retryJob,
    onSuccess: (data) => {
      enqueueSnackbar(`Retry initiated! New Job ID: ${data.new_job_id}`, { variant: 'success' });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
    onError: (e) => enqueueSnackbar(`Retry failed: ${e.message}`, { variant: 'error' }),
  });

  const rows = useMemo(() => {
    if (!serverJobs) return [];
    const sorted = [...serverJobs].sort((a, b) => {
      const aActive = ['running', 'queued'].includes(a.status);
      const bActive = ['running', 'queued'].includes(b.status);
      if (aActive && !bActive) return -1;
      if (!aActive && bActive) return 1;
      // newest first by numerical ID
      const aId = parseInt(a.id) || 0;
      const bId = parseInt(b.id) || 0;
      if (aId !== bId) return bId - aId;
      return new Date(b.started_at) - new Date(a.started_at);
    });
    return filter === 'all' ? sorted : sorted.filter((j) => j.status === filter);
  }, [filter, serverJobs]);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const paginatedRows = useMemo(() => {
    return rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [rows, page, rowsPerPage]);

  return (
    <PageContainer>
      <PageHeader
        title="Jobs"
        subtitle="Batch runs, CSV uploads, and retries. Click a row for live detail."
        actions={
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
            <ToggleButtonGroup
              size="small"
              value={filter}
              exclusive
              onChange={(e, v) => v && setFilter(v)}
              sx={{ flexWrap: 'wrap' }}
            >
              {STATUS_FILTERS.map((s) => (
                <ToggleButton key={s} value={s} sx={{ textTransform: 'capitalize', px: 1.5 }}>
                  {s}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            <Tooltip title="Refresh Jobs">
              <IconButton aria-label="Refresh Jobs" onClick={() => refetch()} disabled={isLoading || isFetching}>
                <RefreshCw size={18} className={isFetching ? 'animate-spin' : ''} />
              </IconButton>
            </Tooltip>
          </Box>
        }
      />
      <Box sx={{ mb: 3 }}>
        {isError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error.message}
          </Alert>
        )}
        <StyledTableContainer>
          <Table aria-label="jobs table">
            <StyledTableHead>
              <TableRow>
                <StyledHeaderCell width="12%">Job ID</StyledHeaderCell>
                <StyledHeaderCell width="25%">Sheet Name</StyledHeaderCell>
                <StyledHeaderCell width="15%">Domain</StyledHeaderCell>
                <StyledHeaderCell width="15%">Task</StyledHeaderCell>
                <StyledHeaderCell width="15%">Status</StyledHeaderCell>
                <StyledHeaderCell align="right">Total SKUs</StyledHeaderCell>
                <StyledHeaderCell align="right">High Conf</StyledHeaderCell>
                <StyledHeaderCell align="right">Duration</StyledHeaderCell>
                <StyledHeaderCell align="right" width="10%">
                  Actions
                </StyledHeaderCell>
              </TableRow>
            </StyledTableHead>
            <StyledTableBody>
              {isLoading ? (
                <TableSkeleton columns={9} rows={5} />
              ) : paginatedRows.length === 0 ? (
                <StyledTableRow>
                  <TableCell colSpan={9} align="center">
                    <Typography variant="body2" color="text.secondary">
                      No jobs found.
                    </Typography>
                  </TableCell>
                </StyledTableRow>
              ) : (
                paginatedRows.map((row) => {
                  const canCancel = ['running', 'queued'].includes(row.status);
                  const rowId = row.id;
                  return (
                    <StyledTableRow
                      key={row.id}
                      hover
                      onClick={() => {
                        if (['running', 'queued'].includes(row.status)) {
                          setModalJobId(row.id);
                        } else if (row.status === 'completed') {
                          navigate('/sku-results', { state: { jobId: row.id } });
                        } else {
                          setConfirm({
                            open: true,
                            title: 'Job Not Completed Fully',
                            message: `Job ${rowId} (${row.target_sheet || row.sheet_name || 'N/A'}) had a status of '${
                              row.status
                            }' and was not processed fully. Would you like to rerun it?`,
                            onConfirm: () => retryMutation.mutate(rowId),
                          });
                        }
                      }}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>{row.id}</TableCell>
                      <TableCell>{row.target_sheet || row.sheet_name || 'N/A'}</TableCell>
                      <TableCell sx={{ textTransform: 'capitalize' }}>{row.domain}</TableCell>
                      <TableCell sx={{ textTransform: 'capitalize' }}>{row.task || row.type}</TableCell>
                      <TableCell>
                        <StatusChip status={row.status} />
                      </TableCell>
                      <TableCell align="right">{row.total_items ?? '-'}</TableCell>
                      <TableCell align="right">{row.high_conf ?? '-'}</TableCell>
                      <TableCell align="right">{fmtDuration(row.duration_minutes)}</TableCell>
                      <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 0.5 }}>
                          <Tooltip title={canCancel ? 'Cancel job' : 'Nothing to cancel'}>
                            <span>
                              <IconButton
                                aria-label={canCancel ? 'Cancel job' : 'Nothing to cancel'}
                                size="small"
                                disabled={!canCancel || cancelMutation.isPending}
                                onClick={() => {
                                  setConfirm({
                                    open: true,
                                    title: 'Cancel Job',
                                    message: `Are you sure you want to cancel Job ${rowId}?`,
                                    onConfirm: () => cancelMutation.mutate(rowId),
                                  });
                                }}
                              >
                                <XCircle size={17} />
                              </IconButton>
                            </span>
                          </Tooltip>
                          <Tooltip title="Retry job">
                            <IconButton
                              aria-label="Retry job"
                              size="small"
                              disabled={retryMutation.isPending}
                              onClick={() => {
                                setConfirm({
                                  open: true,
                                  title: 'Retry Job',
                                  message: `Are you sure you want to retry Job ${rowId}?`,
                                  onConfirm: () => retryMutation.mutate(rowId),
                                });
                              }}
                            >
                              <RotateCw size={16} />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </StyledTableRow>
                  );
                })
              )}
            </StyledTableBody>
          </Table>
          <TablePagination
            rowsPerPageOptions={[10, 25, 50, 100]}
            component="div"
            count={rows.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
            sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
          />
        </StyledTableContainer>
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
