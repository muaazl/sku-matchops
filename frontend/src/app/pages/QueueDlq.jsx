import React, { useState } from 'react';
import { Alert, Box, Button, Chip, Table, TableCell, TableRow, Tooltip, Typography } from '@mui/material';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { RotateCcw } from 'lucide-react';
import { getDeadLetters, replayDeadLetter } from '../api';
import { useAuth } from '../hooks/useAuth';
import { fmtTime } from '../utils';
import { ConfirmDialog, PageContainer, PageHeader } from '../components/ui';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
  TableSkeleton,
} from '../../components/Common/StyledTable';

const COLUMNS = ['Task', 'Job / Batch', 'Tier', 'Error', 'Attempts', 'Failed at', ''];
const MONO = { fontFamily: 'monospace' };

export default function QueueDlq() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const { isAdmin, role } = useAuth();
  const [confirm, setConfirm] = useState(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['dlq'],
    queryFn: getDeadLetters,
    enabled: isAdmin, // the API is admin-only; don't fire a guaranteed 403
    refetchInterval: 15000,
  });

  const replay = useMutation({
    mutationFn: replayDeadLetter,
    onSuccess: (res) => {
      enqueueSnackbar(`Job ${res.job_id} re-queued`, { variant: 'success' });
      queryClient.invalidateQueries({ queryKey: ['dlq'] });
      queryClient.invalidateQueries({ queryKey: ['system-status'] });
    },
    onError: (e) => enqueueSnackbar(e.response?.data?.detail || e.message, { variant: 'error' }),
  });

  const status = error?.response?.status;
  return (
    <PageContainer>
      <PageHeader
        title="Queue & Dead Letters"
        subtitle="Jobs that exhausted their retries. Replay re-queues one with a fresh retry budget."
      />

      {!isAdmin && role && (
        <Alert severity="info" sx={{ mb: 2 }}>
          The dead-letter queue is available to admins. Your role ({role.replace('_', ' ')}) cannot view or replay
          tasks.
        </Alert>
      )}
      {status === 503 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          The Redis task queue is offline or disabled; jobs are being dispatched directly to the engine.
        </Alert>
      )}

      {isAdmin && (
        <StyledTableContainer>
          <Table>
            <StyledTableHead>
              <TableRow>
                {COLUMNS.map((c) => (
                  <StyledHeaderCell key={c} align={c === '' || c === 'Attempts' ? 'right' : 'left'}>
                    {c}
                  </StyledHeaderCell>
                ))}
              </TableRow>
            </StyledTableHead>
            <StyledTableBody>
              {isLoading ? (
                <TableSkeleton columns={COLUMNS.length} />
              ) : !data?.length ? (
                <TableRow>
                  <TableCell colSpan={COLUMNS.length} align="center">
                    <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                      No dead-lettered jobs.
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                data.map((item) => (
                  <StyledTableRow key={item.task_id} hover>
                    <TableCell sx={MONO}>{item.task_id.slice(0, 12)}</TableCell>
                    <TableCell sx={MONO}>
                      {item.job_id}
                      {item.batch_id && item.batch_id !== item.job_id ? ` / ${item.batch_id}` : ''}
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        variant="outlined"
                        label={item.queue?.split(':').pop() || 'n/a'}
                        sx={{ textTransform: 'capitalize' }}
                      />
                    </TableCell>
                    <TableCell sx={{ maxWidth: 420 }}>
                      <Tooltip title={item.error}>
                        <Box
                          sx={{
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            color: 'error.main',
                          }}
                        >
                          {item.error}
                        </Box>
                      </Tooltip>
                    </TableCell>
                    <TableCell align="right" sx={MONO}>
                      {item.attempts}
                    </TableCell>
                    <TableCell>{fmtTime(item.failed_at)}</TableCell>
                    <TableCell align="right">
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={<RotateCcw size={14} />}
                        disabled={replay.isPending}
                        onClick={() =>
                          setConfirm({
                            title: 'Replay job',
                            message: `Re-queue job ${item.job_id} with a fresh retry budget?`,
                            onConfirm: () => replay.mutate(item.task_id),
                          })
                        }
                      >
                        Replay
                      </Button>
                    </TableCell>
                  </StyledTableRow>
                ))
              )}
            </StyledTableBody>
          </Table>
        </StyledTableContainer>
      )}

      <ConfirmDialog
        open={Boolean(confirm)}
        onClose={() => setConfirm(null)}
        title={confirm?.title || ''}
        message={confirm?.message || ''}
        onConfirm={() => {
          confirm.onConfirm();
          setConfirm(null);
        }}
        confirmText="Replay"
        cancelText="Cancel"
      />
    </PageContainer>
  );
}
