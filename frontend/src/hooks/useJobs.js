import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { cancelJob, getJobs, retryJob, subscribeJobsStream } from '../services/jobApiService';
import { useJobStream } from './useJobStream';

export function useJobs({
  page,
  limit,
  status,
  domain,
  type,
  sheetName,
  queryKeyPrefix = 'jobs',
  enableStream = false,
}) {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const [sseFallback, setSseFallback] = useState(false);

  const query = useQuery({
    queryKey: [queryKeyPrefix, page, limit, status, domain, type, sheetName],
    queryFn: () =>
      getJobs({
        page: page + 1,
        limit,
        ...(status && status !== 'all' ? { status } : {}),
        ...(domain && domain !== 'all' ? { domain } : {}),
        ...(type && type !== 'all' ? { type } : {}),
        ...(sheetName?.trim() ? { sheet_name: sheetName.trim() } : {}),
      }),
    refetchInterval: (q) => {
      if (!enableStream || !sseFallback) return false;
      const rows = q.state.data?.data;
      if (rows && rows.some((j) => j.status === 'running' || j.status === 'queued')) {
        return 3000;
      }
      return false;
    },
  });

  const jobs = query.data?.data || [];
  const totalJobs = query.data?.total || 0;
  const hasActiveJobs = jobs.some((j) => j.status === 'running' || j.status === 'queued');

  const { sseFallback: streamFallback } = useJobStream({
    enabled: enableStream && hasActiveJobs,
    subscribe: subscribeJobsStream,
    onMessage: (evt) => {
      if (evt.type === 'snapshot' && Array.isArray(evt.jobs)) {
        const byId = new Map(evt.jobs.map((j) => [String(j.id), j]));
        let finishedDuringConnect = false;
        queryClient.setQueriesData({ queryKey: [queryKeyPrefix] }, (old) => {
          if (!old?.data) return old;
          return {
            ...old,
            data: old.data.map((r) => {
              const snap = byId.get(String(r.id));
              if (
                snap &&
                (r.status === 'running' || r.status === 'queued') &&
                !['running', 'queued'].includes(snap.status)
              ) {
                finishedDuringConnect = true;
              }
              return snap || r;
            }),
          };
        });
        if (finishedDuringConnect) {
          queryClient.invalidateQueries({ queryKey: [queryKeyPrefix] });
        }
        return;
      }
      if (!evt.id) return;
      queryClient.setQueriesData({ queryKey: [queryKeyPrefix] }, (old) =>
        old?.data
          ? {
              ...old,
              data: old.data.map((r) => (String(r.id) === String(evt.id) ? { ...r, ...evt } : r)),
            }
          : old
      );
      if (['queued', 'completed', 'failed', 'cancelled'].includes(evt.status)) {
        queryClient.invalidateQueries({ queryKey: [queryKeyPrefix] });
      }
    },
  });

  useEffect(() => {
    setSseFallback(streamFallback);
  }, [streamFallback]);

  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      enqueueSnackbar('Job cancellation requested', { variant: 'info' });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
    onError: (err) => enqueueSnackbar(`Cancel failed: ${err.message}`, { variant: 'error' }),
  });

  const retryMutation = useMutation({
    mutationFn: retryJob,
    onSuccess: (data) => {
      enqueueSnackbar(`Retry initiated! New Job ID: ${data.new_job_id}`, { variant: 'success' });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
    onError: (err) => enqueueSnackbar(`Retry failed: ${err.message}`, { variant: 'error' }),
  });

  return {
    jobs,
    totalJobs,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
    cancelJob: cancelMutation.mutate,
    isCancelling: cancelMutation.isPending,
    retryJob: retryMutation.mutate,
    retryJobAsync: retryMutation.mutateAsync,
    isRetrying: retryMutation.isPending,
  };
}
