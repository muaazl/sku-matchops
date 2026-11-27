import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getJob, subscribeJobDetailStream } from '../services/jobApiService';
import { computeElapsedSeconds, fmtDuration, fmtEta } from '../utils/formatters';
import { useJobStream } from './useJobStream';

export function useJobDetail({ jobId, open }) {
  const queryClient = useQueryClient();
  const [sseFallback, setSseFallback] = useState(false);

  const {
    data: job,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId),
    enabled: !!jobId && open,
    refetchInterval: (query) => {
      if (!sseFallback) return false;
      const j = query.state.data;
      return j && (j.status === 'running' || j.status === 'queued') ? 2500 : false;
    },
  });

  const isActive = job?.status === 'running' || job?.status === 'queued';

  const { sseFallback: streamFallback, resetFallback } = useJobStream({
    enabled: Boolean(open && jobId && isActive),
    streamKey: `${jobId}-${open}`,
    subscribe: (callbacks) => subscribeJobDetailStream(jobId, callbacks),
    onMessage: (evt) => {
      queryClient.setQueryData(['job', jobId], (old) => (old ? { ...old, ...evt } : evt));
      if (['completed', 'failed', 'cancelled'].includes(evt.status)) {
        queryClient.invalidateQueries({ queryKey: ['job', jobId] });
        queryClient.invalidateQueries({ queryKey: ['jobs'] });
      }
    },
  });

  useEffect(() => {
    setSseFallback(streamFallback);
  }, [streamFallback]);

  useEffect(() => {
    resetFallback();
    setSseFallback(false);
  }, [jobId, open, resetFallback]);

  const [liveEta, setLiveEta] = useState(null);
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    setLiveEta(job?.eta_seconds ?? null);
  }, [job?.eta_seconds, job?.progress_pct, job?.current_stage, job?.status]);

  useEffect(() => {
    if (!open || job?.status !== 'running') return undefined;
    const timer = setInterval(() => {
      setLiveEta((prev) => (typeof prev === 'number' && prev > 1 ? prev - 1 : prev));
      setNowMs(Date.now());
    }, 1000);
    return () => clearInterval(timer);
  }, [open, job?.status]);

  let liveDurationText = fmtDuration(job?.duration_minutes);
  if (
    (job?.duration_minutes === null || job?.duration_minutes === undefined) &&
    job?.status === 'running' &&
    job?.started_at
  ) {
    const elapsedSec = computeElapsedSeconds(job.started_at, nowMs);
    if (elapsedSec !== null) {
      liveDurationText = fmtEta(elapsedSec);
    }
  }

  return {
    job,
    isLoading,
    isError,
    liveEta,
    liveDurationText,
  };
}
