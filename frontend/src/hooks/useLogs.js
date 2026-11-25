import { useCallback, useEffect, useMemo, useState } from 'react';
import { useSnackbar } from 'notistack';
import { getLogs } from '../services/systemApiService';
import { countNonEmptyLogLines, parseLogLines } from '../utils/logUtils';

export function useLogs({ lines, filterQuery, autoRefresh }) {
  const { enqueueSnackbar } = useSnackbar();
  const [logsText, setLogsText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const fetchLogs = useCallback(
    async (silent = false) => {
      if (!silent) setIsLoading(true);
      try {
        const data = await getLogs(lines);
        setLogsText(data.logs || '');
      } catch (err) {
        if (!silent) {
          enqueueSnackbar(`Failed to fetch logs: ${err.message}`, { variant: 'error' });
        }
      } finally {
        if (!silent) setIsLoading(false);
      }
    },
    [lines, enqueueSnackbar]
  );

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const interval = setInterval(() => {
      fetchLogs(true);
    }, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchLogs]);

  const parsedLines = useMemo(() => parseLogLines(logsText, filterQuery), [logsText, filterQuery]);
  const totalNonEmptyLines = useMemo(() => countNonEmptyLogLines(logsText), [logsText]);

  return {
    logsText,
    parsedLines,
    totalNonEmptyLines,
    isLoading,
    refreshLogs: () => fetchLogs(false),
  };
}
