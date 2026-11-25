import { useEffect, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { getModelsStatus, loadModels } from '../services/modelApiService';

export function useModelStatus() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const wasLoadingRef = useRef(false);

  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['models-status'],
    queryFn: getModelsStatus,
    refetchInterval: (query) => {
      const d = query.state.data;
      if (d?.loading_in_progress) return 2000;
      if (d?.loaded) return false;
      return 10000;
    },
    refetchIntervalInBackground: false,
  });

  const loadMutation = useMutation({
    mutationFn: loadModels,
    onSuccess: (res) => {
      if (res?.status === 'success' || res?.message?.includes('already fully loaded')) {
        enqueueSnackbar('Models are already fully loaded!', { variant: 'success' });
        return;
      }
      wasLoadingRef.current = true;
      enqueueSnackbar('Model loading initiated in background...', { variant: 'info' });
      queryClient.setQueryData(['models-status'], (old) => ({
        ...old,
        loaded: false,
        loading_in_progress: true,
        status: 'loading',
      }));
      setTimeout(() => {
        refetchStatus();
      }, 500);
    },
    onError: (err) => {
      enqueueSnackbar(`Failed to initiate model load: ${err.message}`, { variant: 'error' });
    },
  });

  useEffect(() => {
    if (statusData?.loaded && !statusData?.loading_in_progress) {
      if (wasLoadingRef.current) {
        enqueueSnackbar('Models loaded successfully!', { variant: 'success' });
        wasLoadingRef.current = false;
      }
    } else if (statusData?.loading_in_progress) {
      wasLoadingRef.current = true;
    }
  }, [statusData?.loaded, statusData?.loading_in_progress, enqueueSnackbar]);

  const isLoading = loadMutation.isPending || !!statusData?.loading_in_progress;
  const isLoaded = !!statusData?.loaded && !statusData?.loading_in_progress && !loadMutation.isPending;

  return {
    isLoading,
    isLoaded,
    triggerLoadModels: () => loadMutation.mutate(),
  };
}
