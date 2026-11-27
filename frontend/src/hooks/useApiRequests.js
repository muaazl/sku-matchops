import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getApiRequest, getApiRequests } from '../services/requestLogApiService';
import { safeJsonParse } from '../utils/formatters';
import { useDebounce } from './useDebounce';

export function useApiRequests({ pathQuery, status, selectedId }) {
  const debouncedPathQuery = useDebounce(pathQuery, 350);
  const statusCodeParam = status === 'all' ? undefined : Number.parseInt(status, 10);

  const listQuery = useQuery({
    queryKey: ['api-requests', debouncedPathQuery, status],
    queryFn: () =>
      getApiRequests({
        path: debouncedPathQuery || undefined,
        status_code: statusCodeParam,
      }),
    refetchInterval: 15000,
    refetchIntervalInBackground: false,
  });

  const detailQuery = useQuery({
    queryKey: ['api-request', selectedId],
    queryFn: () => getApiRequest(selectedId),
    enabled: !!selectedId,
  });

  const parsedHeaders = useMemo(
    () => safeJsonParse(detailQuery.data?.headers_json, {}) || {},
    [detailQuery.data?.headers_json]
  );

  const parsedQueryParams = useMemo(
    () => safeJsonParse(detailQuery.data?.query_params_json, {}) || {},
    [detailQuery.data?.query_params_json]
  );

  return {
    debouncedPathQuery,
    requests: listQuery.data || [],
    isLoading: listQuery.isLoading,
    isFetching: listQuery.isFetching,
    refetch: listQuery.refetch,
    selectedDetail: detailQuery.data,
    isDetailLoading: detailQuery.isLoading,
    parsedHeaders,
    parsedQueryParams,
  };
}
