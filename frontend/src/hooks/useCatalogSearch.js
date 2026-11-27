import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import {
  buildCatalogCache,
  checkCatalogSync,
  getCatalogSummaryStats,
  searchCatalog,
} from '../services/catalogApiService';
import { useDebounce } from './useDebounce';

export function useCatalogSearch({ dataset, domain, filters, page, pageSize, sortModel }) {
  const { enqueueSnackbar } = useSnackbar();
  const queryClient = useQueryClient();

  const [isChecking, setIsChecking] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [diffData, setDiffData] = useState(null);
  const [diffDialogOpen, setDiffDialogOpen] = useState(false);

  const debouncedFilters = useDebounce(filters, 500);

  const summaryQuery = useQuery({
    queryKey: ['catalog-summary-stats', domain],
    queryFn: () => getCatalogSummaryStats({ domain }),
    staleTime: 5 * 60 * 1000,
  });

  const catalogQuery = useQuery({
    queryKey: ['catalog', dataset, domain, debouncedFilters, page, pageSize, sortModel],
    queryFn: () => {
      const sort = sortModel[0];
      return searchCatalog({
        dataset,
        domain,
        query: debouncedFilters.query.trim() || undefined,
        page: page + 1,
        page_size: pageSize,
        ...(sort && { sort_by: sort.field, sort_order: sort.sort }),
        ...(dataset === 'catalog' && {
          min_price: debouncedFilters.priceMin ? Number(debouncedFilters.priceMin) : undefined,
          max_price: debouncedFilters.priceMax ? Number(debouncedFilters.priceMax) : undefined,
          region: debouncedFilters.region || undefined,
          category: debouncedFilters.category || undefined,
          gk_contains: debouncedFilters.gkContains || undefined,
          brand: debouncedFilters.brandFilter || undefined,
          basictype: debouncedFilters.btFilter || undefined,
        }),
      });
    },
  });

  const checkSync = async () => {
    setIsChecking(true);
    try {
      const res = await checkCatalogSync({ limit: 50 });
      setDiffData(res);
      setDiffDialogOpen(true);
      if (res.has_changes) {
        enqueueSnackbar('Google Sheets has pending updates!', { variant: 'info' });
      } else {
        enqueueSnackbar('Your local cache is fully in-sync with Google Sheets.', { variant: 'success' });
      }
    } catch (err) {
      enqueueSnackbar(`Check failed: ${err.message || err}`, { variant: 'error' });
    } finally {
      setIsChecking(false);
    }
  };

  const syncCache = async () => {
    setIsSyncing(true);
    try {
      await buildCatalogCache();
      enqueueSnackbar('Background cache build and pre-training initiated. Updates will apply shortly.', {
        variant: 'success',
      });
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['catalog'] });
        queryClient.invalidateQueries({ queryKey: ['catalog-summary-stats'] });
      }, 5000);
    } catch (err) {
      enqueueSnackbar(`Sync failed: ${err.message || err}`, { variant: 'error' });
    } finally {
      setIsSyncing(false);
    }
  };

  return {
    summaryStats: summaryQuery.data,
    data: catalogQuery.data || { results: [], total: 0 },
    isLoading: catalogQuery.isLoading,
    isChecking,
    isSyncing,
    diffData,
    diffDialogOpen,
    setDiffDialogOpen,
    checkSync,
    syncCache,
  };
}
