import { useQuery } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { retryJob } from '../services/jobApiService';
import { getProcessedSkus } from '../services/skuApiService';
import { download, toCsv } from '../utils/csvUtils';
import { buildAuditExportRows, buildCleanExportRows, sanitizeExportFilename } from '../utils/skuExportUtils';
import { useDebounce } from './useDebounce';

export function useSkuResults({
  selectedJob,
  page,
  rowsPerPage,
  skuSearch,
  skuBtFilter,
  skuGkFilter,
  skuSourceFilter,
}) {
  const { enqueueSnackbar } = useSnackbar();

  const debouncedSkuSearch = useDebounce(skuSearch, 300);
  const debouncedSkuBtFilter = useDebounce(skuBtFilter, 300);
  const debouncedSkuGkFilter = useDebounce(skuGkFilter, 300);

  const query = useQuery({
    queryKey: [
      'processed_skus',
      selectedJob?.id,
      page,
      rowsPerPage,
      debouncedSkuSearch,
      debouncedSkuBtFilter,
      debouncedSkuGkFilter,
      skuSourceFilter,
    ],
    queryFn: () =>
      getProcessedSkus({
        batch_id: selectedJob?.batch_id || selectedJob?.id,
        page: page + 1,
        limit: rowsPerPage,
        ...(debouncedSkuSearch.trim() ? { sku_name: debouncedSkuSearch.trim() } : {}),
        ...(debouncedSkuBtFilter.trim() ? { bt: debouncedSkuBtFilter.trim() } : {}),
        ...(debouncedSkuGkFilter.trim() ? { gk: debouncedSkuGkFilter.trim() } : {}),
        ...(skuSourceFilter !== 'all' ? { match_source: skuSourceFilter } : {}),
      }),
    enabled: !!selectedJob,
  });

  const exportRows = async (type = 'clean') => {
    if (!selectedJob) return;
    try {
      const allSkusRes = await getProcessedSkus({
        batch_id: selectedJob?.batch_id || selectedJob?.id,
        limit: 0,
        ...(debouncedSkuSearch.trim() ? { sku_name: debouncedSkuSearch.trim() } : {}),
        ...(debouncedSkuBtFilter.trim() ? { bt: debouncedSkuBtFilter.trim() } : {}),
        ...(debouncedSkuGkFilter.trim() ? { gk: debouncedSkuGkFilter.trim() } : {}),
        ...(skuSourceFilter !== 'all' ? { match_source: skuSourceFilter } : {}),
      });

      const skusList = allSkusRes.data || [];
      if (!skusList.length) {
        enqueueSnackbar('No SKU data available to export', { variant: 'warning' });
        return;
      }

      const rawSheetName = selectedJob.target_sheet || selectedJob.sheet_name || 'Sheet';
      const safeSheetName = sanitizeExportFilename(rawSheetName);

      if (type === 'clean') {
        const cleanRows = buildCleanExportRows(skusList, selectedJob.domain);
        const filename = `Job_${selectedJob.id}_${safeSheetName}_clean.csv`;
        download(filename, toCsv(cleanRows));
        enqueueSnackbar(`Exported ${cleanRows.length} rows to ${filename} (Clean Version)`, { variant: 'success' });
      } else if (type === 'audit') {
        const auditRows = buildAuditExportRows(skusList);
        const filename = `Job_${selectedJob.id}_${safeSheetName}_audit.csv`;
        download(filename, toCsv(auditRows));
        enqueueSnackbar(`Exported ${auditRows.length} rows to ${filename} (Audit Version)`, { variant: 'success' });
      }
    } catch {
      enqueueSnackbar('Failed to fetch data for export', { variant: 'error' });
    }
  };

  const rerunIncompleteJob = async (job) => {
    const res = await retryJob(job.id);
    enqueueSnackbar(res.message || 'Job rerun successfully initiated!', { variant: 'success' });
    return res;
  };

  return {
    skus: query.data?.data || [],
    totalSkus: query.data?.total || 0,
    isLoadingSkus: query.isLoading,
    exportRows,
    rerunIncompleteJob,
  };
}
