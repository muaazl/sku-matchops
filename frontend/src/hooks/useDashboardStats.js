import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getDashboardStats, getTagStats } from '../services/jobApiService';
import { normalizeMatchSources } from '../utils/formatters';

const DEFAULT_STATS = {
  totalProcessedSkus: 0,
  avgConfidencePct: 0.0,
  highConfidenceCount: 0,
  highConfidencePct: 0.0,
  mediumConfidenceCount: 0,
  mediumConfidencePct: 0.0,
  lowConfidenceCount: 0,
  lowConfidencePct: 0.0,
};

export function useDashboardStats(domain, timeframe) {
  const { data: dashboardData, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard-stats', domain, timeframe],
    queryFn: () => getDashboardStats({ domain, timeframe }),
    staleTime: 15 * 1000,
  });

  const { data: tagData, isLoading: tagLoading } = useQuery({
    queryKey: ['tag-stats', domain, timeframe],
    queryFn: () => getTagStats({ domain, timeframe }),
    staleTime: 15 * 1000,
  });

  const rawSources = dashboardData?.matchSourceDistribution;
  const matchSourceDistribution = useMemo(() => normalizeMatchSources(rawSources || []), [rawSources]);

  return {
    stats: dashboardData?.stats || DEFAULT_STATS,
    confidenceDistribution: dashboardData?.confidenceDistribution || [],
    matchSourceDistribution,
    domainBreakdown: dashboardData?.domainBreakdown || {},
    volumeTrend: dashboardData?.volumeTrend || [],
    basicTypeDistribution: tagData?.basicTypeDistribution || [],
    regionDistribution: tagData?.regionDistribution || [],
    statsLoading,
    tagLoading,
  };
}
