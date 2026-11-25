import { useState, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { getCollection, getCollections, searchCollection } from '../services/vectorDbApiService';
import { runInteractiveAudit } from '../services/interactiveApiService';
import { COLLECTION_DOMAINS } from '../utils/constants';
import { transformAuditToVectorResults } from '../utils/auditUtils';

async function fetchEnrichedCollections() {
  const data = await getCollections();
  const preloadedDetails = data.details && typeof data.details === 'object' ? data.details : null;

  return Promise.all(
    data.collections.map(async (name) => {
      try {
        const info = preloadedDetails?.[name] ?? (await getCollection(name));
        const size = info.config?.params?.vectors?.dense?.size || info.config?.params?.vectors?.size || 1024;
        const rawDist =
          info.config?.params?.vectors?.dense?.distance || info.config?.params?.vectors?.distance || 'Cosine';
        const dist = typeof rawDist === 'string' ? rawDist : String(rawDist);
        return {
          name,
          status: info.status || 'unknown',
          vectors_count: info.indexed_vectors_count ?? info.vectors_count ?? 0,
          points_count: info.points_count ?? 0,
          segments_count: info.segments_count ?? 0,
          vector_size: size,
          distance: dist,
          disk_bytes: info.optimizer_status?.disk_bytes || 0,
          ram_bytes: info.optimizer_status?.ram_bytes || 0,
          shards_count: info.config?.params?.shard_number || 1,
          vectors_config: `dense, ${size}, ${dist}`,
        };
      } catch {
        return { name, status: 'error', vectors_count: 0, points_count: 0, segments_count: 0 };
      }
    })
  );
}

export function useVectorCollections() {
  const { enqueueSnackbar } = useSnackbar();
  const [collection, setCollection] = useState('');
  const [query, setQuery] = useState('');
  const [searchMode, setSearchMode] = useState('dense');
  const [results, setResults] = useState(null);

  const matcherDomain = COLLECTION_DOMAINS[collection] || null;

  useEffect(() => {
    if (!matcherDomain && searchMode === 'pipeline') {
      setSearchMode('dense');
    }
  }, [matcherDomain, searchMode]);

  const { data: collections = [], isLoading } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchEnrichedCollections,
  });

  useEffect(() => {
    if (collections.length > 0 && !collection) {
      setCollection(collections[0].name);
    }
  }, [collections, collection]);

  useEffect(() => {
    setResults(null);
  }, [collection, searchMode]);

  const searchMutation = useMutation({
    mutationFn: searchCollection,
    onSuccess: (data) => setResults(data.results),
    onError: (err) => {
      enqueueSnackbar(`Search failed: ${err.message}`, { variant: 'error' });
    },
  });

  const auditMutation = useMutation({
    mutationFn: runInteractiveAudit,
    onSuccess: (data) => setResults(transformAuditToVectorResults(data)),
    onError: (err) => {
      enqueueSnackbar(`Matcher pipeline search failed: ${err.message}`, { variant: 'error' });
    },
  });

  const runSearch = () => {
    if (!query.trim() || !collection) {
      setResults([]);
      return;
    }
    if (searchMode === 'pipeline') {
      if (matcherDomain) {
        auditMutation.mutate({ sku_name: query.trim(), domain: matcherDomain, task: 'matcher' });
      } else {
        enqueueSnackbar('Pipeline search is only available for catalog collections.', { variant: 'warning' });
      }
      return;
    }
    searchMutation.mutate({
      name: collection,
      query: query.trim(),
      topK: 10,
      mode: searchMode,
      scoreThreshold: searchMode === 'dense' ? 0.1 : undefined,
    });
  };

  return {
    collections,
    isLoading,
    collection,
    setCollection,
    query,
    setQuery,
    searchMode,
    setSearchMode,
    matcherDomain,
    results,
    isSearching: searchMutation.isPending || auditMutation.isPending,
    runSearch,
  };
}
