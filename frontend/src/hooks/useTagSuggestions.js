import { useEffect, useMemo, useState } from 'react';
import { searchCatalog } from '../services/catalogApiService';
import { resolveTagSuggestionTarget } from '../utils/catalogUtils';
import { useDebounce } from './useDebounce';

export function useTagSuggestions({ type, ruleDomain, inputValue }) {
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);

  const { dataset, domains } = useMemo(() => resolveTagSuggestionTarget(type, ruleDomain), [type, ruleDomain]);
  const debouncedInput = useDebounce(inputValue, 400);

  useEffect(() => {
    if (!dataset || domains.length === 0) {
      setOptions([]);
      return undefined;
    }

    let active = true;
    const fetchSuggestions = async () => {
      setLoading(true);
      try {
        const responses = await Promise.all(
          domains.map((dom) =>
            searchCatalog({
              dataset,
              domain: dom,
              query: debouncedInput,
              page_size: 50,
            })
          )
        );
        if (!active) return;

        const mergedResults = [];
        const seenNames = new Set();
        responses.forEach((res) => {
          if (res?.results) {
            res.results.forEach((item) => {
              const nameLower = item.name.toLowerCase().trim();
              if (!seenNames.has(nameLower)) {
                seenNames.add(nameLower);
                mergedResults.push(item.name);
              }
            });
          }
        });
        mergedResults.sort((a, b) => a.localeCompare(b));
        setOptions(mergedResults);
      } catch (err) {
        console.error('Failed to fetch autocomplete suggestions:', err);
      } finally {
        if (active) setLoading(false);
      }
    };

    void fetchSuggestions();
    return () => {
      active = false;
    };
  }, [debouncedInput, dataset, domains]);

  return {
    dataset,
    options,
    loading,
  };
}
