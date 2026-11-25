import { useState } from 'react';
import { useSnackbar } from 'notistack';
import { getTemplateSuggestions, runInteractiveAudit, runInteractiveSingle } from '../services/interactiveApiService';
import { DOMAINS } from '../utils/constants';

export function useSkuMatch() {
  const { enqueueSnackbar } = useSnackbar();

  const [viewMode, setViewMode] = useState('interactive');
  const [auditTrace, setAuditTrace] = useState(null);
  const [auditLoading, setAuditLoading] = useState(false);

  const [sku, setSku] = useState('Red Bull Energy Drink 250ml');
  const [price, setPrice] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [task, setTask] = useState('pipeline');
  const [domain, setDomain] = useState(DOMAINS[0]);

  const [result, setResult] = useState(null);
  const [resultTask, setResultTask] = useState('pipeline');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState(null);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [enableSuggestions, setEnableSuggestions] = useState(true);

  const runSingleMatch = async () => {
    if (!sku) {
      enqueueSnackbar('Please enter a SKU name', { variant: 'warning' });
      return;
    }
    setLoading(true);
    setResult(null);
    setSuggestions(null);
    try {
      const payload = {
        sku_name: sku,
        domain,
        task,
        price: price ? parseFloat(price) : 0.0,
        description,
        category,
      };
      const res = await runInteractiveSingle(payload);
      setResult(res);
      setResultTask(task);
      const firstResult = res.results?.[0] || {};
      enqueueSnackbar('Execution completed successfully', { variant: 'success' });

      if (enableSuggestions) {
        setSuggestionsLoading(true);
        getTemplateSuggestions({
          sku_name: sku,
          domain,
          exclude_bt: firstResult.suggested_bt || '',
          exclude_gk: firstResult.suggested_gk || '',
        })
          .then((suggRes) => setSuggestions(suggRes))
          .catch((err) => console.error('Failed to fetch template suggestions:', err))
          .finally(() => setSuggestionsLoading(false));
      }
    } catch (err) {
      console.error(err);
      enqueueSnackbar(`Execution failed: ${err.response?.data?.detail || err.message}`, { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const runAudit = async () => {
    if (!sku) {
      enqueueSnackbar('Please enter a SKU name', { variant: 'warning' });
      return;
    }
    setAuditLoading(true);
    try {
      const auditTaskToRun = resultTask || task;
      const payload = {
        sku_name: sku,
        domain: result?.domain || domain,
        task: auditTaskToRun,
        price: price ? parseFloat(price) : 0.0,
        description,
        category,
      };
      const auditRes = await runInteractiveAudit(payload);
      setAuditTrace(auditRes);
      setViewMode('audit');
      enqueueSnackbar('Audit telemetry loaded successfully', { variant: 'success' });
    } catch (err) {
      console.error('Audit fetch error:', err);
      enqueueSnackbar(`Failed to load audit: ${err.response?.data?.detail || err.message}`, { variant: 'error' });
    } finally {
      setAuditLoading(false);
    }
  };

  return {
    viewMode,
    setViewMode,
    auditTrace,
    auditLoading,
    sku,
    setSku,
    price,
    setPrice,
    description,
    setDescription,
    category,
    setCategory,
    task,
    setTask,
    domain,
    setDomain,
    result,
    resultTask,
    loading,
    suggestions,
    suggestionsLoading,
    enableSuggestions,
    setEnableSuggestions,
    runSingleMatch,
    runAudit,
  };
}
