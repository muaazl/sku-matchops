import { useCallback, useEffect, useState } from 'react';
import { checkHealth } from '../services/systemApiService';

export function useHealthCheck() {
  const [isOffline, setIsOffline] = useState(false);
  const [checking, setChecking] = useState(false);

  const runHealthCheck = useCallback(async () => {
    setChecking(true);
    try {
      await checkHealth();
      setIsOffline(false);
    } catch {
      setIsOffline(true);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    runHealthCheck();
  }, [runHealthCheck]);

  return { isOffline, checking, runHealthCheck };
}
