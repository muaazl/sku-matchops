import { useCallback, useEffect, useRef, useState } from 'react';

export function useJobStream({ enabled, subscribe, onMessage, streamKey }) {
  const [sseFallback, setSseFallback] = useState(false);
  const subscribeRef = useRef(subscribe);
  const onMessageRef = useRef(onMessage);
  subscribeRef.current = subscribe;
  onMessageRef.current = onMessage;

  const resetFallback = useCallback(() => {
    setSseFallback(false);
  }, []);

  useEffect(() => {
    if (!enabled || sseFallback) return undefined;
    return subscribeRef.current({
      onMessage: (evt) => onMessageRef.current?.(evt),
      onError: () => setSseFallback(true),
    });
  }, [enabled, sseFallback, streamKey]);

  return { sseFallback, resetFallback };
}
