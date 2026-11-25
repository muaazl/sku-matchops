import { getActiveApiKey, resolveBaseUrl } from './httpClient';

export function subscribeSseStream(path, { onMessage, onError } = {}) {
  const controller = new AbortController();
  (async () => {
    try {
      const base = await resolveBaseUrl();
      const headers = {
        Accept: 'text/event-stream',
        'X-API-Key': getActiveApiKey(),
      };
      const resp = await fetch(`${base}${path}`, { headers, signal: controller.signal });
      if (!resp.ok || !resp.body) {
        throw new Error(`SSE stream failed (${resp.status})`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
        let sep;
        while ((sep = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          for (const line of frame.split('\n')) {
            if (line.startsWith('data: ')) {
              onMessage?.(JSON.parse(line.slice(6)));
            }
          }
        }
      }
    } catch (err) {
      if (err?.name !== 'AbortError') {
        onError?.(err);
      }
    }
  })();
  return () => controller.abort();
}
