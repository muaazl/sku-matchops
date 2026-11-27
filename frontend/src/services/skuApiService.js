import { httpClient } from './httpClient';

export async function getProcessedSkus(params) {
  const resp = await httpClient.get('/processed-skus', { params });
  return {
    data: resp.data,
    total: Number.parseInt(resp.headers['x-total-count'] || resp.data.length, 10),
  };
}
