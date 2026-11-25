import { httpClient } from './httpClient';

export async function createBatch(formData) {
  return (
    await httpClient.post('/batches', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  ).data;
}

export async function fetchMerchantSkus(payload) {
  return (await httpClient.post('/merchant-fetch', payload)).data;
}

export async function getBatch(id) {
  return (await httpClient.get(`/batches/${id}`)).data;
}
