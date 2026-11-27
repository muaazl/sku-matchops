import { httpClient } from './httpClient';

export async function searchCatalog(params) {
  return (await httpClient.get('/catalog', { params })).data;
}

export async function getCatalogSummaryStats(params) {
  return (await httpClient.get('/catalog/summary-stats', { params })).data;
}

export async function refreshCatalog() {
  return (await httpClient.post('/catalog/refresh')).data;
}

export async function buildCatalogCache() {
  return (await httpClient.post('/catalog/build-cache')).data;
}

export async function checkCatalogSync(params) {
  return (await httpClient.get('/catalog/check-sync', { params })).data;
}
