import { httpClient } from './httpClient';

export async function getApiRequests(params) {
  return (await httpClient.get('/api-requests', { params })).data;
}

export async function getApiRequest(id) {
  return (await httpClient.get(`/api-requests/${id}`)).data;
}
