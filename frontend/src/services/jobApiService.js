import { httpClient } from './httpClient';
import { subscribeSseStream } from './sseClient';

export async function getJobs(params) {
  const resp = await httpClient.get('/jobs', { params });
  return {
    data: resp.data,
    total: Number.parseInt(resp.headers['x-total-count'] || resp.data.length, 10),
  };
}

export async function getJob(id) {
  return (await httpClient.get(`/jobs/${id}`)).data;
}

export async function getDashboardStats(params) {
  return (await httpClient.get('/jobs/dashboard-stats', { params })).data;
}

export async function getTagStats(params) {
  return (await httpClient.get('/jobs/tag-stats', { params })).data;
}

export async function cancelJob(id) {
  return (await httpClient.post(`/jobs/${id}/cancel`)).data;
}

export async function retryJob(id) {
  return (await httpClient.post(`/jobs/${id}/retry`)).data;
}

export function subscribeJobsStream(callbacks) {
  return subscribeSseStream('/jobs/stream', callbacks);
}

export function subscribeJobDetailStream(jobId, callbacks) {
  return subscribeSseStream(`/jobs/${jobId}/stream`, callbacks);
}
