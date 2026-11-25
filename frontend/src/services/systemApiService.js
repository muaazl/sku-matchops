import { httpClient } from './httpClient';

export async function getLogs(lines) {
  return (await httpClient.get('/logs', { params: { lines } })).data;
}

export async function checkHealth() {
  return (await httpClient.get('/health', { timeout: 4000 })).data;
}

export async function getSystemStatus() {
  return (await httpClient.get('/system/status')).data;
}
