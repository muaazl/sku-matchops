import { httpClient } from './httpClient';

export async function getModelsStatus() {
  return (await httpClient.get('/models-status')).data;
}

export async function loadModels() {
  return (await httpClient.post('/load-models')).data;
}
