import { httpClient } from './httpClient';

export async function getCollections() {
  return (await httpClient.get('/vector-db/collections')).data;
}

export async function getCollection(name) {
  return (await httpClient.get(`/vector-db/collections/${name}`)).data;
}

export async function searchCollection({ name, query, topK = 10, scoreThreshold, filters, mode = 'dense' }) {
  return (
    await httpClient.post(`/vector-db/collections/${name}/search`, {
      query,
      top_k: topK,
      score_threshold: scoreThreshold,
      filters,
      mode,
    })
  ).data;
}
