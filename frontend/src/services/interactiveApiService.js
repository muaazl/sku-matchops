import { httpClient } from './httpClient';

export async function runInteractiveSingle(payload) {
  return (await httpClient.post('/interactive/run-single', payload)).data;
}

export async function rerunInteractiveRules(payload) {
  return (await httpClient.post('/interactive/rerun-rules', payload)).data;
}

export async function getTemplateSuggestions(payload) {
  return (await httpClient.post('/interactive/suggest', payload)).data;
}

export async function runInteractiveAudit(payload) {
  return (await httpClient.post('/interactive/audit', payload)).data;
}
