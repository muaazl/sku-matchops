import { httpClient } from './httpClient';

export async function getRules(params) {
  return (await httpClient.get('/rules', { params })).data;
}

export async function createRule(rule) {
  return (await httpClient.post('/rules', rule)).data;
}

export async function updateRule(rule) {
  return (await httpClient.put(`/rules/${rule.rule_id}`, rule)).data;
}

export async function saveRule(rule) {
  if (rule.rule_id.startsWith('new_rule_')) {
    return createRule(rule);
  }
  return updateRule(rule);
}

export async function deleteRule(id) {
  return (await httpClient.delete(`/rules/${id}`)).data;
}

export async function reorderRules(payload) {
  return (await httpClient.put('/rules/reorder', payload)).data;
}

export async function testRule({ rule, sampleRecord }) {
  return (await httpClient.post('/rules/test-draft', { rule, sample_record: sampleRecord })).data;
}
