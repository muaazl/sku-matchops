export const DOMAINS = ['food', 'market'];

export const RULE_MODULES = ['bt_override', 'gk_injection', 'formatter', 'visibility'];

export const CONDITION_TYPES = [
  'sku_contains',
  'bt_is',
  'gk_contains',
  'category_contains',
  'region_is',
  'price_below',
  'price_above',
  'flavor_contains',
  'flavor_is',
];

export const ACTION_TYPES = [
  'set_bt',
  'add_gk',
  'remove_gk',
  'set_region',
  'set_category',
  'set_visibility',
  'normalize_sku',
];

export const JOB_STAGES = [
  'queued',
  'embedding',
  'vector_search',
  'reranking',
  'classifying',
  'applying_rules',
  'writing_results',
  'done',
];

export const JOB_STAGE_LABELS = {
  queued: 'Queued',
  embedding: 'Embedding',
  vector_search: 'Vector Search',
  reranking: 'Reranking',
  classifying: 'Classifying',
  applying_rules: 'Applying Rules',
  writing_results: 'Writing Results',
  done: 'Done',
};
