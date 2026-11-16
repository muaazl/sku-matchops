export const DOMAINS = ['food', 'market'];

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

export const CONDITION_LABELS = {
  sku_contains: 'SKU Name contains',
  bt_is: 'Basic Type (BT) is',
  gk_contains: 'Generic Keyword (GK) has',
  category_contains: 'Category contains',
  region_is: 'Region is',
  price_below: 'Price is below',
  price_above: 'Price is above',
  flavor_contains: 'Flavor contains',
  flavor_is: 'Flavor type is',
};

export const ACTION_TYPES = ['set_bt', 'add_gk', 'remove_gk', 'set_region', 'set_category'];

export const ACTION_LABELS = {
  set_bt: 'Set Basic Type (BT)',
  add_gk: 'Add Generic Keyword (GK)',
  remove_gk: 'Remove Generic Keyword (GK)',
  set_region: 'Set Region',
  set_category: 'Set Category',
};

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
