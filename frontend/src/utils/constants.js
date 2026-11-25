export const DOMAINS = ['food', 'market'];

export const TASK_TYPES = ['matcher', 'classifier', 'pipeline'];

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

export const JOB_STATUS_FILTERS = ['all', 'running', 'queued', 'completed', 'failed', 'cancelled'];

export const COLLECTION_DOMAINS = {
  market_catalog: 'market',
  food_catalog: 'food',
};

export const COLLECTION_STATUS_COLORS = {
  green: 'success.main',
  yellow: 'warning.main',
  red: 'error.main',
};

export const DICT_TYPE_LABELS = {
  gk: 'Generic Keywords',
  bt: 'Basic Type',
  region: 'Region',
  category: 'Category',
};

export const REQUEST_STATUS_OPTIONS = ['all', '200', '400', '401', '500'];

export const HTTP_METHOD_COLORS = {
  GET: 'info',
  POST: 'success',
  PUT: 'warning',
  DELETE: 'error',
};

export const STATUS_COLOR_MAP = {
  completed: 'success',
  done: 'success',
  running: 'info',
  queued: 'default',
  pending: 'warning',
  failed: 'error',
  cancelled: 'default',
  approved: 'success',
  rejected: 'error',
  matched: 'success',
  review: 'warning',
  confident: 'success',
};

const ROUTE_TITLES = {
  '/dashboard': 'Dashboard',
  '/guide': 'Guide',
  '/jobs': 'Jobs',
  '/requests': 'Requests',
  '/collections': 'Collections',
  '/process-skus': 'Process SKUs',
  '/interactive': 'Interactive',
  '/sku-results': 'SKU Results',
  '/rules': 'Rules Engine',
  '/catalog': 'Catalog Search',
  '/logs': 'Logs',
};

export function getPageName(pathname) {
  if (pathname && pathname.startsWith('/jobs/')) return 'Job Detail';
  return ROUTE_TITLES[pathname] || 'SKU MatchOps';
}
