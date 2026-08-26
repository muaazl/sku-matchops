-- 1. Rules Engine Tables
CREATE TABLE IF NOT EXISTS rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id         TEXT UNIQUE NOT NULL,       -- e.g. "food_bt_001"
    domain          TEXT NOT NULL,              -- "food", "market", "shared"
    module          TEXT NOT NULL,              -- "bt_override", "gk_injection", "formatter", "visibility"
    priority        INTEGER NOT NULL DEFAULT 100, -- lower number runs first within module
    description     TEXT NOT NULL,              -- human-readable rule name
    reasoning       TEXT NOT NULL,              -- why this rule exists (goes into audit trail)
    condition_logic TEXT NOT NULL DEFAULT 'AND', -- top-level logic between condition groups: "AND" | "OR"
    is_active       INTEGER NOT NULL DEFAULT 1, -- 1 = active, 0 = disabled
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conditions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id          TEXT NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
    condition_group  INTEGER NOT NULL DEFAULT 1,  -- group number, starts at 1
    condition_type   TEXT NOT NULL,               -- e.g. "sku_contains"
    value            TEXT NOT NULL,               -- the value to check against
    negate           INTEGER NOT NULL DEFAULT 0   -- 0 = normal, 1 = NOT
);

CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id     TEXT NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,    -- e.g. "set_bt"
    value       TEXT NOT NULL     -- the value to apply
);

-- 2. Jobs
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  batch_id TEXT,
  type TEXT, -- 'batch_csv','merchant_fetch','retry'
  status TEXT, -- 'queued','running','completed','failed','cancelled'
  current_stage TEXT, -- 'queued','embedding','vector_search','reranking','classifying','applying_rules','writing_results','done'
  total_items INTEGER DEFAULT 0,
  completed_items INTEGER DEFAULT 0,
  cancel_requested INTEGER DEFAULT 0,
  retry_of_job_id TEXT NULL,
  error_message TEXT,
  created_by TEXT,
  started_at TEXT,
  updated_at TEXT,
  completed_at TEXT,
  domain TEXT,
  sheet_name TEXT,
  target_sheet TEXT,
  duration_minutes REAL,
  high_conf INTEGER DEFAULT 0,
  med_conf INTEGER DEFAULT 0,
  low_conf INTEGER DEFAULT 0,
  match_rate REAL DEFAULT 0.0,
  input_skus_json TEXT
);

-- 3. Batches
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY,
  source TEXT, -- 'upload','merchant'
  merchant_id TEXT,
  filename TEXT,
  domain TEXT, -- 'food','market'
  status TEXT,
  total_skus INTEGER,
  created_by TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  completed_at TEXT
);

-- 4. Per-SKU audit trail
CREATE TABLE IF NOT EXISTS processed_skus (
  id TEXT PRIMARY KEY,
  batch_id TEXT,
  sku_name TEXT,
  domain TEXT,
  bt TEXT,
  gk_json TEXT,
  region TEXT,
  category TEXT,
  flavor_extraction TEXT,
  brand_extraction TEXT,
  confidence REAL,
  match_source TEXT,
  rules_applied_json TEXT,
  logic_notes TEXT,
  matched_catalog_name TEXT,
  match_score REAL,
  bt_confidence REAL,
  gk_confidence REAL,
  region_confidence REAL,
  created_at TEXT DEFAULT (datetime('now'))
);

-- 5. Request log
CREATE TABLE IF NOT EXISTS api_requests (
  id TEXT PRIMARY KEY,
  method TEXT,
  path TEXT,
  payload_json_redacted TEXT,
  response_json TEXT,
  status_code INTEGER,
  duration_ms INTEGER,
  ip_address TEXT,
  headers_json TEXT,
  query_params_json TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

-- 6. Catalog items reference table
CREATE TABLE IF NOT EXISTS catalog_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  domain TEXT NOT NULL,
  name TEXT NOT NULL,
  brand TEXT,
  flavor TEXT,
  basictype TEXT,
  generic_keywords TEXT,
  price REAL,
  category TEXT,
  region TEXT,
  description TEXT,
  clean_text TEXT,
  weight_val REAL,
  entities_json TEXT,
  token_count INTEGER,
  clean_no_weights TEXT,
  row_hash TEXT NOT NULL
);

-- 7. Brand/Flavor list table
CREATE TABLE IF NOT EXISTS brand_flavors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  domain TEXT NOT NULL,
  name TEXT NOT NULL,
  aliases TEXT,
  is_weak INTEGER DEFAULT 0,
  is_meat INTEGER DEFAULT 0,
  is_vegetable INTEGER DEFAULT 0,
  is_seafood INTEGER DEFAULT 0,
  row_hash TEXT NOT NULL
);

-- 8. Classifier Keyword Dictionaries table
CREATE TABLE IF NOT EXISTS classifier_dictionaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  domain TEXT NOT NULL,
  tag_type TEXT NOT NULL,
  tag TEXT NOT NULL
);

-- 9. BT-GK mapping rules table
CREATE TABLE IF NOT EXISTS bt_gk_map (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  domain TEXT NOT NULL,
  basictype TEXT NOT NULL,
  generic_keywords TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_conditions_rule_id ON conditions(rule_id);
CREATE INDEX IF NOT EXISTS idx_actions_rule_id ON actions(rule_id);
CREATE INDEX IF NOT EXISTS idx_rules_domain_module ON rules(domain, module, priority);
CREATE INDEX IF NOT EXISTS idx_jobs_started_at ON jobs(started_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status_started ON jobs(status, started_at);
CREATE INDEX IF NOT EXISTS idx_api_requests_created_at ON api_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_catalog_items_domain ON catalog_items(domain);
CREATE INDEX IF NOT EXISTS idx_catalog_items_domain_hash ON catalog_items(domain, row_hash);
CREATE INDEX IF NOT EXISTS idx_brand_flavors_domain ON brand_flavors(domain);
CREATE INDEX IF NOT EXISTS idx_classifier_dict_domain ON classifier_dictionaries(domain, tag_type);
CREATE INDEX IF NOT EXISTS idx_bt_gk_map_domain ON bt_gk_map(domain, basictype);
CREATE INDEX IF NOT EXISTS idx_processed_skus_created_at ON processed_skus(created_at);
CREATE INDEX IF NOT EXISTS idx_processed_skus_batch_id ON processed_skus(batch_id);
CREATE INDEX IF NOT EXISTS idx_processed_skus_domain_created ON processed_skus(domain, created_at);

