import os
import sqlite3
import logging
from engine import config

logger = logging.getLogger("matchops.db")

SCHEMA_SQL = """
-- 1. Rules Engine Tables
CREATE TABLE IF NOT EXISTS rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id         TEXT UNIQUE NOT NULL,
    domain          TEXT NOT NULL,
    module          TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 100,
    description     TEXT NOT NULL,
    reasoning       TEXT NOT NULL,
    condition_logic TEXT NOT NULL DEFAULT 'AND',
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conditions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id          TEXT NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
    condition_group  INTEGER NOT NULL DEFAULT 1,
    condition_type   TEXT NOT NULL,
    value            TEXT NOT NULL,
    negate           INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id     TEXT NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    value       TEXT NOT NULL
);

-- 2. Job Queue & Batches
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    batch_id TEXT,
    type TEXT,
    status TEXT,
    current_stage TEXT,
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

CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    source TEXT,
    merchant_id TEXT,
    filename TEXT,
    domain TEXT,
    status TEXT,
    total_skus INTEGER,
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

-- 3. Per-SKU Audit Trail
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

-- 4. Request Audit Log
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

-- 5. Catalog Items Reference
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

-- 6. Brand / Flavor Reference
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

-- 7. Classifier Dictionaries
CREATE TABLE IF NOT EXISTS classifier_dictionaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    tag_type TEXT NOT NULL,
    tag TEXT NOT NULL
);

-- 8. BT to GK Map
CREATE TABLE IF NOT EXISTS bt_gk_map (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    basictype TEXT NOT NULL,
    generic_keywords TEXT NOT NULL
);

-- 9. Indexes for Performance
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
"""

def ensure_db_initialized(conn_or_path=None) -> sqlite3.Connection:
    """
    Ensures that the SQLite database directory exists, WAL mode is active,
    and all required tables and indexes are created.
    """
    is_provided_conn = hasattr(conn_or_path, "cursor")
    
    if is_provided_conn:
        conn = conn_or_path
    else:
        db_path = conn_or_path if isinstance(conn_or_path, str) else config.DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)

    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA_SQL)
        
        # Apply any column migrations if necessary
        cursor = conn.cursor()
        
        # api_requests columns
        cursor.execute("PRAGMA table_info(api_requests);")
        api_cols = [row[1] for row in cursor.fetchall()]
        for col_name in ("ip_address", "headers_json", "query_params_json"):
            if col_name not in api_cols:
                conn.execute(f"ALTER TABLE api_requests ADD COLUMN {col_name} TEXT;")
                
        # jobs columns
        cursor.execute("PRAGMA table_info(jobs);")
        jobs_cols = [row[1] for row in cursor.fetchall()]
        new_cols_jobs = {
            "domain": "TEXT",
            "sheet_name": "TEXT",
            "target_sheet": "TEXT",
            "duration_minutes": "REAL",
            "high_conf": "INTEGER DEFAULT 0",
            "med_conf": "INTEGER DEFAULT 0",
            "low_conf": "INTEGER DEFAULT 0",
            "match_rate": "REAL DEFAULT 0.0",
            "input_skus_json": "TEXT"
        }
        for col_name, col_type in new_cols_jobs.items():
            if col_name not in jobs_cols:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type};")

        # processed_skus columns
        cursor.execute("PRAGMA table_info(processed_skus);")
        sku_cols = [row[1] for row in cursor.fetchall()]
        new_cols_skus = {
            "logic_notes": "TEXT",
            "matched_catalog_name": "TEXT",
            "match_score": "REAL",
            "bt_confidence": "REAL",
            "gk_confidence": "REAL",
            "region_confidence": "REAL"
        }
        for col_name, col_type in new_cols_skus.items():
            if col_name not in sku_cols:
                conn.execute(f"ALTER TABLE processed_skus ADD COLUMN {col_name} {col_type};")

        conn.commit()
    except Exception as e:
        logger.error(f"[DB] Error ensuring database initialization: {e}")
        if not is_provided_conn:
            conn.close()
        raise e

    return conn

def init_db(db_path: str = None) -> None:
    """Public helper to initialize the SQLite database."""
    conn = ensure_db_initialized(db_path)
    conn.close()
