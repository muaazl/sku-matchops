"""
Shared PostgreSQL schema for the Gateway and the Engine.

Alembic (migrations/) is the source of truth for the live schema; these Table
definitions mirror it and back the bulk-upsert statements in engine.core.db.
"""

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
)

metadata = MetaData()


def _ts(name: str, default: bool = False) -> Column:
    return Column(name, DateTime(timezone=True), server_default=func.now() if default else None)


# 1. Rules Engine
rules = Table(
    "rules",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("rule_id", Text, nullable=False, unique=True),
    Column("domain", Text, nullable=False),
    Column("priority", Integer, nullable=False, server_default="100"),
    Column("description", Text, nullable=False),
    Column("reasoning", Text, nullable=False),
    Column("condition_logic", Text, nullable=False, server_default="AND"),
    Column("is_active", Integer, nullable=False, server_default="1"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("idx_rules_domain_priority", "domain", "priority"),
)

conditions = Table(
    "conditions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column(
        "rule_id", Text, ForeignKey("rules.rule_id", ondelete="CASCADE"), nullable=False, index=True
    ),
    Column("condition_group", Integer, nullable=False, server_default="1"),
    Column("condition_type", Text, nullable=False),
    Column("value", Text, nullable=False),
    Column("negate", Integer, nullable=False, server_default="0"),
)

actions = Table(
    "actions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column(
        "rule_id", Text, ForeignKey("rules.rule_id", ondelete="CASCADE"), nullable=False, index=True
    ),
    Column("action_type", Text, nullable=False),
    Column("value", Text, nullable=False),
)

# 2. Job Queue & Batches
jobs = Table(
    "jobs",
    metadata,
    Column("id", Text, primary_key=True),
    Column("batch_id", Text),
    Column("type", Text),
    Column("status", Text),
    Column("current_stage", Text),
    Column("total_items", Integer, server_default="0"),
    Column("completed_items", Integer, server_default="0"),
    Column("cancel_requested", Integer, server_default="0"),
    Column("retry_of_job_id", Text),
    Column("error_message", Text),
    Column("created_by", Text),
    _ts("started_at"),
    _ts("updated_at"),
    _ts("completed_at"),
    Column("domain", Text),
    Column("sheet_name", Text),
    Column("target_sheet", Text),
    Column("duration_minutes", Float),
    Column("high_conf", Integer, server_default="0"),
    Column("med_conf", Integer, server_default="0"),
    Column("low_conf", Integer, server_default="0"),
    Column("match_rate", Float, server_default="0"),
    Column("input_skus_json", Text),
    Index("idx_jobs_started_at", "started_at"),
    Index("idx_jobs_status_started", "status", "started_at"),
)

batches = Table(
    "batches",
    metadata,
    Column("id", Text, primary_key=True),
    Column("source", Text),
    Column("merchant_id", Text),
    Column("filename", Text),
    Column("domain", Text),
    Column("status", Text),
    Column("total_skus", Integer),
    Column("created_by", Text),
    _ts("created_at", default=True),
    _ts("completed_at"),
)

# 3. Per-SKU Audit Trail (match results). batch_id holds the job id.
processed_skus = Table(
    "processed_skus",
    metadata,
    Column("id", Text, primary_key=True),
    Column("batch_id", Text),
    Column("sku_index", Integer),
    Column("rule_version", Text),
    Column("sku_name", Text),
    Column("domain", Text),
    Column("bt", Text),
    Column("gk_json", Text),
    Column("region", Text),
    Column("category", Text),
    Column("flavor_extraction", Text),
    Column("brand_extraction", Text),
    Column("confidence", Float),
    Column("match_source", Text),
    Column("rules_applied_json", Text),
    Column("logic_notes", Text),
    Column("matched_catalog_name", Text),
    Column("match_score", Float),
    Column("bt_confidence", Float),
    Column("gk_confidence", Float),
    Column("region_confidence", Float),
    Column("input_price", Float),
    Column("input_description", Text),
    Column("input_category", Text),
    Column("upload_job_id", Text),
    Column("upload_status", Text),
    _ts("created_at", default=True),
    # Idempotency key: re-delivering the same job batch updates rows instead of duplicating them
    UniqueConstraint(
        "batch_id", "sku_index", "rule_version", name="uq_processed_skus_job_sku_rule"
    ),
    Index("idx_processed_skus_created_at", "created_at"),
    Index("idx_processed_skus_batch_id", "batch_id"),
    Index("idx_processed_skus_domain_created", "domain", "created_at"),
)

# 4. Request Audit Log
api_requests = Table(
    "api_requests",
    metadata,
    Column("id", Text, primary_key=True),
    Column("method", Text),
    Column("path", Text),
    Column("payload_json_redacted", Text),
    Column("response_json", Text),
    Column("status_code", Integer),
    Column("duration_ms", Integer),
    Column("ip_address", Text),
    Column("headers_json", Text),
    Column("query_params_json", Text),
    _ts("created_at", default=True),
    Index("idx_api_requests_created_at", "created_at"),
)

# 5. Catalog Items Reference
catalog_items = Table(
    "catalog_items",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("domain", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("brand", Text),
    Column("flavor", Text),
    Column("basictype", Text),
    Column("generic_keywords", Text),
    Column("price", Float),
    Column("category", Text),
    Column("region", Text),
    Column("description", Text),
    Column("clean_text", Text),
    # JSON text: the (value, unit, type) weight tuple, or a bare number expanded on read
    Column("weight_val", Text),
    Column("entities_json", Text),
    Column("token_count", Integer),
    Column("clean_no_weights", Text),
    Column("row_hash", Text, nullable=False),
    Index("idx_catalog_items_domain_hash", "domain", "row_hash"),
)

# 6. Brand / Flavor Reference
brand_flavors = Table(
    "brand_flavors",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("domain", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("aliases", Text),
    Column("is_weak", Integer, server_default="0"),
    Column("is_meat", Integer, server_default="0"),
    Column("is_vegetable", Integer, server_default="0"),
    Column("is_seafood", Integer, server_default="0"),
    Column("catalog_count", Integer, server_default="0"),
    Column("row_hash", Text, nullable=False),
    Index("idx_brand_flavors_count", "domain", "catalog_count"),
)

# 7. Classifier Dictionaries
classifier_dictionaries = Table(
    "classifier_dictionaries",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("domain", Text, nullable=False),
    Column("tag_type", Text, nullable=False),
    Column("tag", Text, nullable=False),
    Column("catalog_count", Integer, server_default="0"),
    Column("metadata_json", Text),
    Index("idx_classifier_dict_lookup", "domain", "tag_type", "catalog_count"),
)

# 8. BT to GK Map
bt_gk_map = Table(
    "bt_gk_map",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("domain", Text, nullable=False),
    Column("basictype", Text, nullable=False),
    Column("generic_keywords", Text, nullable=False),
    Column("gk_count", Integer, server_default="0"),
    Column("catalog_count", Integer, server_default="0"),
    Index("idx_bt_gk_map_domain", "domain", "basictype"),
    Index("idx_bt_gk_map_count", "domain", "catalog_count"),
)
