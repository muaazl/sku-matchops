"""
Database package for SKU MatchOps Backend.
Encapsulates database connection management, session pooling, and table schemas.
"""

from backend.app.db.init import init_db
from backend.app.db.schema import (
    actions,
    api_requests,
    batches,
    brand_flavors,
    bt_gk_map,
    catalog_items,
    classifier_dictionaries,
    conditions,
    jobs,
    metadata,
    processed_skus,
    rules,
    users,
)
from backend.app.db.session import (
    dispose_engine,
    get_connection,
    get_db_connection,
    get_engine,
    get_next_job_id,
    log_outbound_request,
    upsert_processed_skus,
)

__all__ = [
    "actions",
    "api_requests",
    "batches",
    "brand_flavors",
    "bt_gk_map",
    "catalog_items",
    "classifier_dictionaries",
    "conditions",
    "dispose_engine",
    "get_connection",
    "get_db_connection",
    "get_engine",
    "get_next_job_id",
    "init_db",
    "jobs",
    "log_outbound_request",
    "metadata",
    "processed_skus",
    "rules",
    "upsert_processed_skus",
    "users",
]
