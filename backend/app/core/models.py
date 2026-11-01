"""
Re-export SQLAlchemy Table models and metadata from backend.app.db.schema for backwards compatibility.
"""

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

__all__ = [
    "actions",
    "api_requests",
    "batches",
    "brand_flavors",
    "bt_gk_map",
    "catalog_items",
    "classifier_dictionaries",
    "conditions",
    "jobs",
    "metadata",
    "processed_skus",
    "rules",
    "users",
]
