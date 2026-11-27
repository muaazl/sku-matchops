"""SQLAlchemy ORM models and shared metadata for PostgreSQL schema registration."""

from backend.models.audit_model import ApiRequest, api_requests
from backend.models.base_model import Base, metadata
from backend.models.batch_model import Batch, batches
from backend.models.catalog_model import (
    BrandFlavor,
    BtGkMap,
    CatalogItem,
    ClassifierDictionary,
    brand_flavors,
    bt_gk_map,
    catalog_items,
    classifier_dictionaries,
)
from backend.models.job_model import Job, jobs
from backend.models.rule_model import Rule, RuleAction, RuleCondition, actions, conditions, rules
from backend.models.sku_model import ProcessedSku, processed_skus

__all__ = [
    "ApiRequest",
    "Base",
    "Batch",
    "BrandFlavor",
    "BtGkMap",
    "CatalogItem",
    "ClassifierDictionary",
    "Job",
    "ProcessedSku",
    "Rule",
    "RuleAction",
    "RuleCondition",
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
]
