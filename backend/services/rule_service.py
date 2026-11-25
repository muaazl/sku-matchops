"""Business Rules orchestration, snapshot versioning, cache invalidation, and evaluation."""

import hashlib
import json
from typing import Any

from backend.core.exceptions import ResourceNotFoundError
from backend.repositories import rule_cache_repository, rule_repository
from backend.schemas.rule_schema import RuleModel
from backend.services import engine_gateway_service, rule_evaluation_service


def compute_rules_version(rules_data: list[dict[str, Any]]) -> str:
    """Computes a deterministic 16-character SHA-256 fingerprint for active rules."""
    if not rules_data:
        return "4f53cda18c2baa0c"
    fingerprint = [
        [
            r.get("rule_id"),
            r.get("domain"),
            r.get("priority"),
            r.get("condition_logic"),
            r.get("conditions", []),
            r.get("actions", []),
        ]
        for r in rules_data
    ]
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def publish_rules_snapshot(rules_data: list[dict[str, Any]]) -> tuple[str, bool]:
    """Publishes active rules and version hash to Redis cache."""
    version = compute_rules_version(rules_data)
    return rule_cache_repository.publish_rules_snapshot(rules_data, version)


def load_active_rules(db: Any = None) -> list[dict[str, Any]]:
    """Loads active business rules from PostgreSQL."""
    return rule_repository.fetch_active_rules(db=db)


def get_rule_version() -> str:
    """Returns the current active rules version hash from cache or PostgreSQL."""
    cached = rule_cache_repository.fetch_cached_rule_version()
    if cached:
        return cached
    return compute_rules_version(load_active_rules())


def sync_rules_everywhere(db: Any = None) -> tuple[str, bool]:
    """Publishes active rules to Redis and falls back to gRPC RefreshRules if Redis fails."""
    rules_data = load_active_rules(db=db)
    version, published = publish_rules_snapshot(rules_data)
    if not published:
        engine_gateway_service.refresh_engine_rules(rules_data=rules_data, version=version)
    return version, published


def _extract_rule_payload(
    rule: RuleModel,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rule_fields = {
        "domain": rule.domain,
        "priority": rule.priority,
        "description": rule.description,
        "reasoning": rule.reasoning,
        "condition_logic": rule.condition_logic,
        "is_active": rule.is_active,
    }
    conditions_data = [cond.model_dump() for cond in rule.conditions]
    actions_data = [act.model_dump() for act in rule.actions]
    return rule_fields, conditions_data, actions_data


def list_rules(domain: str | None = None) -> list[dict[str, Any]]:
    """Fetches rules (optionally scoped by domain) with conditions and actions."""
    return rule_repository.fetch_rules(domain=domain)


def fetch_rule_by_id(rule_id: str) -> dict[str, Any] | None:
    """Fetches a single rule dictionary by rule_id."""
    return rule_repository.fetch_rule_by_id(rule_id)


def create_rule(rule: RuleModel) -> str:
    """Persists a new rule and broadcasts cache invalidation."""
    rule_fields, conditions_data, actions_data = _extract_rule_payload(rule)
    new_id = rule_repository.insert_rule_with_details(rule_fields, conditions_data, actions_data)
    sync_rules_everywhere()
    return new_id


def update_rule(rule_id: str, rule: RuleModel) -> bool:
    """Updates an existing rule and broadcasts cache invalidation."""
    rule_fields, conditions_data, actions_data = _extract_rule_payload(rule)
    updated = rule_repository.replace_rule_with_details(
        rule_id, rule_fields, conditions_data, actions_data
    )
    if not updated:
        raise ResourceNotFoundError(f"Rule '{rule_id}' not found.")
    sync_rules_everywhere()
    return True


def delete_rule(rule_id: str) -> bool:
    """Deletes a rule and broadcasts cache invalidation."""
    deleted = rule_repository.delete_rule_with_details(rule_id)
    if not deleted:
        raise ResourceNotFoundError(f"Rule '{rule_id}' not found.")
    sync_rules_everywhere()
    return True


def reorder_rules(ordered_ids: list[str]) -> bool:
    """Reorders rule priorities and broadcasts cache invalidation."""
    if not ordered_ids:
        return False
    reordered = rule_repository.update_rule_priorities(ordered_ids)
    if reordered:
        sync_rules_everywhere()
    return reordered


def _evaluate_rule_with_fallback(
    rule_dict: dict[str, Any], sample_record: dict[str, Any]
) -> dict[str, Any]:
    try:
        return engine_gateway_service.test_rule_via_engine(rule_dict, sample_record)
    except ValueError:
        raise
    except Exception:
        return rule_evaluation_service.evaluate_rule_locally(rule_dict, sample_record)


def test_rule_draft(rule: RuleModel, sample_record: dict[str, Any]) -> dict[str, Any]:
    """Tests an unsaved draft rule against a sample record via Engine with local fallback."""
    rule_dict = rule.model_dump()
    rule_dict["conditions"] = [
        c if isinstance(c, dict) else c.model_dump() if hasattr(c, "model_dump") else dict(c)
        for c in rule_dict.get("conditions", [])
    ]
    rule_dict["actions"] = [
        a if isinstance(a, dict) else a.model_dump() if hasattr(a, "model_dump") else dict(a)
        for a in rule_dict.get("actions", [])
    ]
    return _evaluate_rule_with_fallback(rule_dict, dict(sample_record))


def test_stored_rule(rule_id: str, sample_record: dict[str, Any]) -> dict[str, Any]:
    """Tests a persisted rule against a sample record via Engine with local fallback."""
    rule_dict = fetch_rule_by_id(rule_id)
    if not rule_dict:
        raise ResourceNotFoundError(f"Rule '{rule_id}' not found.")
    return _evaluate_rule_with_fallback(rule_dict, dict(sample_record))


def rerun_rules(
    *,
    sku_name: str,
    domain: str,
    bt: str,
    gk: str,
    region: str = "",
    category: str = "",
    price: float = 0.0,
    confidence: float = 1.0,
    match_source: str = "Classifier",
    matched_sku: str = "",
    reasoning: str = "",
) -> dict[str, Any]:
    """Re-evaluates business rules for a single SKU via Engine with local fallback."""
    try:
        return engine_gateway_service.rerun_rules_via_engine(
            sku_name=sku_name,
            domain=domain,
            bt=bt,
            gk=gk,
            region=region,
            category=category,
            price=price,
            confidence=confidence,
            match_source=match_source,
            matched_sku=matched_sku,
            reasoning=reasoning,
        )
    except ValueError:
        raise
    except Exception:
        return rule_evaluation_service.rerun_rules_locally(
            sku_name=sku_name,
            domain=domain,
            bt=bt,
            gk=gk,
            region=region,
            category=category,
            price=price,
            confidence=confidence,
            match_source=match_source,
            matched_sku=matched_sku,
            reasoning=reasoning,
        )
