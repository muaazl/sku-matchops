import hashlib
import json
from typing import Any

from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from backend.app.core import config
from backend.app.core.config import DEFAULT_TENANT
from backend.app.db.session import get_connection, next_seq_id
from backend.app.schemas.models import RuleModel
from backend.app.services.engine_client import refresh_engine_rules

RULES_DATA_KEY = "matchops:rules:data"
RULES_VERSION_KEY = "matchops:rules:version"
RULES_INVALIDATE_CHANNEL = "matchops:rules:invalidate"


def compute_rules_version(rules_data: list[dict[str, Any]]) -> str:
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
    import sys

    # In monorepo unit test environments where engine loader is loaded in the same process:
    if "engine.rules_engine.loader" in sys.modules:
        loader = sys.modules["engine.rules_engine.loader"]
        return loader.publish_rules_snapshot(rules_data)

    version = compute_rules_version(rules_data)
    try:
        import redis

        client = redis.Redis.from_url(
            config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
        )
        payload = json.dumps(rules_data, default=str)
        pipe = client.pipeline(transaction=True)
        pipe.set(RULES_DATA_KEY, payload)
        pipe.set(RULES_VERSION_KEY, version)
        pipe.publish(RULES_INVALIDATE_CHANNEL, version)
        pipe.execute()
        return version, True
    except Exception:
        return version, False


def get_rule_version(tenant_id: str = DEFAULT_TENANT) -> str:
    import sys

    if "engine.rules_engine.loader" in sys.modules:
        loader = sys.modules["engine.rules_engine.loader"]
        return loader.snapshot_rules(tenant_id).version

    try:
        import redis

        client = redis.Redis.from_url(
            config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
        )
        v = client.get(RULES_VERSION_KEY)
        if v:
            return v
    except Exception:
        pass
    active_rules = [
        r for r in load_active_rules_from_db() if r.get("tenant_id", DEFAULT_TENANT) == tenant_id
    ]
    return compute_rules_version(active_rules)


def load_active_rules_from_db(db: Connection | None = None) -> list[dict[str, Any]]:
    """Loads all active rules (across all tenants) from PostgreSQL for Redis/gRPC distribution."""
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    try:
        rule_rows = (
            conn.execute(
                text("""
                SELECT id, rule_id, domain, priority, description, reasoning,
                       condition_logic, is_active, tenant_id
                FROM rules
                WHERE is_active = 1
                ORDER BY priority ASC, rule_id ASC
                """)
            )
            .mappings()
            .fetchall()
        )
        if not rule_rows:
            return []

        rules = [dict(row) for row in rule_rows]
        rule_ids = [r["rule_id"] for r in rules]

        cond_rows = (
            conn.execute(
                text(
                    "SELECT rule_id, condition_group, condition_type, value, negate "
                    "FROM conditions WHERE rule_id = ANY(:ids) ORDER BY id"
                ),
                {"ids": rule_ids},
            )
            .mappings()
            .fetchall()
        )
        act_rows = (
            conn.execute(
                text(
                    "SELECT rule_id, action_type, value "
                    "FROM actions WHERE rule_id = ANY(:ids) ORDER BY id"
                ),
                {"ids": rule_ids},
            )
            .mappings()
            .fetchall()
        )
    finally:
        if owns_conn:
            conn.close()

    conds_by_rule: dict[str, list[dict[str, Any]]] = {}
    for c in cond_rows:
        conds_by_rule.setdefault(c["rule_id"], []).append(
            {
                "condition_group": c["condition_group"],
                "condition_type": c["condition_type"],
                "value": c["value"],
                "negate": bool(c["negate"]),
            }
        )

    acts_by_rule: dict[str, list[dict[str, Any]]] = {}
    for a in act_rows:
        acts_by_rule.setdefault(a["rule_id"], []).append(
            {"action_type": a["action_type"], "value": a["value"]}
        )

    for r in rules:
        r["tenant_id"] = r.get("tenant_id") or DEFAULT_TENANT
        r["conditions"] = conds_by_rule.get(r["rule_id"], [])
        r["actions"] = acts_by_rule.get(r["rule_id"], [])

    return rules


def sync_rules_everywhere(db: Connection | None = None) -> tuple[str, bool]:
    """Loads active rules from PostgreSQL, publishes the snapshot + version to Redis, and falls
    back to direct gRPC RefreshRules if Redis is unreachable."""
    rules_data = load_active_rules_from_db(db)
    version, published = publish_rules_snapshot(rules_data)
    if not published:
        refresh_engine_rules(rules_data=rules_data, version=version)
    return version, published


def _refresh_rules_everywhere(db: Connection | None = None) -> None:
    sync_rules_everywhere(db)


def _rule_fields(rule: RuleModel) -> dict[str, Any]:
    return {
        "domain": rule.domain,
        "priority": rule.priority,
        "description": rule.description,
        "reasoning": rule.reasoning,
        "condition_logic": rule.condition_logic,
        "is_active": rule.is_active,
    }


def _insert_conditions_and_actions(db: Connection, rule_id: str, rule: RuleModel) -> None:
    if rule.conditions:
        db.execute(
            text("""
            INSERT INTO conditions (rule_id, condition_group, condition_type, value, negate)
            VALUES (:rule_id, :condition_group, :condition_type, :value, :negate)
            """),
            [{"rule_id": rule_id, **cond.model_dump()} for cond in rule.conditions],
        )
    if rule.actions:
        db.execute(
            text(
                "INSERT INTO actions (rule_id, action_type, value) VALUES (:rule_id, :action_type, :value)"
            ),
            [{"rule_id": rule_id, **act.model_dump()} for act in rule.actions],
        )


def _owned(db: Connection, rule_id: str, tenant_id: str) -> bool:
    return (
        db.execute(
            text("SELECT 1 FROM rules WHERE rule_id = :rule_id AND tenant_id = :tenant_id"),
            {"rule_id": rule_id, "tenant_id": tenant_id},
        ).fetchone()
        is not None
    )


def get_next_rule_id(db: Connection) -> str:
    """Gets the next sequential rule ID starting from 1."""
    return next_seq_id(db, "rules_id_seq", "rules", "rule_id")


def get_rules_list(
    db: Connection, domain: str | None = None, tenant_id: str = DEFAULT_TENANT
) -> list[dict[str, Any]]:
    """Fetches the tenant's rules with their conditions and actions in batched queries."""
    query = "SELECT * FROM rules WHERE tenant_id = :tenant_id"
    params = {"tenant_id": tenant_id}

    if domain and domain != "all":
        query += " AND domain = :domain"
        params["domain"] = domain

    query += " ORDER BY priority ASC"
    rows = db.execute(text(query), params).mappings().fetchall()
    if not rows:
        return []

    rules = [dict(row) for row in rows]
    rule_ids = [r["rule_id"] for r in rules]

    # Batch fetch conditions and actions for all retrieved rules in 2 queries total
    cond_rows = (
        db.execute(
            text(
                "SELECT rule_id, condition_group, condition_type, value, negate FROM conditions WHERE rule_id = ANY(:ids) ORDER BY id"
            ),
            {"ids": rule_ids},
        )
        .mappings()
        .fetchall()
    )

    act_rows = (
        db.execute(
            text(
                "SELECT rule_id, action_type, value FROM actions WHERE rule_id = ANY(:ids) ORDER BY id"
            ),
            {"ids": rule_ids},
        )
        .mappings()
        .fetchall()
    )

    conds_by_rule: dict[str, list[dict[str, Any]]] = {}
    for c in cond_rows:
        c_dict = dict(c)
        rid = c_dict.pop("rule_id", None)
        if rid:
            conds_by_rule.setdefault(rid, []).append(c_dict)

    acts_by_rule: dict[str, list[dict[str, Any]]] = {}
    for a in act_rows:
        a_dict = dict(a)
        rid = a_dict.pop("rule_id", None)
        if rid:
            acts_by_rule.setdefault(rid, []).append(a_dict)

    for r in rules:
        r["conditions"] = conds_by_rule.get(r["rule_id"], [])
        r["actions"] = acts_by_rule.get(r["rule_id"], [])

    return rules


def get_rule_by_id(
    db: Connection, rule_id: str, tenant_id: str = DEFAULT_TENANT
) -> dict[str, Any] | None:
    """Fetches one of the tenant's rules by rule_id including conditions and actions."""
    params = {"rule_id": rule_id, "tenant_id": tenant_id}
    rule_row = (
        db.execute(
            text("SELECT * FROM rules WHERE rule_id = :rule_id AND tenant_id = :tenant_id"),
            params,
        )
        .mappings()
        .fetchone()
    )
    if not rule_row:
        return None

    rule_dict = dict(rule_row)
    conds = (
        db.execute(
            text(
                "SELECT condition_group, condition_type, value, negate FROM conditions WHERE rule_id = :rule_id ORDER BY id"
            ),
            params,
        )
        .mappings()
        .fetchall()
    )
    rule_dict["conditions"] = [dict(c) for c in conds]
    acts = (
        db.execute(
            text("SELECT action_type, value FROM actions WHERE rule_id = :rule_id ORDER BY id"),
            params,
        )
        .mappings()
        .fetchall()
    )
    rule_dict["actions"] = [dict(a) for a in acts]
    return rule_dict


def create_rule_entry(db: Connection, rule: RuleModel, tenant_id: str = DEFAULT_TENANT) -> str:
    """Inserts a new rule and its conditions/actions atomically, then refreshes cache.

    Retries on a rare rule_id collision (two near-simultaneous rule creations both computing
    the same "next id" before either commits) instead of surfacing an opaque IntegrityError.
    """
    max_attempts = 5
    new_id = None
    for attempt in range(1, max_attempts + 1):
        new_id = get_next_rule_id(db)
        try:
            db.execute(
                text("""
                INSERT INTO rules (rule_id, domain, priority, description, reasoning, condition_logic, is_active, tenant_id)
                VALUES (:rule_id, :domain, :priority, :description, :reasoning, :condition_logic, :is_active, :tenant_id)
                """),
                {"rule_id": new_id, "tenant_id": tenant_id, **_rule_fields(rule)},
            )
            _insert_conditions_and_actions(db, new_id, rule)
            db.commit()
            _refresh_rules_everywhere(db)
            return new_id
        except IntegrityError as e:
            db.rollback()
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Could not allocate a unique rule id after {max_attempts} attempts: {e}"
                )
            continue
        except Exception as e:
            db.rollback()
            raise e
    raise RuntimeError("Unreachable")


def update_rule_entry(
    db: Connection, rule_id: str, rule: RuleModel, tenant_id: str = DEFAULT_TENANT
) -> bool:
    """Updates one of the tenant's rules and replaces its conditions/actions, then refreshes cache."""
    params = {"rule_id": rule_id}
    if not _owned(db, rule_id, tenant_id):
        return False

    try:
        db.execute(
            text("""
            UPDATE rules
            SET domain = :domain, priority = :priority, description = :description, reasoning = :reasoning,
                condition_logic = :condition_logic, is_active = :is_active, updated_at = now()
            WHERE rule_id = :rule_id
            """),
            {"rule_id": rule_id, **_rule_fields(rule)},
        )

        # Clear and replace existing conditions and actions
        db.execute(text("DELETE FROM conditions WHERE rule_id = :rule_id"), params)
        db.execute(text("DELETE FROM actions WHERE rule_id = :rule_id"), params)
        _insert_conditions_and_actions(db, rule_id, rule)
        db.commit()
        _refresh_rules_everywhere(db)
        return True
    except Exception as e:
        db.rollback()
        raise e


def delete_rule_entry(db: Connection, rule_id: str, tenant_id: str = DEFAULT_TENANT) -> bool:
    """Deletes one of the tenant's rules and its conditions/actions, then refreshes cache."""
    params = {"rule_id": rule_id}
    if not _owned(db, rule_id, tenant_id):
        return False

    try:
        db.execute(text("DELETE FROM conditions WHERE rule_id = :rule_id"), params)
        db.execute(text("DELETE FROM actions WHERE rule_id = :rule_id"), params)
        db.execute(text("DELETE FROM rules WHERE rule_id = :rule_id"), params)
        db.commit()
        _refresh_rules_everywhere(db)
        return True
    except Exception as e:
        db.rollback()
        raise e


def reorder_rules_entries(
    db: Connection, ordered_ids: list[str], tenant_id: str = DEFAULT_TENANT
) -> bool:
    """Reorders the tenant's rules by priority according to the given ordered list of rule IDs
    (ids belonging to other tenants are ignored)."""
    if not ordered_ids:
        return False

    try:
        db.execute(
            text(
                "UPDATE rules SET priority = :priority, updated_at = now() "
                "WHERE rule_id = :rule_id AND tenant_id = :tenant_id"
            ),
            [
                {"priority": (index + 1) * 10, "rule_id": rule_id, "tenant_id": tenant_id}
                for index, rule_id in enumerate(ordered_ids)
            ],
        )
        db.commit()
        _refresh_rules_everywhere(db)
        return True
    except Exception as e:
        db.rollback()
        raise e
