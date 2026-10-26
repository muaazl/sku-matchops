from typing import Any

from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from backend.app.schemas.models import RuleModel
from backend.app.services.engine_client import refresh_engine_rules
from engine.rules_engine import refresh_rules_cache


def _refresh_rules_everywhere() -> None:
    """Refreshes the rules cache in this (backend) process and notifies the live engine
    process to do the same, so rule changes take effect immediately for real jobs."""
    refresh_rules_cache()
    refresh_engine_rules()


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


def get_next_rule_id(db: Connection) -> str:
    """Gets the next sequential rule ID starting from 1."""
    row = db.execute(
        text(
            "SELECT COALESCE(MAX(CAST(rule_id AS BIGINT)), 0) + 1 FROM rules WHERE rule_id ~ '^[0-9]+$'"
        )
    ).fetchone()
    return str(row[0] if row and row[0] else 1)


def get_rules_list(db: Connection, domain: str | None = None) -> list[dict[str, Any]]:
    """Fetches all rules along with their associated conditions and actions in batched queries."""
    query = "SELECT * FROM rules WHERE 1=1"
    params = {}

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


def get_rule_by_id(db: Connection, rule_id: str) -> dict[str, Any] | None:
    """Fetches a single rule by rule_id including conditions and actions."""
    params = {"rule_id": rule_id}
    rule_row = (
        db.execute(text("SELECT * FROM rules WHERE rule_id = :rule_id"), params)
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


def create_rule_entry(db: Connection, rule: RuleModel) -> str:
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
                INSERT INTO rules (rule_id, domain, priority, description, reasoning, condition_logic, is_active)
                VALUES (:rule_id, :domain, :priority, :description, :reasoning, :condition_logic, :is_active)
                """),
                {"rule_id": new_id, **_rule_fields(rule)},
            )
            _insert_conditions_and_actions(db, new_id, rule)
            db.commit()
            _refresh_rules_everywhere()
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


def update_rule_entry(db: Connection, rule_id: str, rule: RuleModel) -> bool:
    """Updates an existing rule and replaces its conditions/actions, then refreshes cache."""
    params = {"rule_id": rule_id}
    existing = db.execute(
        text("SELECT rule_id FROM rules WHERE rule_id = :rule_id"), params
    ).fetchone()
    if not existing:
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
        _refresh_rules_everywhere()
        return True
    except Exception as e:
        db.rollback()
        raise e


def delete_rule_entry(db: Connection, rule_id: str) -> bool:
    """Deletes a rule and its associated conditions/actions, then refreshes cache."""
    params = {"rule_id": rule_id}
    existing = db.execute(
        text("SELECT rule_id FROM rules WHERE rule_id = :rule_id"), params
    ).fetchone()
    if not existing:
        return False

    try:
        db.execute(text("DELETE FROM conditions WHERE rule_id = :rule_id"), params)
        db.execute(text("DELETE FROM actions WHERE rule_id = :rule_id"), params)
        db.execute(text("DELETE FROM rules WHERE rule_id = :rule_id"), params)
        db.commit()
        _refresh_rules_everywhere()
        return True
    except Exception as e:
        db.rollback()
        raise e


def reorder_rules_entries(db: Connection, ordered_ids: list[str]) -> bool:
    """Reorders rules by priority according to the given ordered list of rule IDs."""
    if not ordered_ids:
        return False

    try:
        db.execute(
            text(
                "UPDATE rules SET priority = :priority, updated_at = now() WHERE rule_id = :rule_id"
            ),
            [
                {"priority": (index + 1) * 10, "rule_id": rule_id}
                for index, rule_id in enumerate(ordered_ids)
            ],
        )
        db.commit()
        _refresh_rules_everywhere()
        return True
    except Exception as e:
        db.rollback()
        raise e
