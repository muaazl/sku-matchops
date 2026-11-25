"""SQLAlchemy repository for Business Rules, conditions, and actions."""

from typing import Any

from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from backend.core.database import get_connection
from backend.repositories.sequence_repository import allocate_next_rule_id


def _insert_conditions_and_actions(
    conn: Connection,
    rule_id: str,
    conditions_data: list[dict[str, Any]],
    actions_data: list[dict[str, Any]],
) -> None:
    if conditions_data:
        conn.execute(
            text(
                """
                INSERT INTO conditions (rule_id, condition_group, condition_type, value, negate, operator)
                VALUES (:rule_id, :condition_group, :condition_type, :value, :negate, :operator)
                """
            ),
            [{"rule_id": rule_id, **cond} for cond in conditions_data],
        )
    if actions_data:
        conn.execute(
            text(
                "INSERT INTO actions (rule_id, action_type, value) VALUES (:rule_id, :action_type, :value)"
            ),
            [{"rule_id": rule_id, **act} for act in actions_data],
        )


def check_rule_exists(conn: Connection, rule_id: str) -> bool:
    """Returns True if a rule with `rule_id` exists in `rules`."""
    return (
        conn.execute(
            text("SELECT 1 FROM rules WHERE rule_id = :rule_id"),
            {"rule_id": rule_id},
        ).fetchone()
        is not None
    )


def fetch_active_rules(db: Connection | None = None) -> list[dict[str, Any]]:
    """Fetches all active rules with batched conditions and actions ordered by priority."""
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    try:
        rule_rows = (
            conn.execute(
                text(
                    """
                    SELECT id, rule_id, domain, priority, description, reasoning,
                           condition_logic, is_active
                    FROM rules
                    WHERE is_active = 1
                    ORDER BY priority ASC, rule_id ASC
                    """
                )
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
                    "SELECT rule_id, condition_group, condition_type, value, negate, operator "
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
                "operator": c.get("operator", "AND"),
            }
        )

    acts_by_rule: dict[str, list[dict[str, Any]]] = {}
    for a in act_rows:
        acts_by_rule.setdefault(a["rule_id"], []).append(
            {"action_type": a["action_type"], "value": a["value"]}
        )

    for r in rules:
        r["conditions"] = conds_by_rule.get(r["rule_id"], [])
        r["actions"] = acts_by_rule.get(r["rule_id"], [])
    return rules


def fetch_rules(domain: str | None = None, db: Connection | None = None) -> list[dict[str, Any]]:
    """Fetches rules (optionally filtered by domain) with batched conditions and actions."""
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    try:
        if domain and domain != "all":
            query = "SELECT * FROM rules WHERE domain = :domain ORDER BY priority ASC"
            params: dict[str, Any] = {"domain": domain}
        else:
            query = "SELECT * FROM rules ORDER BY priority ASC"
            params = {}

        rows = conn.execute(text(query), params).mappings().fetchall()
        if not rows:
            return []

        rules = [dict(row) for row in rows]
        rule_ids = [r["rule_id"] for r in rules]

        cond_rows = (
            conn.execute(
                text(
                    "SELECT rule_id, condition_group, condition_type, value, negate, operator "
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
                    "SELECT rule_id, action_type, value FROM actions WHERE rule_id = ANY(:ids) ORDER BY id"
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


def fetch_rule_by_id(rule_id: str, db: Connection | None = None) -> dict[str, Any] | None:
    """Fetches a single rule by `rule_id` along with its conditions and actions."""
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    try:
        params = {"rule_id": rule_id}
        rule_row = (
            conn.execute(text("SELECT * FROM rules WHERE rule_id = :rule_id"), params)
            .mappings()
            .fetchone()
        )
        if not rule_row:
            return None

        rule_dict = dict(rule_row)
        conds = (
            conn.execute(
                text(
                    "SELECT condition_group, condition_type, value, negate, operator "
                    "FROM conditions WHERE rule_id = :rule_id ORDER BY id"
                ),
                params,
            )
            .mappings()
            .fetchall()
        )
        rule_dict["conditions"] = [dict(c) for c in conds]
        acts = (
            conn.execute(
                text("SELECT action_type, value FROM actions WHERE rule_id = :rule_id ORDER BY id"),
                params,
            )
            .mappings()
            .fetchall()
        )
        rule_dict["actions"] = [dict(a) for a in acts]
        return rule_dict
    finally:
        if owns_conn:
            conn.close()


def insert_rule_with_details(
    rule_fields: dict[str, Any],
    conditions_data: list[dict[str, Any]],
    actions_data: list[dict[str, Any]],
    db: Connection | None = None,
) -> str:
    """Inserts a new rule, conditions, and actions atomically with ID collision retry."""
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    max_attempts = 5
    try:
        for attempt in range(1, max_attempts + 1):
            new_id = allocate_next_rule_id(conn)
            try:
                conn.execute(
                    text(
                        """
                        INSERT INTO rules (rule_id, domain, priority, description, reasoning, condition_logic, is_active)
                        VALUES (:rule_id, :domain, :priority, :description, :reasoning, :condition_logic, :is_active)
                        """
                    ),
                    {"rule_id": new_id, **rule_fields},
                )
                _insert_conditions_and_actions(conn, new_id, conditions_data, actions_data)
                conn.commit()
                return new_id
            except IntegrityError as e:
                conn.rollback()
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Could not allocate a unique rule id after {max_attempts} attempts: {e}"
                    )
            except Exception:
                conn.rollback()
                raise
    finally:
        if owns_conn:
            conn.close()
    raise RuntimeError("Unreachable")


def replace_rule_with_details(
    rule_id: str,
    rule_fields: dict[str, Any],
    conditions_data: list[dict[str, Any]],
    actions_data: list[dict[str, Any]],
    db: Connection | None = None,
) -> bool:
    """Updates an existing rule and replaces its conditions and actions atomically."""
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    params = {"rule_id": rule_id}
    try:
        if not check_rule_exists(conn, rule_id):
            return False
        conn.execute(
            text(
                """
                UPDATE rules
                SET domain = :domain, priority = :priority, description = :description, reasoning = :reasoning,
                    condition_logic = :condition_logic, is_active = :is_active, updated_at = now()
                WHERE rule_id = :rule_id
                """
            ),
            {"rule_id": rule_id, **rule_fields},
        )
        conn.execute(text("DELETE FROM conditions WHERE rule_id = :rule_id"), params)
        conn.execute(text("DELETE FROM actions WHERE rule_id = :rule_id"), params)
        _insert_conditions_and_actions(conn, rule_id, conditions_data, actions_data)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_conn:
            conn.close()


def delete_rule_with_details(rule_id: str, db: Connection | None = None) -> bool:
    """Deletes a rule and its dependent conditions and actions atomically."""
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    params = {"rule_id": rule_id}
    try:
        if not check_rule_exists(conn, rule_id):
            return False
        conn.execute(text("DELETE FROM conditions WHERE rule_id = :rule_id"), params)
        conn.execute(text("DELETE FROM actions WHERE rule_id = :rule_id"), params)
        conn.execute(text("DELETE FROM rules WHERE rule_id = :rule_id"), params)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_conn:
            conn.close()


def update_rule_priorities(ordered_ids: list[str], db: Connection | None = None) -> bool:
    """Updates rule priorities in batches of 10 according to `ordered_ids`."""
    if not ordered_ids:
        return False
    owns_conn = db is None
    conn = get_connection() if owns_conn else db
    assert conn is not None
    try:
        conn.execute(
            text(
                "UPDATE rules SET priority = :priority, updated_at = now() WHERE rule_id = :rule_id"
            ),
            [
                {"priority": (index + 1) * 10, "rule_id": rule_id}
                for index, rule_id in enumerate(ordered_ids)
            ],
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_conn:
            conn.close()
