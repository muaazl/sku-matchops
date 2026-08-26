import sqlite3
from typing import Optional, List, Dict, Any
from backend.app.schemas.models import RuleModel
from engine.rules_engine import refresh_rules_cache


def get_next_rule_id(db: sqlite3.Connection) -> str:
    """Gets the next sequential rule ID starting from 1."""
    try:
        row = db.execute("SELECT COALESCE(MAX(CAST(rule_id AS INTEGER)), 0) + 1 FROM rules WHERE rule_id GLOB '[0-9]*'").fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    return "1"


def get_rules_list(
    db: sqlite3.Connection,
    domain: Optional[str] = None,
    module: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetches all rules along with their associated conditions and actions in batched queries."""
    query = "SELECT * FROM rules WHERE 1=1"
    params = []
    
    if domain and domain != "all":
        query += " AND domain = ?"
        params.append(domain)
    if module and module != "all":
        query += " AND module = ?"
        params.append(module)
        
    query += " ORDER BY priority ASC"
    rows = db.execute(query, params).fetchall()
    if not rows:
        return []
    
    rules = [dict(row) for row in rows]
    rule_ids = [r["rule_id"] for r in rules]
    
    # Batch fetch conditions and actions for all retrieved rules in 2 queries total
    placeholders = ",".join("?" for _ in rule_ids)
    
    cond_rows = db.execute(
        f"SELECT rule_id, condition_group, condition_type, value, negate FROM conditions WHERE rule_id IN ({placeholders})",
        rule_ids
    ).fetchall()
    
    act_rows = db.execute(
        f"SELECT rule_id, action_type, value FROM actions WHERE rule_id IN ({placeholders})",
        rule_ids
    ).fetchall()
    
    conds_by_rule: Dict[str, List[Dict[str, Any]]] = {}
    for c in cond_rows:
        c_dict = dict(c)
        rid = c_dict.pop("rule_id", None)
        if rid:
            conds_by_rule.setdefault(rid, []).append(c_dict)
            
    acts_by_rule: Dict[str, List[Dict[str, Any]]] = {}
    for a in act_rows:
        a_dict = dict(a)
        rid = a_dict.pop("rule_id", None)
        if rid:
            acts_by_rule.setdefault(rid, []).append(a_dict)
            
    for r in rules:
        r["conditions"] = conds_by_rule.get(r["rule_id"], [])
        r["actions"] = acts_by_rule.get(r["rule_id"], [])
        
    return rules


def get_rule_by_id(db: sqlite3.Connection, rule_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single rule by rule_id including conditions and actions."""
    rule_row = db.execute("SELECT * FROM rules WHERE rule_id = ?", [rule_id]).fetchone()
    if not rule_row:
        return None
    
    rule_dict = dict(rule_row)
    conds = db.execute(
        "SELECT condition_group, condition_type, value, negate FROM conditions WHERE rule_id = ?",
        [rule_id]
    ).fetchall()
    rule_dict["conditions"] = [dict(c) for c in conds]
    acts = db.execute(
        "SELECT action_type, value FROM actions WHERE rule_id = ?",
        [rule_id]
    ).fetchall()
    rule_dict["actions"] = [dict(a) for a in acts]
    return rule_dict


def create_rule_entry(db: sqlite3.Connection, rule: RuleModel) -> str:
    """Inserts a new rule and its conditions/actions atomically into SQLite, then refreshes cache."""
    new_id = get_next_rule_id(db)
    try:
        db.execute(
            """
            INSERT INTO rules (rule_id, domain, module, priority, description, reasoning, condition_logic, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [new_id, rule.domain, rule.module, rule.priority, rule.description, rule.reasoning, rule.condition_logic, rule.is_active]
        )
        
        for cond in rule.conditions:
            db.execute(
                """
                INSERT INTO conditions (rule_id, condition_group, condition_type, value, negate)
                VALUES (?, ?, ?, ?, ?)
                """,
                [new_id, cond.condition_group, cond.condition_type, cond.value, cond.negate]
            )
            
        for act in rule.actions:
            db.execute(
                """
                INSERT INTO actions (rule_id, action_type, value)
                VALUES (?, ?, ?)
                """,
                [new_id, act.action_type, act.value]
            )
        db.commit()
        refresh_rules_cache()
        return new_id
    except Exception as e:
        db.rollback()
        raise e


def update_rule_entry(db: sqlite3.Connection, rule_id: str, rule: RuleModel) -> bool:
    """Updates an existing rule and replaces its conditions/actions, then refreshes cache."""
    existing = db.execute("SELECT rule_id FROM rules WHERE rule_id = ?", [rule_id]).fetchone()
    if not existing:
        return False
        
    try:
        db.execute(
            """
            UPDATE rules
            SET domain = ?, module = ?, priority = ?, description = ?, reasoning = ?, condition_logic = ?, is_active = ?, updated_at = datetime('now')
            WHERE rule_id = ?
            """,
            [rule.domain, rule.module, rule.priority, rule.description, rule.reasoning, rule.condition_logic, rule.is_active, rule_id]
        )
        
        # Clear and replace existing conditions and actions
        db.execute("DELETE FROM conditions WHERE rule_id = ?", [rule_id])
        db.execute("DELETE FROM actions WHERE rule_id = ?", [rule_id])
        
        for cond in rule.conditions:
            db.execute(
                """
                INSERT INTO conditions (rule_id, condition_group, condition_type, value, negate)
                VALUES (?, ?, ?, ?, ?)
                """,
                [rule_id, cond.condition_group, cond.condition_type, cond.value, cond.negate]
            )
            
        for act in rule.actions:
            db.execute(
                """
                INSERT INTO actions (rule_id, action_type, value)
                VALUES (?, ?, ?)
                """,
                [rule_id, act.action_type, act.value]
            )
        db.commit()
        refresh_rules_cache()
        return True
    except Exception as e:
        db.rollback()
        raise e


def delete_rule_entry(db: sqlite3.Connection, rule_id: str) -> bool:
    """Deletes a rule and its associated conditions/actions, then refreshes cache."""
    existing = db.execute("SELECT rule_id FROM rules WHERE rule_id = ?", [rule_id]).fetchone()
    if not existing:
        return False
        
    try:
        db.execute("DELETE FROM conditions WHERE rule_id = ?", [rule_id])
        db.execute("DELETE FROM actions WHERE rule_id = ?", [rule_id])
        db.execute("DELETE FROM rules WHERE rule_id = ?", [rule_id])
        db.commit()
        refresh_rules_cache()
        return True
    except Exception as e:
        db.rollback()
        raise e


def reorder_rules_entries(db: sqlite3.Connection, ordered_ids: List[str]) -> bool:
    """Reorders rules by priority according to the given ordered list of rule IDs."""
    if not ordered_ids:
        return False
        
    try:
        for index, rule_id in enumerate(ordered_ids):
            db.execute(
                "UPDATE rules SET priority = ?, updated_at = datetime('now') WHERE rule_id = ?",
                [(index + 1) * 10, rule_id]
            )
        db.commit()
        refresh_rules_cache()
        return True
    except Exception as e:
        db.rollback()
        raise e
