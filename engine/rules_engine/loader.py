import functools
import hashlib
import json
import logging

from sqlalchemy import text

from engine.core.db import get_connection

logger = logging.getLogger("matchops.rules_loader")

# In-memory cache: { domain: { module: [ rule_dicts ] } }
_RULES_CACHE = None
# Fingerprint of the loaded active rule set; part of the processed_skus idempotency key
_RULES_VERSION = "none"


class Rule:
    def __init__(self, row):
        self.id = row[0]
        self.rule_id = row[1]
        self.domain = row[2]
        self.priority = row[3]
        self.description = row[4]
        self.reasoning = row[5]
        self.condition_logic = row[6]
        self.is_active = row[7]
        self.conditions = []
        self.actions = []


def load_rules_from_db():
    """Loads all active rules from the database into the in-memory cache."""
    global _RULES_CACHE, _RULES_VERSION
    try:
        with get_connection() as conn:
            rules_rows = conn.execute(
                text("""
                SELECT id, rule_id, domain, priority, description, reasoning, condition_logic, is_active
                FROM rules WHERE is_active = 1
                ORDER BY priority ASC, rule_id ASC
            """)
            ).fetchall()
            cond_rows = conn.execute(
                text(
                    "SELECT rule_id, condition_group, condition_type, value, negate FROM conditions ORDER BY id"
                )
            ).fetchall()
            act_rows = conn.execute(
                text("SELECT rule_id, action_type, value FROM actions ORDER BY id")
            ).fetchall()
    except Exception as e:
        logger.warning(f"Could not load rules from the database: {e}. Rules engine will be empty.")
        _RULES_CACHE = {}
        _RULES_VERSION = "none"
        return

    rules_map = {}
    new_cache = {}

    # 1. Active rules
    for row in rules_rows:
        rule = Rule(row)
        rules_map[rule.rule_id] = rule

        if rule.domain not in new_cache:
            new_cache[rule.domain] = []

        new_cache[rule.domain].append(rule)

    # 2. Conditions
    for row in cond_rows:
        rule_id = row[0]
        if rule_id in rules_map:
            rules_map[rule_id].conditions.append(
                {
                    "condition_group": row[1],
                    "condition_type": row[2],
                    "value": row[3],
                    "negate": bool(row[4]),
                }
            )

    # 3. Actions
    for row in act_rows:
        rule_id = row[0]
        if rule_id in rules_map:
            rules_map[rule_id].actions.append({"action_type": row[1], "value": row[2]})

    _RULES_CACHE = new_cache
    fingerprint = [
        [r.rule_id, r.domain, r.priority, r.condition_logic, r.conditions, r.actions]
        for r in rules_map.values()
    ]
    _RULES_VERSION = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    logger.info(f"Loaded {len(rules_map)} active rules into memory.")


@functools.lru_cache(maxsize=128)
def get_rules(domain: str):
    """Returns all active rules for a domain (plus 'shared' domain rules), sorted by priority."""
    global _RULES_CACHE
    if _RULES_CACHE is None:
        load_rules_from_db()

    rules = []
    # Load shared first
    if _RULES_CACHE and "shared" in _RULES_CACHE:
        rules.extend(_RULES_CACHE["shared"])

    # Load domain specific second
    if _RULES_CACHE and domain in _RULES_CACHE:
        rules.extend(_RULES_CACHE[domain])

    # Sort by priority
    rules.sort(key=lambda r: r.priority)
    return rules


def get_rules_version() -> str:
    """Returns the fingerprint of the active rule set currently loaded in memory."""
    if _RULES_CACHE is None:
        load_rules_from_db()
    return _RULES_VERSION


def refresh_rules_cache():
    """Forces a reload of the rules cache from DB."""
    global _RULES_CACHE
    _RULES_CACHE = None
    get_rules.cache_clear()
    load_rules_from_db()
