import functools
import logging

from engine.core import config
from engine.core.db import ensure_db_initialized

logger = logging.getLogger("matchops.rules_loader")

DB_PATH = config.DB_PATH

# In-memory cache: { domain: { module: [ rule_dicts ] } }
_RULES_CACHE = None

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
    global _RULES_CACHE
    try:
        conn = ensure_db_initialized(DB_PATH)
    except Exception as e:
        logger.warning(f"Could not initialize rules database at {DB_PATH}: {e}. Rules engine will be empty.")
        _RULES_CACHE = {}
        return

    cursor = conn.cursor()

    # 1. Load active rules
    cursor.execute("""
        SELECT id, rule_id, domain, priority, description, reasoning, condition_logic, is_active
        FROM rules WHERE is_active = 1
        ORDER BY priority ASC
    """)
    rules_rows = cursor.fetchall()

    rules_map = {}
    new_cache = {}

    for row in rules_rows:
        rule = Rule(row)
        rules_map[rule.rule_id] = rule

        if rule.domain not in new_cache:
            new_cache[rule.domain] = []

        new_cache[rule.domain].append(rule)

    # 2. Load conditions
    cursor.execute("SELECT rule_id, condition_group, condition_type, value, negate FROM conditions")
    for row in cursor.fetchall():
        rule_id = row[0]
        if rule_id in rules_map:
            rules_map[rule_id].conditions.append({
                "condition_group": row[1],
                "condition_type": row[2],
                "value": row[3],
                "negate": bool(row[4])
            })
            
    # 3. Load actions
    cursor.execute("SELECT rule_id, action_type, value FROM actions")
    for row in cursor.fetchall():
        rule_id = row[0]
        if rule_id in rules_map:
            rules_map[rule_id].actions.append({
                "action_type": row[1],
                "value": row[2]
            })
            
    conn.close()
    
    _RULES_CACHE = new_cache
    logger.info(f"Loaded {len(rules_map)} active rules into memory.")

@functools.lru_cache(maxsize=128)
def get_rules(domain: str):
    """Returns all active rules for a domain (plus 'shared' domain rules), sorted by priority."""
    global _RULES_CACHE
    if _RULES_CACHE is None:
        load_rules_from_db()

    rules = []
    # Load shared first
    if _RULES_CACHE and 'shared' in _RULES_CACHE:
        rules.extend(_RULES_CACHE['shared'])

    # Load domain specific second
    if _RULES_CACHE and domain in _RULES_CACHE:
        rules.extend(_RULES_CACHE[domain])

    # Sort by priority
    rules.sort(key=lambda r: r.priority)
    return rules

def refresh_rules_cache():
    """Forces a reload of the rules cache from DB."""
    global _RULES_CACHE
    _RULES_CACHE = None
    get_rules.cache_clear()
    load_rules_from_db()
