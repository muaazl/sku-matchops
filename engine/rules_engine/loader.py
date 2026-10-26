import contextvars
import functools
import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import text

from engine import config
from engine.core.db import get_connection

logger = logging.getLogger("matchops.rules_loader")

# In-memory cache: { domain: { module: [ rule_dicts ] } }
_RULES_CACHE = None
# Fingerprint of the loaded active rule set; part of the processed_skus idempotency key
_RULES_VERSION = "none"
# Guards swapping (_RULES_CACHE, _RULES_VERSION) as one pair
_STATE_LOCK = threading.Lock()


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
        if _RULES_CACHE is not None:
            logger.warning(f"Could not reload rules from the database: {e}. Keeping cached rules.")
            return
        logger.warning(f"Could not load rules from the database: {e}. Rules engine will be empty.")
        with _STATE_LOCK:
            _RULES_CACHE, _RULES_VERSION = {}, "none"
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

    fingerprint = [
        [r.rule_id, r.domain, r.priority, r.condition_logic, r.conditions, r.actions]
        for r in rules_map.values()
    ]
    version = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    with _STATE_LOCK:
        _RULES_CACHE, _RULES_VERSION = new_cache, version
    logger.info(f"Loaded {len(rules_map)} active rules into memory.")


def _rules_for_domain(cache: dict, domain: str) -> list:
    """Shared rules first, then domain rules, sorted by priority."""
    rules = list(cache.get("shared", [])) + list(cache.get(domain, []))
    rules.sort(key=lambda r: r.priority)
    return rules


@functools.lru_cache(maxsize=128)
def _cached_rules(domain: str) -> list:
    if _RULES_CACHE is None:
        load_rules_from_db()
    return _rules_for_domain(_RULES_CACHE or {}, domain)


@dataclass(frozen=True)
class RulesSnapshot:
    """An immutable (rules, rule_version) pair, pinned to one job so reloads can't drift it."""

    version: str
    _cache: dict

    def rules(self, domain: str) -> tuple:
        return tuple(_rules_for_domain(self._cache, domain))


_PINNED: contextvars.ContextVar[RulesSnapshot | None] = contextvars.ContextVar(
    "pinned_rules", default=None
)


def get_rules(domain: str):
    """Returns all active rules for a domain (plus 'shared' domain rules), sorted by priority.
    Inside pinned_rules(), returns the job's snapshot instead of the live cache."""
    pinned = _PINNED.get()
    if pinned is not None:
        return pinned.rules(domain)
    return _cached_rules(domain)


def snapshot_rules() -> RulesSnapshot:
    """Captures the current rules and their version as one consistent pair."""
    if _RULES_CACHE is None:
        load_rules_from_db()
    # Reloads build a brand-new dict and Rule objects, so holding a reference is a stable copy
    with _STATE_LOCK:
        return RulesSnapshot(version=_RULES_VERSION, _cache=_RULES_CACHE or {})


@contextmanager
def pinned_rules(snapshot: RulesSnapshot) -> Iterator[RulesSnapshot]:
    """Makes get_rules() serve `snapshot` for the current thread/context (one job)."""
    token = _PINNED.set(snapshot)
    try:
        yield snapshot
    finally:
        _PINNED.reset(token)


def get_rules_version() -> str:
    """Returns the fingerprint of the active rule set currently loaded in memory."""
    if _RULES_CACHE is None:
        load_rules_from_db()
    return _RULES_VERSION


def refresh_rules_cache():
    """Forces a reload of the rules cache from DB (swapped in atomically; readers never see it empty)."""
    load_rules_from_db()
    _cached_rules.cache_clear()


# --- Cross-process invalidation via Redis ---
RULES_VERSION_KEY = "matchops:rules:version"
RULES_INVALIDATE_CHANNEL = "matchops:rules:invalidate"

_redis_client = None


def _redis():
    """Shared Redis client with short timeouts, so a Redis outage degrades instead of stalling."""
    global _redis_client
    if _redis_client is None:
        import redis

        _redis_client = redis.Redis.from_url(
            config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
        )
    return _redis_client


def publish_rules_invalidation() -> bool:
    """
    Gateway side: call after a rule change is committed. Reloads this process's cache, stores the
    new rule_version in Redis and broadcasts it. Returns False if Redis is unreachable.
    """
    refresh_rules_cache()
    version = _RULES_VERSION
    try:
        client = _redis()
        client.set(RULES_VERSION_KEY, version)
        client.publish(RULES_INVALIDATE_CHANNEL, version)
        return True
    except Exception as e:
        logger.warning(f"[RULES] Redis unavailable, invalidation not broadcast: {e}")
        return False


def sync_rules_version() -> str:
    """
    Engine side: call at the start of each job. Costs one Redis GET; reloads from PostgreSQL only
    when another process published a newer rule_version. If Redis is down, reloads from
    PostgreSQL directly so rule changes are never missed. Returns the active rule_version.
    """
    local = get_rules_version()
    try:
        remote = _redis().get(RULES_VERSION_KEY)
    except Exception as e:
        logger.warning(f"[RULES] Redis unavailable ({e}); reloading rules from PostgreSQL.")
        refresh_rules_cache()
        return _RULES_VERSION

    if remote is None:
        # Fresh Redis: seed it with what PostgreSQL holds now
        try:
            _redis().set(RULES_VERSION_KEY, local, nx=True)
        except Exception:
            pass
    elif remote != local:
        refresh_rules_cache()
    return _RULES_VERSION


def start_invalidation_listener() -> threading.Thread:
    """Engine side: daemon thread that reloads rules as soon as an invalidation is published."""

    def _listen():
        import redis

        backoff = 1.0
        while True:
            try:
                # Own connection: no read timeout while idle, health-check pings detect dead sockets
                client = redis.Redis.from_url(
                    config.REDIS_URL,
                    socket_connect_timeout=1,
                    health_check_interval=30,
                    decode_responses=True,
                )
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(RULES_INVALIDATE_CHANNEL)
                backoff = 1.0
                # Catch up on anything published while we were disconnected
                sync_rules_version()
                while True:
                    msg = pubsub.get_message(timeout=30)
                    if msg and msg.get("data") != _RULES_VERSION:
                        refresh_rules_cache()
                        logger.info(f"[RULES] Reloaded rules (version {_RULES_VERSION}).")
            except Exception as e:
                logger.debug(f"[RULES] Invalidation listener disconnected: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    thread = threading.Thread(target=_listen, name="rules-invalidation", daemon=True)
    thread.start()
    return thread
