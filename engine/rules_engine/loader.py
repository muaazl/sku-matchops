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

from engine import config

logger = logging.getLogger("matchops.rules_loader")

# In-memory cache: { domain: [ Rule ] }
_RULES_CACHE = None
# Fingerprint of the loaded active rule set; part of the processed_skus idempotency key
_RULES_VERSION = "none"
# Guards swapping (_RULES_CACHE, _RULES_VERSION) as one unit
_STATE_LOCK = threading.Lock()
_REDIS_LOCK = threading.Lock()

# --- Cross-process rule distribution & invalidation via Redis ---
RULES_DATA_KEY = "matchops:rules:data"
RULES_VERSION_KEY = "matchops:rules:version"
RULES_INVALIDATE_CHANNEL = "matchops:rules:invalidate"

_redis_client = None
_redis_url: str | None = None
_invalidation_thread: threading.Thread | None = None


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


def _fingerprint(rules: list) -> str:
    fingerprint = [
        [r.rule_id, r.domain, r.priority, r.condition_logic, r.conditions, r.actions] for r in rules
    ]
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _redis():
    """Shared Redis client with short timeouts, so a Redis outage degrades instead of stalling."""
    global _redis_client, _redis_url
    url = config.REDIS_URL
    with _REDIS_LOCK:
        if _redis_client is not None and _redis_url != url:
            try:
                _redis_client.close()
            except Exception:
                pass
            _redis_client = None
            _redis_url = None
        if _redis_client is None:
            import redis

            _redis_client = redis.Redis.from_url(
                url, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
            )
            _redis_url = url
        return _redis_client


def set_rules_from_dicts(rules_data: list[dict], version: str | None = None) -> None:
    """Sets rules in cache from a list of rule dictionaries (from Redis snapshot or gRPC push)."""
    global _RULES_CACHE, _RULES_VERSION
    rules_map = {}
    new_cache: dict[str, list[Rule]] = {}

    for r in rules_data:
        rule_row = (
            r.get("id", 0),
            r["rule_id"],
            r.get("domain", "market"),
            r.get("priority", 100),
            r.get("description", ""),
            r.get("reasoning", ""),
            r.get("condition_logic", "AND"),
            r.get("is_active", 1),
        )
        rule = Rule(rule_row)
        rule.conditions = [
            {
                "condition_group": c.get("condition_group", 1),
                "condition_type": c["condition_type"],
                "value": c["value"],
                "negate": bool(c.get("negate", False)),
            }
            for c in r.get("conditions", [])
        ]
        rule.actions = [
            {"action_type": a["action_type"], "value": a["value"]} for a in r.get("actions", [])
        ]
        rules_map[rule.rule_id] = rule
        new_cache.setdefault(rule.domain, []).append(rule)

    all_rules = list(rules_map.values())
    v = version or _fingerprint(all_rules)

    with _STATE_LOCK:
        _RULES_CACHE, _RULES_VERSION = new_cache, v

    _cached_rules.cache_clear()
    logger.info(f"Loaded {len(rules_map)} rules into memory (version {v}).")


def load_rules_from_cache() -> None:
    """Loads active rules into the in-memory cache from Redis (`matchops:rules:data`).
    Failed reloads preserve the existing in-memory cache."""
    global _RULES_CACHE, _RULES_VERSION
    try:
        raw = _redis().get(RULES_DATA_KEY)
    except Exception as e:
        logger.debug(f"Redis rules cache not accessible: {e}")
        if _RULES_CACHE is None:
            with _STATE_LOCK:
                _RULES_CACHE, _RULES_VERSION = {}, "none"
        return

    if raw is not None:
        try:
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"Malformed rules JSON in Redis cache: {e}")
            if _RULES_CACHE is None:
                with _STATE_LOCK:
                    _RULES_CACHE, _RULES_VERSION = {}, "none"
            return
        if isinstance(data, list):
            set_rules_from_dicts(data)
            return

    empty_version = _fingerprint([])
    with _STATE_LOCK:
        _RULES_CACHE, _RULES_VERSION = {}, empty_version
    _cached_rules.cache_clear()


def _rules_for_domain(cache: dict, domain: str) -> list:
    """Shared rules first, then domain rules, sorted by priority."""
    rules = list(cache.get("shared", [])) + list(cache.get(domain, []))
    rules.sort(key=lambda r: r.priority)
    return rules


@functools.lru_cache(maxsize=128)
def _cached_rules(domain: str) -> list:
    """Unpinned lookups (no job context) use active rules."""
    if _RULES_CACHE is None:
        load_rules_from_cache()
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
    """Captures current rules and their version as one consistent pair."""
    if _RULES_CACHE is None:
        load_rules_from_cache()
    with _STATE_LOCK:
        return RulesSnapshot(
            version=_RULES_VERSION or _fingerprint([]),
            _cache=dict(_RULES_CACHE or {}),
        )


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
        load_rules_from_cache()
    return _RULES_VERSION


def refresh_rules_cache() -> None:
    """Forces a reload of the rules cache from Redis."""
    load_rules_from_cache()


def publish_rules_snapshot(rules_data: list[dict]) -> tuple[str, bool]:
    """Updates this process's in-memory rules cache, writes the serialized snapshot and version to
    Redis, and broadcasts invalidation on pub/sub. Returns `(version, published_to_redis)`."""
    set_rules_from_dicts(rules_data)
    version = _RULES_VERSION
    try:
        client = _redis()
        payload = json.dumps(rules_data, default=str)
        pipe = client.pipeline(transaction=True) if hasattr(client, "pipeline") else None
        if pipe is not None:
            pipe.set(RULES_DATA_KEY, payload)
            pipe.set(RULES_VERSION_KEY, version)
            pipe.publish(RULES_INVALIDATE_CHANNEL, version)
            pipe.execute()
        else:
            client.set(RULES_DATA_KEY, payload)
            client.set(RULES_VERSION_KEY, version)
            client.publish(RULES_INVALIDATE_CHANNEL, version)
        return version, True
    except Exception as e:
        logger.warning(f"[RULES] Redis unavailable, snapshot not broadcast: {e}")
        return version, False


def sync_rules_version() -> str:
    """Engine side: call at the start of each job. Costs one Redis GET; reloads from Redis only
    when another process published a newer rule_version. Returns the active rule_version."""
    local = get_rules_version()
    try:
        remote = _redis().get(RULES_VERSION_KEY)
    except Exception as e:
        logger.debug(
            f"[RULES] Redis unavailable ({e}); keeping in-memory rules ({_RULES_VERSION})."
        )
        return _RULES_VERSION

    if remote is None:
        try:
            _redis().set(RULES_VERSION_KEY, local, nx=True)
        except Exception:
            pass
    elif remote != local:
        refresh_rules_cache()
    return _RULES_VERSION


def start_invalidation_listener(stop_event: threading.Event | None = None) -> threading.Thread:
    """Engine side: daemon thread that reloads rules as soon as an invalidation is published."""
    global _invalidation_thread

    def _listen():
        import redis

        backoff = 1.0
        while not (stop_event and stop_event.is_set()):
            client = None
            pubsub = None
            try:
                client = redis.Redis.from_url(
                    config.REDIS_URL,
                    socket_connect_timeout=1,
                    socket_timeout=5,
                    health_check_interval=30,
                    decode_responses=True,
                )
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(RULES_INVALIDATE_CHANNEL)
                backoff = 1.0
                sync_rules_version()
                while not (stop_event and stop_event.is_set()):
                    msg = pubsub.get_message(timeout=1.0)
                    if msg and msg.get("data"):
                        refresh_rules_cache()
                        logger.info(f"[RULES] Reloaded rules (version {_RULES_VERSION}).")
            except Exception as e:
                logger.debug(f"[RULES] Invalidation listener disconnected: {e}")
                if stop_event and stop_event.wait(backoff):
                    break
                if not stop_event:
                    time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                if pubsub is not None:
                    try:
                        pubsub.close()
                    except Exception:
                        pass
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

    if stop_event is None:
        with _REDIS_LOCK:
            if _invalidation_thread is not None and _invalidation_thread.is_alive():
                return _invalidation_thread
            thread = threading.Thread(target=_listen, name="rules-invalidation", daemon=True)
            thread.start()
            _invalidation_thread = thread
            return thread

    thread = threading.Thread(target=_listen, name="rules-invalidation", daemon=True)
    thread.start()
    return thread
