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

# In-memory cache: { tenant_id: { domain: [ Rule ] } }
_RULES_CACHE = None
# Fingerprint of the loaded active rule set; part of the processed_skus idempotency key
_RULES_VERSION = "none"
# Per-tenant fingerprints: the rule_version stored with a tenant's job results
_TENANT_VERSIONS: dict[str, str] = {}
# Guards swapping (_RULES_CACHE, _RULES_VERSION, _TENANT_VERSIONS) as one unit
_STATE_LOCK = threading.Lock()
DEFAULT_TENANT = "default"

# --- Cross-process rule distribution & invalidation via Redis ---
RULES_DATA_KEY = "matchops:rules:data"
RULES_VERSION_KEY = "matchops:rules:version"
RULES_INVALIDATE_CHANNEL = "matchops:rules:invalidate"

_redis_client = None


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
        self.tenant_id = row[8] if len(row) > 8 else DEFAULT_TENANT
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
    global _redis_client
    if _redis_client is None:
        import redis

        _redis_client = redis.Redis.from_url(
            config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
        )
    return _redis_client


def set_rules_from_dicts(
    rules_data: list[dict], version: str | None = None, tenant_id: str | None = None
) -> None:
    """Sets rules in cache from a list of rule dictionaries (from Redis snapshot or gRPC push)."""
    global _RULES_CACHE, _RULES_VERSION, _TENANT_VERSIONS
    rules_map = {}
    new_tenant_cache: dict[str, list[Rule]] = {}
    new_cache: dict[str, dict[str, list[Rule]]] = dict(_RULES_CACHE or {}) if tenant_id else {}

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
            r.get("tenant_id", tenant_id or DEFAULT_TENANT),
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
        if tenant_id:
            new_tenant_cache.setdefault(rule.domain, []).append(rule)
        else:
            new_cache.setdefault(rule.tenant_id, {}).setdefault(rule.domain, []).append(rule)

    all_rules = list(rules_map.values())
    v = version or _fingerprint(all_rules)

    with _STATE_LOCK:
        if tenant_id:
            new_cache[tenant_id] = new_tenant_cache
            new_tenant_versions = dict(_TENANT_VERSIONS or {})
            new_tenant_versions[tenant_id] = v
            _RULES_CACHE = new_cache
            _TENANT_VERSIONS = new_tenant_versions
        else:
            tenant_rules: dict[str, list[Rule]] = {}
            for rule in all_rules:
                tenant_rules.setdefault(rule.tenant_id, []).append(rule)
            tenant_versions = {t: _fingerprint(rs) for t, rs in tenant_rules.items()}
            _RULES_CACHE, _RULES_VERSION, _TENANT_VERSIONS = new_cache, v, tenant_versions

    _cached_rules.cache_clear()
    logger.info(
        f"Loaded {len(rules_map)} rules into memory (version {v}, tenant={tenant_id or 'all'})."
    )


def load_rules_from_cache(tenant_id: str | None = None) -> None:
    """Loads active rules into the in-memory cache from Redis (`matchops:rules:data` or tenant key).
    Failed reloads preserve the existing in-memory cache."""
    global _RULES_CACHE, _RULES_VERSION, _TENANT_VERSIONS
    data_key = f"matchops:rules:data:{tenant_id}" if tenant_id else RULES_DATA_KEY
    try:
        raw = _redis().get(data_key)
        if raw is None and tenant_id:
            raw = _redis().get(RULES_DATA_KEY)
    except Exception as e:
        logger.debug(f"Redis rules cache not accessible: {e}")
        if _RULES_CACHE is None:
            with _STATE_LOCK:
                _RULES_CACHE, _RULES_VERSION, _TENANT_VERSIONS = {}, "none", {}
        return

    if raw is not None:
        try:
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"Malformed rules JSON in Redis cache: {e}")
            if _RULES_CACHE is None:
                with _STATE_LOCK:
                    _RULES_CACHE, _RULES_VERSION, _TENANT_VERSIONS = {}, "none", {}
            return
        if isinstance(data, list):
            set_rules_from_dicts(data, tenant_id=tenant_id)
            return

    empty_version = _fingerprint([])
    with _STATE_LOCK:
        _RULES_CACHE, _RULES_VERSION, _TENANT_VERSIONS = {}, empty_version, {}
    _cached_rules.cache_clear()


def _rules_for_domain(cache: dict, domain: str) -> list:
    """Shared rules first, then domain rules, sorted by priority."""
    rules = list(cache.get("shared", [])) + list(cache.get(domain, []))
    rules.sort(key=lambda r: r.priority)
    return rules


@functools.lru_cache(maxsize=128)
def _cached_rules(domain: str) -> list:
    """Unpinned lookups (no job context) use the default tenant's rules."""
    if _RULES_CACHE is None:
        load_rules_from_cache()
    return _rules_for_domain((_RULES_CACHE or {}).get(DEFAULT_TENANT, {}), domain)


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


def snapshot_rules(tenant_id: str = DEFAULT_TENANT) -> RulesSnapshot:
    """Captures one tenant's current rules and their version as one consistent pair.
    Other tenants' rules are never visible through the snapshot."""
    if _RULES_CACHE is None:
        load_rules_from_cache()
    # Reloads build a brand-new dict and Rule objects, so holding a reference is a stable copy
    with _STATE_LOCK:
        return RulesSnapshot(
            version=_TENANT_VERSIONS.get(tenant_id) or _fingerprint([]),
            _cache=(_RULES_CACHE or {}).get(tenant_id, {}),
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


def refresh_rules_cache(tenant_id: str | None = None) -> None:
    """Forces a reload of the rules cache from Redis (swapped in atomically; readers never see it empty)."""
    load_rules_from_cache(tenant_id=tenant_id)


def publish_rules_snapshot(
    rules_data: list[dict], tenant_id: str | None = None
) -> tuple[str, bool]:
    """Updates this process's in-memory rules cache, writes the serialized snapshot and version to
    Redis, and broadcasts invalidation on pub/sub. Returns `(version, published_to_redis)`."""
    set_rules_from_dicts(rules_data, tenant_id=tenant_id)
    version = _TENANT_VERSIONS.get(tenant_id) if tenant_id else _RULES_VERSION
    version = version or _RULES_VERSION
    try:
        client = _redis()
        payload = json.dumps(rules_data, default=str)
        pipe = client.pipeline(transaction=True) if hasattr(client, "pipeline") else None
        if pipe is not None:
            pipe.set(RULES_DATA_KEY, payload)
            pipe.set(RULES_VERSION_KEY, _RULES_VERSION)
            if tenant_id:
                pipe.set(f"matchops:rules:data:{tenant_id}", payload)
                pipe.set(f"matchops:rules:version:{tenant_id}", version)
                pipe.publish(RULES_INVALIDATE_CHANNEL, f"{tenant_id}:{version}")
            else:
                for t, tv in _TENANT_VERSIONS.items():
                    t_rules = [r for r in rules_data if r.get("tenant_id", DEFAULT_TENANT) == t]
                    pipe.set(f"matchops:rules:data:{t}", json.dumps(t_rules, default=str))
                    pipe.set(f"matchops:rules:version:{t}", tv)
                pipe.publish(RULES_INVALIDATE_CHANNEL, _RULES_VERSION)
            pipe.execute()
        else:
            client.set(RULES_DATA_KEY, payload)
            client.set(RULES_VERSION_KEY, _RULES_VERSION)
            if tenant_id:
                client.set(f"matchops:rules:data:{tenant_id}", payload)
                client.set(f"matchops:rules:version:{tenant_id}", version)
                client.publish(RULES_INVALIDATE_CHANNEL, f"{tenant_id}:{version}")
            else:
                for t, tv in _TENANT_VERSIONS.items():
                    t_rules = [r for r in rules_data if r.get("tenant_id", DEFAULT_TENANT) == t]
                    client.set(f"matchops:rules:data:{t}", json.dumps(t_rules, default=str))
                    client.set(f"matchops:rules:version:{t}", tv)
                client.publish(RULES_INVALIDATE_CHANNEL, _RULES_VERSION)
        return version, True
    except Exception as e:
        logger.warning(f"[RULES] Redis unavailable, snapshot not broadcast: {e}")
        return version, False


def sync_rules_version(tenant_id: str | None = None) -> str:
    """Engine side: call at the start of each job. Costs one Redis GET; reloads from Redis only
    when another process published a newer rule_version. If Redis is down, preserves the current
    in-memory rule set (updated via gRPC RefreshRules fallback). Returns the active rule_version."""
    local = _TENANT_VERSIONS.get(tenant_id) if tenant_id else get_rules_version()
    v_key = f"matchops:rules:version:{tenant_id}" if tenant_id else RULES_VERSION_KEY
    try:
        remote = _redis().get(v_key)
        if remote is None and tenant_id:
            remote = _redis().get(RULES_VERSION_KEY)
    except Exception as e:
        logger.debug(
            f"[RULES] Redis unavailable ({e}); keeping in-memory rules ({_RULES_VERSION})."
        )
        return _TENANT_VERSIONS.get(tenant_id) if tenant_id else _RULES_VERSION

    if remote is None:
        try:
            _redis().set(v_key, local, nx=True)
        except Exception:
            pass
    elif remote != local:
        refresh_rules_cache(tenant_id=tenant_id)
    return _TENANT_VERSIONS.get(tenant_id) if tenant_id else _RULES_VERSION


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
                    if msg and msg.get("data"):
                        inv_data = str(msg["data"])
                        inv_tenant = None
                        if ":" in inv_data:
                            inv_tenant, _ = inv_data.split(":", 1)
                        refresh_rules_cache(tenant_id=inv_tenant)
                        logger.info(
                            f"[RULES] Reloaded rules (tenant={inv_tenant}, version {_RULES_VERSION})."
                        )
            except Exception as e:
                logger.debug(f"[RULES] Invalidation listener disconnected: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    thread = threading.Thread(target=_listen, name="rules-invalidation", daemon=True)
    thread.start()
    return thread
