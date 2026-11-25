"""Redis cache and invalidation Pub/Sub operations for Business Rules snapshots."""

import json
import sys
from typing import Any

from backend.core import config

RULES_DATA_KEY = "matchops:rules:data"
RULES_VERSION_KEY = "matchops:rules:version"
RULES_INVALIDATE_CHANNEL = "matchops:rules:invalidate"


def get_rules_data_key() -> str:
    """Returns the Redis key storing the active rules JSON snapshot."""
    return RULES_DATA_KEY


def get_rules_version_key() -> str:
    """Returns the Redis key storing the active rules version hash."""
    return RULES_VERSION_KEY


def publish_rules_snapshot(
    rules_data: list[dict[str, Any]],
    version: str,
) -> tuple[str, bool]:
    """Publishes serialized active rules and version hash into Redis with Pub/Sub invalidation."""
    # Preserve monorepo in-process test bridge when engine rules loader is loaded in the same process.
    if "engine.pipelines.rules_loader_pipeline" in sys.modules:
        loader = sys.modules["engine.pipelines.rules_loader_pipeline"]
        return loader.publish_rules_snapshot(rules_data)

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


def fetch_cached_rule_version() -> str | None:
    """Reads the currently published rules version from Redis (or in-process engine loader)."""
    if "engine.pipelines.rules_loader_pipeline" in sys.modules:
        loader = sys.modules["engine.pipelines.rules_loader_pipeline"]
        return loader.snapshot_rules().version

    try:
        import redis

        client = redis.Redis.from_url(
            config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
        )
        version = client.get(RULES_VERSION_KEY)
        return str(version) if version else None
    except Exception:
        return None
