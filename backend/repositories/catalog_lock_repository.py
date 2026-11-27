"""Redis distributed lock repository for cluster-wide catalog cache rebuilds."""

import logging

from backend.core import config

logger = logging.getLogger("matchops.repositories.catalog_lock")

BUILD_LOCK_KEY = "matchops:lock:catalog_build"


def get_catalog_lock_redis_client():
    """Creates a short-timeout Redis client for distributed build locking."""
    try:
        import redis

        return redis.Redis.from_url(
            config.REDIS_URL,
            socket_connect_timeout=1,
            socket_timeout=2,
            decode_responses=True,
        )
    except Exception:
        return None


def acquire_distributed_build_lock(ttl_seconds: int = 1800) -> bool | None:
    """Attempts atomic Redis NX lock acquisition. Returns None if Redis is unavailable."""
    client = get_catalog_lock_redis_client()
    if client is None:
        return None
    try:
        return bool(client.set(BUILD_LOCK_KEY, "locked", nx=True, ex=ttl_seconds))
    except Exception as e:
        logger.warning(f"Redis lock failed ({e}); caller may fall back to local lock.")
        return None


def release_distributed_build_lock() -> None:
    """Releases the distributed catalog build lock from Redis if reachable."""
    client = get_catalog_lock_redis_client()
    if client is None:
        return
    try:
        client.delete(BUILD_LOCK_KEY)
    except Exception:
        pass
