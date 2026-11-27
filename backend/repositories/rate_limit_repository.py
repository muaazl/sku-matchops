"""Redis-backed fixed-window counter operations for API rate limiting."""

import logging
import time

from backend.core import config

logger = logging.getLogger("matchops.repositories.rate_limit")

_redis_client = None


def get_rate_limit_redis_client():
    """Returns a cached synchronous Redis client for rate limiting, or None if unavailable."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis

            _redis_client = redis.Redis.from_url(
                config.REDIS_URL,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Could not connect to Redis for rate limiting: {e}")
            return None
    return _redis_client


def increment_rate_limit_counter(
    subject: str,
    endpoint: str,
    window_seconds: int,
) -> tuple[int, int] | None:
    """Increments a fixed-window counter in Redis and returns (count, retry_after) or None on fail-open."""
    client = get_rate_limit_redis_client()
    if client is None:
        return None

    now = int(time.time())
    current_bucket = now // window_seconds
    key = f"matchops:ratelimit:{subject}:{endpoint}:{current_bucket}"
    retry_after = max(1, window_seconds - (now % window_seconds))

    try:
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds * 2)
        results = pipe.execute()
        return int(results[0]), retry_after
    except Exception as err:
        logger.warning(f"Rate limiting Redis check failed ({err}); failing open.")
        return None
