"""Redis synchronous progress cache and asynchronous SSE Pub/Sub repository."""

import json
import logging
import threading
import time
from typing import Any

from backend.core import config

logger = logging.getLogger("matchops.repositories.job_progress")

JOB_PROGRESS_PREFIX = "matchops:job:progress:"
JOB_PROGRESS_TTL = 3600

# In-memory mirror for degraded single-node fallback when Redis is unreachable
_job_progress: dict[str, float] = {}
_job_eta: dict[str, int | None] = {}
_job_stage: dict[str, str] = {}

_redis_lock = threading.Lock()
_redis_client = None
_redis_target: str | None = None
_redis_down_until: float = 0.0

_async_redis_pool = None


def get_job_events_channel() -> str:
    """Returns the Redis Pub/Sub channel name for live job state transitions."""
    return "matchops:jobs:events"


def mark_redis_down() -> None:
    """Trips a short 5-second circuit breaker after a Redis progress error."""
    global _redis_down_until
    _redis_down_until = time.monotonic() + 5.0


def get_progress_redis_client():
    """Returns a cached sync Redis client with a 5-second outage circuit breaker."""
    global _redis_client, _redis_target
    if time.monotonic() < _redis_down_until:
        return None
    url = config.REDIS_URL
    with _redis_lock:
        if _redis_client is None or _redis_target != url:
            try:
                import redis

                _redis_client = redis.Redis.from_url(
                    url, socket_connect_timeout=0.5, socket_timeout=0.5, decode_responses=True
                )
                _redis_target = url
            except Exception:
                mark_redis_down()
                return None
        return _redis_client


def set_job_progress(
    job_id: str,
    pct: float,
    eta: int | None = None,
    stage: str | None = None,
) -> None:
    """Persists live job progress, ETA, and stage into Redis and local fallback memory."""
    jid = str(job_id)
    _job_progress[jid] = float(pct)
    _job_eta[jid] = eta
    if stage is not None:
        _job_stage[jid] = stage

    client = get_progress_redis_client()
    if client is None:
        return
    fields: dict[str, str] = {"pct": str(float(pct)), "eta": "" if eta is None else str(int(eta))}
    if stage is not None:
        fields["stage"] = stage
    try:
        key = f"{JOB_PROGRESS_PREFIX}{jid}"
        pipe = client.pipeline()
        pipe.hset(key, mapping=fields)
        pipe.expire(key, JOB_PROGRESS_TTL)
        pipe.execute()
    except Exception as e:
        logger.debug(f"Redis progress write skipped for job {jid}: {e}")
        mark_redis_down()


def fetch_job_progress_state(job_id: str) -> tuple[float, int | None]:
    """Reads live progress and ETA from Redis (multi-replica safe), falling back to memory."""
    jid = str(job_id)
    client = get_progress_redis_client()
    if client is not None:
        try:
            pct_raw, eta_raw = client.hmget(f"{JOB_PROGRESS_PREFIX}{jid}", "pct", "eta")
            if pct_raw is not None:
                eta_val = int(eta_raw) if eta_raw not in (None, "") else None
                return float(pct_raw), eta_val
        except Exception:
            mark_redis_down()
    return _job_progress.get(jid, 0.0), _job_eta.get(jid)


def fetch_bulk_job_progress_states(job_ids: list[str]) -> dict[str, tuple[float, int | None]]:
    """Batch-fetches live progress and ETA for multiple jobs in one Redis pipeline."""
    if not job_ids:
        return {}
    jids = [str(j) for j in job_ids]
    out: dict[str, tuple[float, int | None]] = {}
    client = get_progress_redis_client()
    if client is not None:
        try:
            pipe = client.pipeline()
            for jid in jids:
                pipe.hmget(f"{JOB_PROGRESS_PREFIX}{jid}", "pct", "eta")
            raw_list = pipe.execute()
            for jid, res in zip(jids, raw_list):
                if res and res[0] is not None:
                    eta_val = int(res[1]) if res[1] not in (None, "") else None
                    out[jid] = (float(res[0]), eta_val)
        except Exception:
            mark_redis_down()
    for jid in jids:
        if jid not in out:
            out[jid] = (_job_progress.get(jid, 0.0), _job_eta.get(jid))
    return out


def fetch_cached_job_stage(job_id: str) -> tuple[str | None, bool]:
    """Returns (last_stage, has_in_memory_stage) from memory or Redis."""
    jid = str(job_id)
    in_mem_stage = _job_stage.get(jid)
    if in_mem_stage is not None:
        return in_mem_stage, True

    last_stage = None
    client = get_progress_redis_client()
    if client is not None:
        try:
            last_stage = client.hget(f"{JOB_PROGRESS_PREFIX}{jid}", "stage")
        except Exception:
            mark_redis_down()
    return last_stage, False


def get_in_memory_progress(job_id: str) -> float:
    """Returns locally cached progress percentage for terminal failed/cancelled jobs."""
    return _job_progress.get(str(job_id), 0.0)


def publish_job_event(event: dict[str, Any]) -> None:
    """Broadcasts a live job state transition on the Redis Pub/Sub channel."""
    if not event:
        return
    client = get_progress_redis_client()
    if client is None:
        return
    try:
        client.publish(get_job_events_channel(), json.dumps(event, default=str))
    except Exception as e:
        logger.debug(f"Redis event publish skipped for job {event.get('id')}: {e}")
        mark_redis_down()


def clear_job_progress(job_id: str) -> None:
    """Evicts terminal job progress, ETA, and stage entries from Redis and memory."""
    jid = str(job_id)
    _job_progress.pop(jid, None)
    _job_eta.pop(jid, None)
    _job_stage.pop(jid, None)
    client = get_progress_redis_client()
    if client is not None:
        try:
            client.delete(f"{JOB_PROGRESS_PREFIX}{jid}")
        except Exception:
            mark_redis_down()


def get_async_redis_pool():
    """Returns a process-wide async Redis ConnectionPool for SSE event streaming."""
    global _async_redis_pool
    if _async_redis_pool is None:
        import redis.asyncio as aioredis

        _async_redis_pool = aioredis.ConnectionPool.from_url(
            config.REDIS_URL,
            max_connections=50,
            socket_connect_timeout=2.0,
            decode_responses=True,
        )
    return _async_redis_pool


async def close_async_redis_pool() -> None:
    """Disconnects the shared async Redis ConnectionPool during shutdown."""
    global _async_redis_pool
    if _async_redis_pool is not None:
        try:
            await _async_redis_pool.disconnect()
        except Exception:
            pass
        _async_redis_pool = None


async def open_job_pubsub():
    """Opens and subscribes an async Redis Pub/Sub connection for job state streams."""
    import redis.asyncio as aioredis

    pool = get_async_redis_pool()
    client = aioredis.Redis(connection_pool=pool)
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    try:
        await pubsub.subscribe(get_job_events_channel())
    except Exception as e:
        await pubsub.aclose()
        await client.aclose()
        raise ConnectionError(f"Job event stream unavailable: {e}") from e
    return client, pubsub
