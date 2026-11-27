"""Redis distributed task queue repository for job dispatch, cancellation, and DLQ management."""

import json
import logging
import sys
import threading
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from backend.core import config

logger = logging.getLogger("matchops.repositories.task_queue")


@runtime_checkable
class TaskQueueProtocol(Protocol):
    def enqueue_task(self, payload: dict[str, Any], queue: str | None = None) -> str: ...
    def request_cancel(self, job_id: str) -> None: ...
    def is_cancelled(self, job_id: str) -> bool: ...
    def fetch_dlq_items(self, limit: int = 100) -> list[dict]: ...
    def count_dlq_items(self) -> int: ...
    def replay_dlq_task(self, task_id: str) -> str | None: ...
    def drop_dlq_task(self, task_id: str) -> bool: ...
    def list_tracked_job_ids(self) -> set[str]: ...
    def get_queue_stats(self) -> dict[str, int]: ...
    def ping_broker(self) -> bool: ...


REALTIME = "matchops:queue:realtime"
BULK = "matchops:queue:bulk"
DLQ = "matchops:queue:dlq"
INFLIGHT = "matchops:queue:inflight"
TASKS = "matchops:queue:tasks"
TASK_JOBS = "matchops:queue:task_jobs"
DELAYED = {REALTIME: f"{REALTIME}:delayed", BULK: f"{BULK}:delayed"}
CANCEL_PREFIX = "matchops:job:cancelled:"


def get_current_timestamp() -> float:
    """Returns current epoch timestamp in seconds."""
    return time.time()


class TaskQueue:
    """Redis-backed priority queue client for Gateway-side enqueueing and DLQ operations."""

    def __init__(
        self,
        client: Any,
        max_attempts: int = 3,
        realtime_max_skus: int = 50,
    ):
        self.r = client
        self.max_attempts = max_attempts
        self.realtime_max_skus = realtime_max_skus

    def enqueue_task(self, payload: dict[str, Any], queue: str | None = None) -> str:
        """Enqueues a job payload onto the realtime or bulk Redis tier."""
        if queue is None:
            queue = REALTIME if len(payload.get("skus", [])) <= self.realtime_max_skus else BULK
        task = {
            "task_id": uuid.uuid4().hex,
            "job_id": str(payload.get("job_id", "")),
            "queue": queue,
            "attempts": 0,
            "max_attempts": 1 if payload.get("task") == "upload" else self.max_attempts,
            "enqueued_at": get_current_timestamp(),
            "payload": payload,
        }
        pipe = self.r.pipeline()
        pipe.hset(TASKS, task["task_id"], json.dumps(task))
        pipe.hset(TASK_JOBS, task["task_id"], task["job_id"])
        pipe.rpush(queue, task["task_id"])
        pipe.execute()
        logger.info(
            f"[QUEUE] Enqueued job {task['job_id']} as {task['task_id']} on {queue} "
            f"(trace {payload.get('trace_id', '-')})."
        )
        return task["task_id"]

    def request_cancel(self, job_id: str) -> None:
        """Marks a job cancelled in Redis and purges any pending/delayed tasks for it."""
        jid = str(job_id)
        self.r.set(f"{CANCEL_PREFIX}{jid}", "1", ex=86400)
        try:
            task_jobs = self.r.hgetall(TASK_JOBS)
            if isinstance(task_jobs, dict):
                matching_ids = [
                    (tid.decode() if isinstance(tid, bytes) else str(tid))
                    for tid, val in task_jobs.items()
                    if (val.decode() if isinstance(val, bytes) else str(val)) == jid
                ]
                if matching_ids:
                    pipe = self.r.pipeline()
                    for tid in matching_ids:
                        pipe.zrem(INFLIGHT, tid)
                        pipe.lrem(REALTIME, 0, tid)
                        pipe.lrem(BULK, 0, tid)
                        pipe.zrem(DELAYED[REALTIME], tid)
                        pipe.zrem(DELAYED[BULK], tid)
                        pipe.hdel(TASKS, tid)
                        pipe.hdel(TASK_JOBS, tid)
                    pipe.execute()
        except Exception as e:
            logger.debug(f"[QUEUE] Could not purge tasks for job {jid}: {e}")

    def is_cancelled(self, job_id: str) -> bool:
        """Checks whether cancellation has been flagged for the given job ID."""
        return bool(self.r.exists(f"{CANCEL_PREFIX}{job_id}"))

    def fetch_dlq_items(self, limit: int = 100) -> list[dict]:
        """Returns up to `limit` dead-letter queue entries."""
        return [json.loads(raw) for raw in self.r.lrange(DLQ, 0, limit - 1)]

    def count_dlq_items(self) -> int:
        """Returns current length of the dead-letter queue."""
        return int(self.r.llen(DLQ) or 0)

    def replay_dlq_task(self, task_id: str) -> str | None:
        """Moves a dead-letter task back onto its original queue with a new task_id."""
        for raw in self.r.lrange(DLQ, 0, -1):
            entry = json.loads(raw)
            if entry["task_id"] == task_id:
                if self.r.lrem(DLQ, 1, raw):
                    return self.enqueue_task(entry["payload"], queue=entry["queue"])
        return None

    def drop_dlq_task(self, task_id: str) -> bool:
        """Removes a dead-letter task permanently from the DLQ."""
        for raw in self.r.lrange(DLQ, 0, -1):
            entry = json.loads(raw)
            if entry["task_id"] == task_id:
                return bool(self.r.lrem(DLQ, 1, raw))
        return False

    def list_tracked_job_ids(self) -> set[str]:
        """Returns the set of job IDs currently tracked in active/delayed Redis queues."""
        pipe = self.r.pipeline()
        pipe.hlen(TASKS)
        pipe.hvals(TASK_JOBS)
        n_tasks, job_ids = pipe.execute()
        if n_tasks == 0:
            return set()
        if len(job_ids) >= n_tasks:
            return {j.decode() if isinstance(j, bytes) else str(j) for j in job_ids}
        return {json.loads(raw)["job_id"] for raw in self.r.hvals(TASKS)}

    def get_queue_stats(self) -> dict[str, int]:
        """Returns current queue depth across realtime, bulk, delayed, inflight, and DLQ."""
        pipe = self.r.pipeline()
        pipe.llen(REALTIME)
        pipe.llen(BULK)
        pipe.zcard(DELAYED[REALTIME])
        pipe.zcard(DELAYED[BULK])
        pipe.zcard(INFLIGHT)
        pipe.llen(DLQ)
        rt, bulk, d_rt, d_bulk, inflight, dlq = pipe.execute()
        return {
            "realtime": rt,
            "bulk": bulk,
            "delayed": d_rt + d_bulk,
            "inflight": inflight,
            "dlq": dlq,
        }

    def ping_broker(self) -> bool:
        """Pings the underlying Redis broker."""
        try:
            return bool(self.r.ping())
        except Exception:
            return False


_queue: TaskQueue | None = None
_queue_url: str | None = None
_client_lock = threading.Lock()


def get_task_queue() -> TaskQueue | None:
    """Returns a cached TaskQueue client connected to Redis, or None if disabled/unreachable."""
    global _queue, _queue_url

    task_queue_enabled = getattr(config, "TASK_QUEUE_ENABLED", True)
    if "engine.core.config" in sys.modules:
        task_queue_enabled = getattr(
            sys.modules["engine.core.config"], "TASK_QUEUE_ENABLED", task_queue_enabled
        )
    if not task_queue_enabled:
        return None

    redis_url = getattr(config, "REDIS_URL", None)
    if "engine.core.config" in sys.modules and hasattr(
        sys.modules["engine.core.config"], "REDIS_URL"
    ):
        redis_url = sys.modules["engine.core.config"].REDIS_URL
    redis_url = redis_url or "redis://localhost:6379/0"

    with _client_lock:
        if _queue is not None:
            if _queue_url != redis_url or not _queue.ping_broker():
                _queue = None
                _queue_url = None

        if _queue is None:
            try:
                import redis

                client = redis.Redis.from_url(
                    redis_url,
                    socket_connect_timeout=1,
                    socket_timeout=2,
                    decode_responses=True,
                )
                client.ping()
                _queue = TaskQueue(client)
                _queue_url = redis_url
            except Exception as e:
                logger.warning(f"[QUEUE] Could not connect to Redis TaskQueue ({redis_url}): {e}")
                return None
    return _queue


def ping_redis() -> bool:
    """Checks basic Redis reachability using configured REDIS_URL."""
    if not config.REDIS_URL:
        return True
    try:
        import redis

        client = redis.Redis.from_url(config.REDIS_URL, socket_timeout=1.0)
        return bool(client.ping())
    except Exception:
        return False
