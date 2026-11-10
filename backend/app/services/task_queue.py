"""
SKU MatchOps Backend - Redis Task Queue Client
Enqueues jobs to Redis, manages DLQ replay/drop, and inspects queue depth for the API Gateway.
"""

import json
import logging
import threading
import time
import uuid
from typing import Any

from backend.app.core import config

logger = logging.getLogger("matchops.backend.task_queue")

REALTIME = "matchops:queue:realtime"
BULK = "matchops:queue:bulk"
DLQ = "matchops:queue:dlq"
DLQ_TENANTS = "matchops:queue:dlq_tenants"
INFLIGHT = "matchops:queue:inflight"
TASKS = "matchops:queue:tasks"
TASK_JOBS = "matchops:queue:task_jobs"
DELAYED = {REALTIME: f"{REALTIME}:delayed", BULK: f"{BULK}:delayed"}
CANCEL_PREFIX = "matchops:job:cancelled:"


def _now() -> float:
    return time.time()


class TaskQueue:
    def __init__(
        self,
        client: Any,
        max_attempts: int = 3,
        realtime_max_skus: int = 50,
    ):
        self.r = client
        self.max_attempts = max_attempts
        self.realtime_max_skus = realtime_max_skus

    def enqueue(self, payload: dict[str, Any], queue: str | None = None) -> str:
        """Adds a job payload; small batches go to realtime, large ones to bulk."""
        if queue is None:
            queue = REALTIME if len(payload.get("skus", [])) <= self.realtime_max_skus else BULK
        task = {
            "task_id": uuid.uuid4().hex,
            "job_id": str(payload.get("job_id", "")),
            "queue": queue,
            "attempts": 0,
            "max_attempts": 1 if payload.get("task") == "upload" else self.max_attempts,
            "enqueued_at": _now(),
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
        self.r.set(f"{CANCEL_PREFIX}{job_id}", "1", ex=86400)

    def is_cancelled(self, job_id: str) -> bool:
        return bool(self.r.exists(f"{CANCEL_PREFIX}{job_id}"))

    def dlq_items(self, limit: int = 100) -> list[dict]:
        return [json.loads(raw) for raw in self.r.lrange(DLQ, 0, limit - 1)]

    def dlq_count(self, tenant_id: str = "default") -> int:
        pipe = self.r.pipeline()
        pipe.llen(DLQ)
        pipe.hlen(DLQ_TENANTS)
        pipe.hget(DLQ_TENANTS, tenant_id)
        total_dlq, tracked_tenants, val = pipe.execute()
        if total_dlq == 0:
            if tracked_tenants:
                self.r.delete(DLQ_TENANTS)
            return 0
        if tracked_tenants > 0:
            return max(0, int(val or 0))
        counts: dict[str, int] = {}
        for raw in self.r.lrange(DLQ, 0, -1):
            t = str(json.loads(raw).get("payload", {}).get("tenant_id") or "default")
            counts[t] = counts.get(t, 0) + 1
        if counts:
            self.r.hset(DLQ_TENANTS, mapping=counts)
        return counts.get(tenant_id, 0)

    def replay(self, task_id: str) -> str | None:
        for raw in self.r.lrange(DLQ, 0, -1):
            entry = json.loads(raw)
            if entry["task_id"] == task_id and self.r.lrem(DLQ, 1, raw):
                tenant_id = str(entry["payload"].get("tenant_id") or "default")
                if self.r.hincrby(DLQ_TENANTS, tenant_id, -1) <= 0:
                    self.r.hdel(DLQ_TENANTS, tenant_id)
                return self.enqueue(entry["payload"], queue=entry["queue"])
        return None

    def drop_dlq(self, task_id: str) -> bool:
        for raw in self.r.lrange(DLQ, 0, -1):
            entry = json.loads(raw)
            if entry["task_id"] == task_id and self.r.lrem(DLQ, 1, raw):
                tenant_id = str(entry["payload"].get("tenant_id") or "default")
                if self.r.hincrby(DLQ_TENANTS, tenant_id, -1) <= 0:
                    self.r.hdel(DLQ_TENANTS, tenant_id)
                return True
        return False

    def tracked_job_ids(self) -> set[str]:
        pipe = self.r.pipeline()
        pipe.hlen(TASKS)
        pipe.hvals(TASK_JOBS)
        n_tasks, job_ids = pipe.execute()
        if n_tasks == 0:
            return set()
        if len(job_ids) >= n_tasks:
            return {j.decode() if isinstance(j, bytes) else str(j) for j in job_ids}
        return {json.loads(raw)["job_id"] for raw in self.r.hvals(TASKS)}

    def stats(self) -> dict[str, int]:
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

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


_queue: TaskQueue | None = None
_client_lock = threading.Lock()


def get_task_queue() -> TaskQueue | None:
    global _queue
    import sys

    task_queue_enabled = getattr(config, "TASK_QUEUE_ENABLED", True)
    if "engine.config" in sys.modules:
        task_queue_enabled = getattr(
            sys.modules["engine.config"], "TASK_QUEUE_ENABLED", task_queue_enabled
        )
    if not task_queue_enabled:
        return None

    redis_url = getattr(config, "REDIS_URL", None)
    if "engine.config" in sys.modules and hasattr(sys.modules["engine.config"], "REDIS_URL"):
        redis_url = sys.modules["engine.config"].REDIS_URL
    redis_url = redis_url or "redis://localhost:6379/0"

    with _client_lock:
        if _queue is not None:
            if not _queue.ping():
                _queue = None

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
            except Exception as e:
                logger.warning(f"[QUEUE] Could not connect to Redis TaskQueue ({redis_url}): {e}")
                return None
    return _queue
