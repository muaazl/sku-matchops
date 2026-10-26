"""
SKU MatchOps - Redis-backed distributed job queue.

Any number of Engine workers consume from two priority tiers (realtime is always drained before
bulk). A claimed task is tracked in an in-flight ZSET scored by its lease deadline; the worker
renews the lease while it runs, so a crashed worker's tasks are reclaimed once the lease lapses.
Failures retry with exponential backoff up to TASK_MAX_RETRIES attempts, then the task moves to
the dead-letter queue and its job is marked failed in PostgreSQL.

Delivery is at-least-once (a worker that stalls past its lease may finish a task another worker
also runs); this is safe because result persistence is an idempotent upsert.
"""

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from engine import config

logger = logging.getLogger("matchops.task_queue")

REALTIME = "matchops:queue:realtime"
BULK = "matchops:queue:bulk"
DLQ = "matchops:queue:dlq"
INFLIGHT = "matchops:queue:inflight"  # ZSET task_id -> lease deadline (epoch seconds)
TASKS = "matchops:queue:tasks"  # HASH task_id -> task JSON
DELAYED = {REALTIME: f"{REALTIME}:delayed", BULK: f"{BULK}:delayed"}  # ZSET task_id -> ready_at
CANCEL_PREFIX = "matchops:job:cancelled:"

# Atomically promotes due retries, then pops realtime before bulk and leases the task.
_CLAIM_LUA = """
local pairs_ = {{KEYS[3], KEYS[1]}, {KEYS[4], KEYS[2]}}
for _, p in ipairs(pairs_) do
  local due = redis.call('ZRANGEBYSCORE', p[1], '-inf', ARGV[1])
  for _, id in ipairs(due) do
    redis.call('ZREM', p[1], id)
    redis.call('RPUSH', p[2], id)
  end
end
local id = redis.call('LPOP', KEYS[1])
if not id then id = redis.call('LPOP', KEYS[2]) end
if id then redis.call('ZADD', KEYS[5], ARGV[2], id) end
return id
"""


def _now() -> float:
    return time.time()


class TaskQueue:
    def __init__(
        self,
        client,
        max_attempts: int | None = None,
        lease_seconds: float | None = None,
        backoff_seconds: float | None = None,
        realtime_max_skus: int | None = None,
    ):
        self.r = client
        self.max_attempts = max_attempts or config.TASK_MAX_RETRIES
        self.lease_seconds = lease_seconds or config.TASK_LEASE_SECONDS
        self.backoff_seconds = (
            config.TASK_RETRY_BACKOFF_SECONDS if backoff_seconds is None else backoff_seconds
        )
        self.realtime_max_skus = realtime_max_skus or config.TASK_REALTIME_MAX_SKUS
        self._claim = client.register_script(_CLAIM_LUA)

    # -- producer -------------------------------------------------------------------------------

    def enqueue(self, payload: dict[str, Any], queue: str | None = None) -> str:
        """Adds a job payload; small batches go to realtime, large ones to bulk."""
        if queue is None:
            queue = REALTIME if len(payload.get("skus", [])) <= self.realtime_max_skus else BULK
        task = {
            "task_id": uuid.uuid4().hex,
            "job_id": str(payload.get("job_id", "")),
            "queue": queue,
            "attempts": 0,
            # Uploads drive an external portal and are not idempotent: never auto-retry them
            "max_attempts": 1 if payload.get("task") == "upload" else self.max_attempts,
            "enqueued_at": _now(),
            "payload": payload,
        }
        pipe = self.r.pipeline()
        pipe.hset(TASKS, task["task_id"], json.dumps(task))
        pipe.rpush(queue, task["task_id"])
        pipe.execute()
        logger.info(f"[QUEUE] Enqueued job {task['job_id']} as {task['task_id']} on {queue}.")
        return task["task_id"]

    # -- consumer -------------------------------------------------------------------------------

    def claim(self) -> dict | None:
        """Leases the next task (realtime first). Returns None when both tiers are empty."""
        now = _now()
        task_id = self._claim(
            keys=[REALTIME, BULK, DELAYED[REALTIME], DELAYED[BULK], INFLIGHT],
            args=[now, now + self.lease_seconds],
        )
        if task_id is None:
            return None
        task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
        raw = self.r.hget(TASKS, task_id)
        if raw is None:  # acked/replayed concurrently
            self.r.zrem(INFLIGHT, task_id)
            return None
        task = json.loads(raw)
        task.setdefault("first_claimed_at", now)
        return task

    def renew(self, task_id: str) -> bool:
        """Extends the lease; False if the lease was already reclaimed by another worker."""
        if self.r.zscore(INFLIGHT, task_id) is None:
            return False
        self.r.zadd(INFLIGHT, {task_id: _now() + self.lease_seconds}, xx=True)
        return True

    def ack(self, task_id: str) -> None:
        pipe = self.r.pipeline()
        pipe.zrem(INFLIGHT, task_id)
        pipe.hdel(TASKS, task_id)
        pipe.execute()

    def fail(self, task: dict, error: str) -> str:
        """Records a failed attempt. Returns 'retry', 'dead', or 'lost' (lease no longer ours)."""
        if not self.r.zrem(INFLIGHT, task["task_id"]):
            return "lost"
        return self._retry_or_dead(task, error)

    def reclaim_expired(self) -> list[dict]:
        """Re-queues tasks whose lease lapsed (crashed worker). Returns the ones that went dead."""
        dead = []
        for task_id in self.r.zrangebyscore(INFLIGHT, "-inf", _now()):
            if not self.r.zrem(INFLIGHT, task_id):  # another worker reclaimed it first
                continue
            raw = self.r.hget(TASKS, task_id)
            if raw is None:
                continue
            task = json.loads(raw)
            logger.warning(f"[QUEUE] Lease expired for job {task['job_id']} ({task_id}).")
            if self._retry_or_dead(task, "lease expired (worker crashed or stalled)") == "dead":
                dead.append(task)
        return dead

    def _retry_or_dead(self, task: dict, error: str) -> str:
        task["attempts"] += 1
        task["last_error"] = error
        if task["attempts"] < task["max_attempts"]:
            delay = self.backoff_seconds * 2 ** (task["attempts"] - 1)
            pipe = self.r.pipeline()
            pipe.hset(TASKS, task["task_id"], json.dumps(task))
            pipe.zadd(DELAYED[task["queue"]], {task["task_id"]: _now() + delay})
            pipe.execute()
            logger.warning(
                f"[QUEUE] Job {task['job_id']} attempt {task['attempts']} failed ({error}); "
                f"retrying in {delay:.1f}s."
            )
            return "retry"

        entry = {
            "task_id": task["task_id"],
            "job_id": task["job_id"],
            "batch_id": task["payload"].get("batch_id") or task["job_id"],
            "queue": task["queue"],
            "error": error,
            "attempts": task["attempts"],
            "failed_at": datetime.now(UTC).isoformat(),
            "payload": task["payload"],
        }
        pipe = self.r.pipeline()
        pipe.rpush(DLQ, json.dumps(entry))
        pipe.hdel(TASKS, task["task_id"])
        pipe.execute()
        logger.error(
            f"[QUEUE] Job {task['job_id']} moved to DLQ after {task['attempts']} attempts: {error}"
        )
        self._mark_job_failed(task, error)
        return "dead"

    @staticmethod
    def _mark_job_failed(task: dict, error: str) -> None:
        from engine.pipeline.persistence import record_job_failure

        started = task.get("first_claimed_at") or task.get("enqueued_at") or _now()
        try:
            record_job_failure(
                task["job_id"],
                f"{error} (gave up after {task['attempts']} attempts)",
                round((_now() - started) / 60, 4),
            )
        except Exception as e:
            logger.error(f"[QUEUE] Could not mark job {task['job_id']} failed: {e}")

    # -- dead-letter queue ----------------------------------------------------------------------

    def dlq_items(self, limit: int = 100) -> list[dict]:
        return [json.loads(raw) for raw in self.r.lrange(DLQ, 0, limit - 1)]

    def replay(self, task_id: str) -> str | None:
        """Moves a DLQ entry back onto its queue with a fresh attempt budget. Returns the new
        task id, or None if no such entry."""
        for raw in self.r.lrange(DLQ, 0, -1):
            entry = json.loads(raw)
            if entry["task_id"] == task_id and self.r.lrem(DLQ, 1, raw):
                return self.enqueue(entry["payload"], queue=entry["queue"])
        return None

    # -- bookkeeping ----------------------------------------------------------------------------

    def tracked_job_ids(self) -> set[str]:
        """Job ids still owned by the queue (queued, delayed for retry, or in flight)."""
        return {json.loads(raw)["job_id"] for raw in self.r.hvals(TASKS)}

    def request_cancel(self, job_id: str) -> None:
        self.r.set(f"{CANCEL_PREFIX}{job_id}", "1", ex=86400)

    def is_cancelled(self, job_id: str) -> bool:
        return bool(self.r.exists(f"{CANCEL_PREFIX}{job_id}"))


_queue: TaskQueue | None = None


def get_task_queue() -> TaskQueue | None:
    """Process-wide queue, or None when TASK_QUEUE_ENABLED is off (callers use legacy dispatch).
    Redis connectivity errors surface on first use so callers can fall back."""
    global _queue
    if not config.TASK_QUEUE_ENABLED:
        return None
    if _queue is None:
        import redis

        client = redis.Redis.from_url(
            config.REDIS_URL, socket_connect_timeout=1, socket_timeout=5, decode_responses=True
        )
        _queue = TaskQueue(client)
    return _queue


def run_worker(
    queue: TaskQueue,
    handler: Callable[[dict], None],
    on_dead: Callable[[dict], None] | None = None,
    stop: threading.Event | None = None,
    lock=None,  # a threading.Lock shared with other job runners in this process
    poll_interval: float | None = None,
    reclaim_interval: float = 15.0,
) -> None:
    """
    Consumer loop: reclaim stale leases, claim the next task, run `handler(payload)` while a
    heartbeat renews the lease, then ack or record the failure. `on_dead(task)` runs for tasks
    that exhausted their attempts (after the DLQ move and PostgreSQL update). `lock` serializes
    execution with other job runners in the same process.
    """
    stop = stop or threading.Event()
    poll = config.TASK_POLL_INTERVAL if poll_interval is None else poll_interval
    last_reclaim = 0.0
    backoff = 1.0

    while not stop.is_set():
        try:
            if _now() - last_reclaim >= reclaim_interval:
                last_reclaim = _now()
                for task in queue.reclaim_expired():
                    if on_dead:
                        on_dead(task)

            if lock:
                lock.acquire()
            try:
                task = queue.claim()
                if task is not None:
                    _run_claimed(queue, task, handler, on_dead)
            finally:
                if lock:
                    lock.release()
            backoff = 1.0
            if task is None:
                stop.wait(poll)
        except Exception as e:  # Redis outage: keep the worker alive and retry
            logger.warning(f"[QUEUE] Worker loop error ({e}); retrying in {backoff:.0f}s.")
            stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)


def _run_claimed(queue: TaskQueue, task: dict, handler, on_dead) -> None:
    done = threading.Event()

    def _heartbeat():
        while not done.wait(queue.lease_seconds / 3):
            try:
                if not queue.renew(task["task_id"]):
                    logger.warning(f"[QUEUE] Lost lease on job {task['job_id']}.")
                    return
            except Exception as e:
                logger.debug(f"[QUEUE] Lease renewal failed: {e}")

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    try:
        handler(task["payload"])
    except Exception as e:
        outcome = queue.fail(task, str(e))
        if outcome == "dead" and on_dead:
            task["last_error"] = str(e)
            on_dead(task)
    else:
        queue.ack(task["task_id"])
    finally:
        done.set()
        heartbeat.join(timeout=1)
