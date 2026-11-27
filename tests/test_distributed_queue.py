"""Redis distributed job queue - priority tiers, concurrency, retries/DLQ, lease recovery."""

import json
import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from backend.grpc.callback_servicer import start_backend_grpc_server
from engine.core import config
from engine.grpc.callback_client import register_engine_callbacks
from engine.pipelines import queue_pipeline as task_queue
from engine.pipelines import worker_pipeline as worker_runner  # registers failure notifier
from engine.pipelines.queue_pipeline import (
    BULK,
    DLQ,
    INFLIGHT,
    REALTIME,
    TASKS,
    TaskQueue,
    run_worker,
)

REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
register_engine_callbacks()
_redis_available: bool | None = None


@pytest.fixture
def redis_client():
    global _redis_available
    import redis

    if _redis_available is False:
        pytest.skip(f"Redis not available at {REDIS_URL}")

    client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1, decode_responses=True)
    if _redis_available is None:
        try:
            client.ping()
            _redis_available = True
        except Exception:
            _redis_available = False
            pytest.skip(f"Redis not available at {REDIS_URL}")

    client.flushdb()  # dedicated test DB
    yield client
    client.flushdb()


@pytest.fixture
def queue(redis_client):
    return TaskQueue(redis_client, max_attempts=3, lease_seconds=30, backoff_seconds=0)


@pytest.fixture
def backend_grpc():
    srv = start_backend_grpc_server(port=0, host="127.0.0.1")
    try:
        yield f"127.0.0.1:{srv._bound_port}"
    finally:
        srv.stop(grace=0)


def _payload(
    job_id: str,
    n_skus: int = 1,
    task: str = "pipeline",
    backend_grpc_url: str | None = None,
) -> dict:
    p = {"job_id": job_id, "task": task, "domain": "market", "skus": [{"name": "x"}] * n_skus}
    if backend_grpc_url:
        p["backend_grpc_url"] = backend_grpc_url
    return p


def _drain(queue: TaskQueue, handler, workers: int = 1, until=lambda: False, timeout=10.0):
    """Runs `workers` consumer threads until `until()` is true, then stops them."""
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=run_worker,
            args=(queue, handler),
            kwargs={"stop": stop, "poll_interval": 0.01, "reclaim_interval": 0.05},
            daemon=True,
        )
        for _ in range(workers)
    ]
    for t in threads:
        t.start()
    deadline = time.time() + timeout
    while not until() and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    assert until(), "workers did not finish in time"


# ---------------------------------------------------------------- priority tiers


def test_routing_by_batch_size(queue, redis_client):
    small = queue.enqueue(_payload("small", n_skus=config.TASK_REALTIME_MAX_SKUS))
    large = queue.enqueue(_payload("large", n_skus=config.TASK_REALTIME_MAX_SKUS + 1))
    assert redis_client.lrange(REALTIME, 0, -1) == [small]
    assert redis_client.lrange(BULK, 0, -1) == [large]


def test_realtime_is_always_consumed_before_waiting_bulk(queue):
    for i in range(5):
        queue.enqueue(_payload(f"bulk-{i}"), queue=BULK)
    for i in range(3):
        queue.enqueue(_payload(f"rt-{i}"), queue=REALTIME)

    order = []
    first = queue.claim()
    order.append(first["job_id"])
    queue.ack(first["task_id"])
    # A realtime task arriving while bulk is waiting still jumps ahead
    queue.enqueue(_payload("rt-late"), queue=REALTIME)
    while (task := queue.claim()) is not None:
        order.append(task["job_id"])
        queue.ack(task["task_id"])

    assert order == ["rt-0", "rt-1", "rt-2", "rt-late"] + [f"bulk-{i}" for i in range(5)]


def test_empty_queue_claims_nothing(queue):
    assert queue.claim() is None


# ---------------------------------------------------------------- horizontal concurrency


def test_concurrent_workers_process_every_job_exactly_once(queue, redis_client):
    n_jobs = 200
    for i in range(n_jobs):
        queue.enqueue(_payload(f"job-{i}"), queue=REALTIME if i % 3 == 0 else BULK)

    seen, lock = [], threading.Lock()

    def handler(payload):
        time.sleep(0.001)
        with lock:
            seen.append(payload["job_id"])

    _drain(queue, handler, workers=8, until=lambda: len(seen) >= n_jobs)

    assert len(seen) == n_jobs
    assert sorted(seen) == sorted(f"job-{i}" for i in range(n_jobs))  # no duplicates, none lost
    assert redis_client.hlen(TASKS) == 0 and redis_client.zcard(INFLIGHT) == 0


# ---------------------------------------------------------------- retries & dead-letter queue


def test_task_failing_twice_succeeds_on_third_attempt(queue, redis_client):
    queue.enqueue(_payload("flaky"))
    attempts = []

    def handler(payload):
        attempts.append(time.time())
        if len(attempts) < 3:
            raise ConnectionError("transient")

    _drain(queue, handler, until=lambda: redis_client.hlen(TASKS) == 0)

    assert len(attempts) == 3
    assert redis_client.llen(DLQ) == 0


def test_retry_backoff_is_exponential(redis_client):
    q = TaskQueue(redis_client, max_attempts=3, backoff_seconds=10)
    q.enqueue(_payload("slow"))
    task = q.claim()
    before = time.time()
    assert q.fail(task, "boom") == "retry"
    ready_at = redis_client.zscore(task_queue.DELAYED[REALTIME], task["task_id"])
    assert 10 <= ready_at - before < 11
    assert q.claim() is None  # not due yet

    redis_client.zadd(task_queue.DELAYED[REALTIME], {task["task_id"]: 0})  # fast-forward
    task = q.claim()
    assert task["attempts"] == 1
    before = time.time()
    q.fail(task, "boom")
    assert 20 <= redis_client.zscore(task_queue.DELAYED[REALTIME], task["task_id"]) - before < 21


@pytest.fixture
def pg_job(pg):
    job_id = f"test-q-{uuid.uuid4().hex[:10]}"
    with pg.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO jobs (id, batch_id, type, status, current_stage, started_at, domain) "
                "VALUES (:id, :id, 'pipeline', 'queued', 'queued', now(), 'market')"
            ),
            {"id": job_id},
        )
    yield job_id
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job_id})


def _job_row(pg, job_id):
    with pg.connect() as conn:
        return (
            conn.execute(text("SELECT * FROM jobs WHERE id = :id"), {"id": job_id})
            .mappings()
            .fetchone()
        )


def test_exhausted_retries_land_in_dlq_and_fail_the_job(
    queue, redis_client, pg, pg_job, backend_grpc
):
    task_id = queue.enqueue(_payload(pg_job, backend_grpc_url=backend_grpc))
    attempts, dead = [], []

    def handler(payload):
        attempts.append(1)
        raise RuntimeError("model crashed")

    stop = threading.Event()
    worker = threading.Thread(
        target=run_worker,
        args=(queue, handler, dead.append),
        kwargs={"stop": stop, "poll_interval": 0.01},
        daemon=True,
    )
    worker.start()
    deadline = time.time() + 10
    while not dead and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    worker.join(timeout=5)

    assert len(attempts) == config.TASK_MAX_RETRIES == 3
    [entry] = queue.dlq_items()
    assert entry["task_id"] == task_id
    assert entry["job_id"] == pg_job and entry["batch_id"] == pg_job
    assert entry["error"] == "model crashed" and entry["attempts"] == 3
    assert entry["failed_at"] and entry["payload"]["job_id"] == pg_job
    assert redis_client.hlen(TASKS) == 0  # no longer tracked as live
    assert dead and dead[0]["job_id"] == pg_job  # on_dead hook ran

    row = _job_row(pg, pg_job)
    assert row["status"] == "failed"
    assert "model crashed" in row["error_message"]


def test_upload_jobs_are_never_retried(queue, monkeypatch):
    monkeypatch.setattr(TaskQueue, "_mark_job_failed", staticmethod(lambda task, error: None))
    queue.enqueue(_payload("upload-1", task="upload"))
    task = queue.claim()
    assert task["max_attempts"] == 1
    assert queue.fail(task, "portal error") == "dead"
    assert queue.dlq_items()[0]["attempts"] == 1


def test_dlq_replay_endpoint_requeues_and_resets_job(
    queue, redis_client, pg, pg_job, backend_grpc, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.endpoints import queue_router as queue_api
    from backend.repositories import task_queue_repository
    from backend.repositories.task_queue_repository import TaskQueue as BackendTaskQueue

    backend_queue = BackendTaskQueue(redis_client)
    monkeypatch.setattr(task_queue_repository, "get_task_queue", lambda: backend_queue)
    app = FastAPI()
    app.include_router(queue_api.router)
    client = TestClient(app)

    queue.enqueue(_payload(pg_job, backend_grpc_url=backend_grpc), queue=BULK)
    for _ in range(3):  # backoff is 0, so each retry is claimable immediately
        queue.fail(queue.claim(), "boom")
    assert _job_row(pg, pg_job)["status"] == "failed"

    listed = client.get("/queue/dlq").json()
    assert [e["job_id"] for e in listed] == [pg_job]

    resp = client.post(f"/queue/dlq/{listed[0]['task_id']}/replay")
    assert resp.status_code == 200
    assert redis_client.llen(DLQ) == 0
    replayed = queue.claim()
    assert replayed["job_id"] == pg_job and replayed["attempts"] == 0
    assert replayed["queue"] == BULK  # original tier kept
    row = _job_row(pg, pg_job)
    assert row["status"] == "queued" and row["error_message"] is None

    assert client.post("/queue/dlq/does-not-exist/replay").status_code == 404


# ---------------------------------------------------------------- crashed worker recovery


def test_crashed_worker_lease_is_reclaimed_and_job_completes(redis_client):
    q = TaskQueue(redis_client, max_attempts=3, lease_seconds=0.3, backoff_seconds=0)
    q.enqueue(_payload("orphan"))

    crashed = q.claim()  # worker A takes the task, then dies without ack/fail/renew
    assert redis_client.zscore(INFLIGHT, crashed["task_id"]) is not None
    assert q.claim() is None  # still leased: nobody else may take it
    time.sleep(0.4)

    done = []
    _drain(q, lambda p: done.append(p["job_id"]), until=lambda: done == ["orphan"])

    assert redis_client.hlen(TASKS) == 0 and redis_client.zcard(INFLIGHT) == 0


def test_repeatedly_crashing_job_goes_to_dlq(redis_client, monkeypatch):
    monkeypatch.setattr(TaskQueue, "_mark_job_failed", staticmethod(lambda task, error: None))
    q = TaskQueue(redis_client, max_attempts=2, lease_seconds=0.1, backoff_seconds=0)
    q.enqueue(_payload("poison"))
    for _ in range(2):
        assert q.claim() is not None  # claimed, then the worker "crashes"
        time.sleep(0.15)
        dead = q.reclaim_expired()
    assert [t["job_id"] for t in dead] == ["poison"]
    assert "lease expired" in q.dlq_items()[0]["error"]


def test_lease_heartbeat_keeps_long_jobs_from_being_reclaimed(redis_client):
    q = TaskQueue(redis_client, max_attempts=3, lease_seconds=0.3, backoff_seconds=0)
    q.enqueue(_payload("long"))
    runs = []

    def slow_handler(payload):
        runs.append(1)
        time.sleep(1.0)  # > 3 lease periods; the heartbeat renews every lease/3

    _drain(q, slow_handler, workers=2, until=lambda: redis_client.hlen(TASKS) == 0)
    assert runs == [1]  # the second worker never stole it


def test_completing_after_lease_was_lost_does_not_duplicate_bookkeeping(redis_client):
    q = TaskQueue(redis_client, max_attempts=3, lease_seconds=30, backoff_seconds=0)
    q.enqueue(_payload("stale"))
    task = q.claim()
    redis_client.zrem(INFLIGHT, task["task_id"])  # someone reclaimed it
    assert q.fail(task, "late failure") == "lost"
    assert redis_client.llen(DLQ) == 0


# ---------------------------------------------------------------- graceful fallback


@pytest.fixture
def unreachable_redis(monkeypatch):
    from backend.repositories import task_queue_repository as backend_task_queue

    monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:1/0")
    monkeypatch.setattr(config, "TASK_QUEUE_ENABLED", True)
    monkeypatch.setattr(task_queue, "_queue", None)
    monkeypatch.setattr(task_queue, "_queue_url", None)
    monkeypatch.setattr(backend_task_queue, "_queue", None)
    monkeypatch.setattr(backend_task_queue, "_queue_url", None)
    yield
    task_queue._queue = None
    task_queue._queue_url = None
    backend_task_queue._queue = None
    backend_task_queue._queue_url = None


def _fake_grpc(monkeypatch, module):
    dispatched = []

    def dispatch_batch(req, metadata=None, timeout=None):
        dispatched.append((req, metadata))
        return SimpleNamespace(status="accepted", job_id=req.job_id)

    fake_stub = SimpleNamespace(DispatchBatchJob=dispatch_batch)
    monkeypatch.setattr(module, "get_engine_stub", lambda: fake_stub)
    return dispatched


def test_gateway_falls_back_to_grpc_dispatch_when_redis_unreachable(unreachable_redis, monkeypatch):
    from backend.grpc import engine_grpc_client
    from backend.schemas.sku_schema import BaseRequest, SKUItem
    from backend.services import job_service

    dispatched = _fake_grpc(monkeypatch, engine_grpc_client)
    req = BaseRequest(skus=[SKUItem(name="Anchor Butter")], domain="market", callback_url="")

    result = job_service.dispatch_batch_payload("fallback-1", req, "pipeline")

    assert result == {"status": "accepted", "job_id": "fallback-1"}
    assert len(dispatched) == 1
    assert dispatched[0][0].job_id == "fallback-1"
    assert dispatched[0][0].skus[0].name == "Anchor Butter"


def test_gateway_uses_grpc_when_queue_disabled(monkeypatch):
    from backend.grpc import engine_grpc_client
    from backend.schemas.sku_schema import BaseRequest, SKUItem
    from backend.services import job_service

    monkeypatch.setattr(config, "TASK_QUEUE_ENABLED", False)
    dispatched = _fake_grpc(monkeypatch, engine_grpc_client)
    req = BaseRequest(skus=[SKUItem(name="x")], domain="market", callback_url="")
    job_service.dispatch_batch_payload("legacy-1", req, "pipeline")
    assert len(dispatched) == 1
    assert task_queue.get_task_queue() is None


def test_gateway_enqueues_to_redis_when_available(redis_client, monkeypatch):
    from backend.grpc import engine_grpc_client
    from backend.repositories import task_queue_repository
    from backend.repositories.task_queue_repository import TaskQueue as BackendTaskQueue
    from backend.schemas.sku_schema import BaseRequest, SKUItem
    from backend.services import job_service

    backend_queue = BackendTaskQueue(redis_client)
    monkeypatch.setattr(task_queue_repository, "get_task_queue", lambda: backend_queue)
    dispatched = _fake_grpc(monkeypatch, engine_grpc_client)
    req = BaseRequest(skus=[SKUItem(name="x")], domain="market", callback_url="")

    result = job_service.dispatch_batch_payload("queued-1", req, "pipeline")

    assert result["status"] == "queued" and dispatched == []
    assert json.loads(redis_client.hget(TASKS, result["task_id"]))["job_id"] == "queued-1"


def test_worker_loop_survives_redis_outage(unreachable_redis):
    q = task_queue.get_task_queue()
    stop = threading.Event()
    worker = threading.Thread(target=run_worker, args=(q, lambda p: None), kwargs={"stop": stop})
    worker.start()
    time.sleep(0.3)
    assert worker.is_alive()  # keeps retrying instead of crashing the engine
    stop.set()
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_cancellation_reaches_queued_job_on_any_worker(queue, monkeypatch):
    monkeypatch.setattr(worker_runner, "get_task_queue", lambda: queue)
    executed = []
    monkeypatch.setattr(worker_runner, "_execute_job", lambda p: executed.append(p["job_id"]))

    queue.request_cancel("cancel-me")  # e.g. sent through a different gateway/engine
    worker_runner._run_queued_job(_payload("cancel-me"))
    worker_runner._run_queued_job(_payload("keep-me"))

    assert executed == ["keep-me"]


def test_queued_job_errors_propagate_for_retry(monkeypatch):
    def boom(payload):
        raise ConnectionError("db blip")

    monkeypatch.setattr(worker_runner, "_execute_job", boom)
    monkeypatch.setattr(worker_runner, "get_task_queue", lambda: None)
    with pytest.raises(ConnectionError):
        worker_runner._run_queued_job(_payload("retry-me"))


def test_engine_get_task_queue_is_thread_safe_and_recreates_on_url_change(monkeypatch):
    import redis

    created_urls = []

    def fake_from_url(url, **_kwargs):
        created_urls.append(url)
        return SimpleNamespace(url=url, register_script=lambda _script: lambda **_kw: None)

    monkeypatch.setattr(config, "TASK_QUEUE_ENABLED", True)
    monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:6379/14")
    monkeypatch.setattr(task_queue, "_queue", None)
    monkeypatch.setattr(task_queue, "_queue_url", None)
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(fake_from_url))

    try:
        results = []
        barrier = threading.Barrier(4)

        def call_get():
            barrier.wait(timeout=2)
            results.append(task_queue.get_task_queue())

        threads = [threading.Thread(target=call_get) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(created_urls) == 1
        assert len({id(q) for q in results}) == 1

        monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:6379/15")
        q2 = task_queue.get_task_queue()
        assert q2 is not results[0]
        assert created_urls == ["redis://localhost:6379/14", "redis://localhost:6379/15"]
    finally:
        task_queue._queue = None
        task_queue._queue_url = None


def test_start_queue_worker_is_idempotent_and_replaces_only_dead_workers(monkeypatch):
    class _FakeThread:
        def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None):
            self.name = name
            self._alive = True

        def start(self):
            return None

        def is_alive(self):
            return self._alive

    spawned = []

    def make_thread(*args, **kwargs):
        t = _FakeThread(*args, **kwargs)
        spawned.append(t)
        return t

    monkeypatch.setattr(worker_runner, "get_task_queue", lambda: SimpleNamespace())
    monkeypatch.setattr(worker_runner, "_queue_workers", [])
    monkeypatch.setattr(worker_runner.threading, "Thread", make_thread)

    try:
        worker_runner.start_queue_worker()
        assert [t.name for t in spawned] == ["queue-worker-0", "queue-worker-1"]

        # Both alive: reload must not spawn duplicates
        worker_runner.start_queue_worker()
        assert len(spawned) == 2

        # Only worker-1 dies: only slot 1 is respawned
        spawned[1]._alive = False
        worker_runner.start_queue_worker()
        assert [t.name for t in spawned] == ["queue-worker-0", "queue-worker-1", "queue-worker-1"]
    finally:
        worker_runner._queue_workers = []
        worker_runner._worker_stop.clear()


def test_request_cancel_purges_queued_inflight_and_delayed_tasks(queue, redis_client):
    from backend.repositories.task_queue_repository import TaskQueue as BackendTaskQueue

    # 1. Queued bulk task + delayed task for same job_id, plus unrelated job
    t_bulk = queue.enqueue(_payload("job-cancel-1", n_skus=config.TASK_REALTIME_MAX_SKUS + 5))
    t_inflight_owner = queue.enqueue(_payload("job-cancel-1"))
    claimed = queue.claim()
    assert claimed["task_id"] == t_inflight_owner
    assert queue.fail(claimed, "retry") == "retry"
    t_keep = queue.enqueue(_payload("job-keep-1"))

    assert "job-cancel-1" in queue.tracked_job_ids()
    assert redis_client.zscore(task_queue.DELAYED[REALTIME], t_inflight_owner) is not None

    # Cancel via backend TaskQueue (purges both queued and delayed tasks and sets cancel key)
    BackendTaskQueue(redis_client).request_cancel("job-cancel-1")
    assert queue.is_cancelled("job-cancel-1") is True
    assert queue.tracked_job_ids() == {"job-keep-1"}
    assert redis_client.lrange(BULK, 0, -1) == []
    assert redis_client.zscore(task_queue.DELAYED[REALTIME], t_inflight_owner) is None
    assert redis_client.hget(TASKS, t_bulk) is None
    assert redis_client.hget(TASKS, t_inflight_owner) is None

    # 2. In-flight task cancelled via engine TaskQueue is also purged from INFLIGHT
    claimed_keep = queue.claim()
    assert claimed_keep["task_id"] == t_keep
    queue.request_cancel("job-keep-1")
    assert queue.is_cancelled("job-keep-1") is True
    assert redis_client.zscore(INFLIGHT, t_keep) is None
    assert redis_client.hlen(TASKS) == 0
