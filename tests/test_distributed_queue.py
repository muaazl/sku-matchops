"""Redis distributed job queue - priority tiers, concurrency, retries/DLQ, lease recovery."""

import json
import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from engine import config
from engine.pipeline import task_queue
from engine.pipeline.task_queue import BULK, DLQ, INFLIGHT, REALTIME, TASKS, TaskQueue, run_worker

REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def redis_client():
    import redis

    client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1, decode_responses=True)
    try:
        client.ping()
    except Exception:
        pytest.skip(f"Redis not available at {REDIS_URL}")
    client.flushdb()  # dedicated test DB
    yield client
    client.flushdb()


@pytest.fixture
def queue(redis_client):
    return TaskQueue(redis_client, max_attempts=3, lease_seconds=30, backoff_seconds=0)


def _payload(job_id: str, n_skus: int = 1, task: str = "pipeline") -> dict:
    return {"job_id": job_id, "task": task, "domain": "market", "skus": [{"name": "x"}] * n_skus}


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


def test_exhausted_retries_land_in_dlq_and_fail_the_job(queue, redis_client, pg, pg_job):
    task_id = queue.enqueue(_payload(pg_job))
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
    assert dead and dead[0]["job_id"] == pg_job  # on_dead hook ran (client notifications)

    row = _job_row(pg, pg_job)
    assert row["status"] == "failed"
    assert "model crashed" in row["error_message"]


def test_upload_jobs_are_never_retried(queue):
    queue.enqueue(_payload("upload-1", task="upload"))
    task = queue.claim()
    assert task["max_attempts"] == 1
    assert queue.fail(task, "portal error") == "dead"
    assert queue.dlq_items()[0]["attempts"] == 1


def test_dlq_replay_endpoint_requeues_and_resets_job(queue, redis_client, pg, pg_job, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.app.api.endpoints import queue as queue_api

    monkeypatch.setattr(queue_api, "get_task_queue", lambda: queue)
    app = FastAPI()
    app.include_router(queue_api.router)
    client = TestClient(app)

    queue.enqueue(_payload(pg_job), queue=BULK)
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
    monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:1/0")
    monkeypatch.setattr(config, "TASK_QUEUE_ENABLED", True)
    monkeypatch.setattr(task_queue, "_queue", None)
    yield
    task_queue._queue = None


def _fake_http(monkeypatch, module):
    posts = []

    def post(url, json=None, timeout=None):
        posts.append((url, json))
        return SimpleNamespace(status_code=202, json=lambda: {"status": "accepted"}, text="")

    monkeypatch.setattr(module._session, "post", post)
    return posts


def test_gateway_falls_back_to_http_dispatch_when_redis_unreachable(unreachable_redis, monkeypatch):
    from backend.app.schemas.models import BaseRequest, SKUItem
    from backend.app.services import engine_client

    posts = _fake_http(monkeypatch, engine_client)
    req = BaseRequest(skus=[SKUItem(name="Anchor Butter")], domain="market", callback_url="")

    result = engine_client.dispatch_batch_job("fallback-1", req, "pipeline")

    assert result == {"status": "accepted"}
    assert posts and posts[0][0].endswith("/engine/process-batch")
    assert posts[0][1]["job_id"] == "fallback-1"


def test_gateway_uses_http_when_queue_disabled(monkeypatch):
    from backend.app.schemas.models import BaseRequest, SKUItem
    from backend.app.services import engine_client

    monkeypatch.setattr(config, "TASK_QUEUE_ENABLED", False)
    posts = _fake_http(monkeypatch, engine_client)
    req = BaseRequest(skus=[SKUItem(name="x")], domain="market", callback_url="")
    engine_client.dispatch_batch_job("legacy-1", req, "pipeline")
    assert len(posts) == 1
    assert task_queue.get_task_queue() is None


def test_gateway_enqueues_to_redis_when_available(queue, redis_client, monkeypatch):
    from backend.app.schemas.models import BaseRequest, SKUItem
    from backend.app.services import engine_client

    monkeypatch.setattr(engine_client, "get_task_queue", lambda: queue)
    posts = _fake_http(monkeypatch, engine_client)
    req = BaseRequest(skus=[SKUItem(name="x")], domain="market", callback_url="")

    result = engine_client.dispatch_batch_job("queued-1", req, "pipeline")

    assert result["status"] == "queued" and posts == []
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
    from engine.pipeline import worker_runner

    monkeypatch.setattr(worker_runner, "get_task_queue", lambda: queue)
    executed = []
    monkeypatch.setattr(worker_runner, "_execute_job", lambda p: executed.append(p["job_id"]))

    queue.request_cancel("cancel-me")  # e.g. sent through a different gateway/engine
    worker_runner._run_queued_job(_payload("cancel-me"))
    worker_runner._run_queued_job(_payload("keep-me"))

    assert executed == ["keep-me"]


def test_queued_job_errors_propagate_for_retry(monkeypatch):
    from engine.pipeline import worker_runner

    def boom(payload):
        raise ConnectionError("db blip")

    monkeypatch.setattr(worker_runner, "_execute_job", boom)
    monkeypatch.setattr(worker_runner, "get_task_queue", lambda: None)
    with pytest.raises(ConnectionError):
        worker_runner._run_queued_job(_payload("retry-me"))
