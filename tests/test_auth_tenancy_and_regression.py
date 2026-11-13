"""API Key authentication, observability, and golden regression tests."""

import io
import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.core import config as backend_config
from engine.core import observability
from engine.rules_engine import loader

# ---------------------------------------------------------------- shared fixtures


@pytest.fixture(scope="module")
def app_client(pg):
    """The real Gateway app (all middleware: trace ids, audit log, ETag); startup hooks not run."""
    from backend.main import app

    return TestClient(app)


def test_api_key_authentication(app_client, monkeypatch):
    """Verifies that API key authentication is enforced when configured."""
    monkeypatch.setattr(backend_config, "API_KEY", "secret-test-key-12345")

    # 1. Valid API key succeeds
    resp = app_client.get("/rules", headers={"X-API-Key": "secret-test-key-12345"})
    assert resp.status_code == 200

    # 2. Invalid API key is rejected with 401
    resp_bad = app_client.get("/rules", headers={"X-API-Key": "wrong-key"})
    assert resp_bad.status_code == 401
    assert resp_bad.json()["detail"] == "Invalid API key."

    # 3. Valid Bearer API key succeeds
    resp_bearer = app_client.get(
        "/rules", headers={"Authorization": "Bearer secret-test-key-12345"}
    )
    assert resp_bearer.status_code == 200

    # 4. Invalid Bearer API key is rejected with 401
    resp_bearer_bad = app_client.get("/rules", headers={"Authorization": "Bearer wrong-key"})
    assert resp_bearer_bad.status_code == 401

    # 5. Direct internal access without key defaults to team-admin
    resp_team = app_client.get("/rules")
    assert resp_team.status_code == 200


# ---------------------------------------------------------------- metrics & observability

METRIC_NAMES = [
    "matchops_skus_processed_total",
    "matchops_match_confidence",
    "matchops_onnx_inference_duration_seconds",
    "matchops_queue_depth",
    "matchops_dlq_size",
]


def test_gateway_and_engine_metrics_endpoints(app_client, monkeypatch):
    stats = {"realtime": 2, "bulk": 5, "delayed": 1, "inflight": 3, "dlq": 4}
    monkeypatch.setattr(
        "engine.pipeline.task_queue.get_task_queue", lambda: SimpleNamespace(stats=lambda: stats)
    )
    observability.SKUS_PROCESSED.labels(domain="market", status="high").inc(3)
    observability.MATCH_CONFIDENCE.observe(0.92)
    observability.ONNX_INFERENCE_SECONDS.labels(model="embed").observe(0.01)

    resp = app_client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    raw_payload, content_type = observability.metrics_payload()
    assert content_type.startswith("text/plain")

    for body in (resp.text, raw_payload.decode("utf-8")):
        for name in METRIC_NAMES:
            assert name in body, name
        assert 'matchops_skus_processed_total{domain="market",status="high"} 3.0' in body
        assert 'matchops_queue_depth{queue="bulk"} 5.0' in body
        assert "matchops_dlq_size 4.0" in body
        assert 'matchops_onnx_inference_duration_seconds_count{model="embed"}' in body


def test_onnx_timing_is_recorded_per_model():
    import numpy as np
    from prometheus_client import REGISTRY

    from engine.nlp.embedding_engine import EmbeddingEngine

    def count(model):
        name = "matchops_onnx_inference_duration_seconds_count"
        return REGISTRY.get_sample_value(name, {"model": model}) or 0.0

    eng = EmbeddingEngine.__new__(EmbeddingEngine)
    eng.rerank_tokenizer = lambda a, b, **kw: {"input_ids": np.ones((len(a), 1), dtype=np.int64)}
    eng.cross_session = SimpleNamespace(run=lambda _o, inp: [inp["input_ids"].astype(np.float32)])
    before = count("cross_encoder")
    eng.score_cross_encoder([["a", f"b{i}"] for i in range(70)], batch_size=32)
    assert count("cross_encoder") - before == 3  # one observation per micro-batch (32+32+6)


def test_trace_id_flows_from_gateway_request_into_engine_worker_logs(
    app_client, monkeypatch, caplog
):
    from backend.app.services import engine_client
    from engine.pipeline import task_queue

    captured = []
    fake_queue = SimpleNamespace(
        enqueue=lambda payload, queue=None: captured.append(payload) or "task-1"
    )
    monkeypatch.setattr(engine_client, "get_task_queue", lambda: fake_queue)

    trace = f"trace-{uuid.uuid4().hex[:12]}"
    resp = app_client.post(
        "/batches",
        data={"domain": "market", "created_by": "tester", "task": "pipeline"},
        files={"file": ("t.csv", io.BytesIO(b"name\nCoca Cola 1.5L\n"), "text/csv")},
        headers={"X-Request-ID": trace},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Request-ID"] == trace  # echoed back
    assert captured[0]["trace_id"] == trace  # stored in the Redis task payload

    # Engine side: the worker runs the task with the gateway's trace bound to its logs
    seen = {}

    def handler(payload):
        seen["trace"] = observability.TRACE_ID.get()
        logging.getLogger("matchops.engine.worker").info("processing job")

    fake = SimpleNamespace(
        ack=lambda tid: None, fail=lambda t, e: "retry", renew=lambda tid: True, lease_seconds=30
    )
    handler_logger = logging.getLogger("matchops.engine.worker")
    filt = observability.TraceIdFilter()
    with caplog.at_level(logging.INFO, logger="matchops.engine.worker"):
        caplog.handler.addFilter(filt)
        handler_logger.addFilter(filt)
        if caplog.handler not in logging.getLogger().handlers:
            logging.getLogger().addHandler(caplog.handler)
        task_queue._run_claimed(
            fake,
            {"task_id": "task-1", "job_id": resp.json()["job_id"], "payload": captured[0]},
            handler,
            None,
        )
        caplog.handler.removeFilter(filt)
        handler_logger.removeFilter(filt)

    assert seen["trace"] == trace
    assert any(
        getattr(r, "trace_id", None) == trace
        for r in caplog.records
        if r.name == handler_logger.name
    )

    from backend.app.db.session import get_engine

    with get_engine().begin() as conn:
        job_id = resp.json()["job_id"]
        conn.execute(text("DELETE FROM batches WHERE id = :i"), {"i": job_id})
        conn.execute(text("DELETE FROM jobs WHERE id = :i"), {"i": job_id})


def test_invalid_incoming_trace_ids_are_replaced(app_client, monkeypatch):
    from backend.app.api import routes

    monkeypatch.setattr(routes, "get_engine_health", lambda: {})
    resp = app_client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert resp.headers["X-Request-ID"] != "bad id\nwith newline"
    assert len(resp.headers["X-Request-ID"]) == 32


def test_json_log_format(monkeypatch):
    stream = io.StringIO()
    log = logging.getLogger(f"json_test_{uuid.uuid4().hex}")
    log.propagate = False
    log.addHandler(logging.StreamHandler(stream))
    monkeypatch.setattr(observability.config, "LOG_FORMAT", "json")
    observability.configure_logging(log)
    with observability.trace_context("abc123"):
        log.warning("hello %s", "world")
    entry = json.loads(stream.getvalue().strip())
    assert entry["message"] == "hello world"
    assert entry["level"] == "WARNING" and entry["trace_id"] == "abc123"
    assert entry["ts"] and entry["logger"] == log.name


# ---------------------------------------------------------------- golden regression

GOLDEN = json.loads((Path(__file__).parent / "golden_dataset.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_rules():
    """Seeds the production business rules captured in the golden dataset."""
    prev = (loader._RULES_CACHE, loader._RULES_VERSION)
    loader.set_rules_from_dicts(GOLDEN["rules"])
    yield
    loader._RULES_CACHE, loader._RULES_VERSION = prev
    loader._cached_rules.cache_clear()


def test_golden_dataset_shape():
    cases = GOLDEN["cases"]
    assert len(cases) >= 50
    assert len({c["id"] for c in cases}) == len(cases)
    categories = {c["category"] for c in cases}
    for required in (
        "seafood_rule",
        "seafood_rule_exclusion",
        "unit_variant",
        "beverage",
        "combo_set_menu",
        "typo",
        "false_positive_trap",
    ):
        assert required in categories, required
    assert {c["domain"] for c in cases} == {"food", "market"}


@pytest.mark.live_models
def test_golden_regression(golden_rules):
    import os

    import httpx

    from engine import config
    from engine.core.resource_loader import get_classifier, get_pipeline
    from engine.pipeline.processor import process_request

    for d in ("food", "market"):
        meta_path = os.path.join(config.CACHE_DIR, f"{d}_catalog_metadata.pkl")
        if (
            not os.path.exists(config.BI_ENCODER_ONNX)
            or not os.path.exists(config.get_bt_arcface_onnx_path(d))
            or not os.path.exists(meta_path)
        ):
            pytest.skip(f"Local ONNX models or synced '{d}' catalog cache not found on disk")
    try:
        if httpx.get(f"{config.QDRANT_URL.rstrip('/')}/readyz", timeout=1.0).status_code >= 500:
            pytest.skip(f"Qdrant not ready at {config.QDRANT_URL}")
    except Exception:
        pytest.skip(f"Qdrant not reachable at {config.QDRANT_URL}")

    outcomes = []
    for domain in ("food", "market"):
        cases = [c for c in GOLDEN["cases"] if c["domain"] == domain]
        get_classifier(domain)
        get_pipeline(domain)
        skus = [{"name": c["sku"], "price": 0.0, "description": "", "category": ""} for c in cases]
        results = process_request(task="classifier", domain=domain, skus=skus)["results"]
        assert len(results) == len(cases)
        for case, res in zip(cases, results):
            gks = {g.strip() for g in (res.get("suggested_gk") or "").split(",") if g.strip()}
            bt_ok = res.get("suggested_bt") in case["expected_bt"]
            missing = set(case.get("must_have_gk", [])) - gks
            forbidden = set(case.get("must_not_have_gk", [])) & gks
            outcomes.append(
                {
                    "case": case,
                    "bt": res.get("suggested_bt"),
                    "ok": bt_ok and not missing and not forbidden,
                    "missing": missing,
                    "forbidden": forbidden,
                }
            )

    accuracy = sum(o["ok"] for o in outcomes) / len(outcomes)
    critical = [o for o in outcomes if o["forbidden"] and o["case"].get("critical")]
    misses = "\n".join(
        f"  {o['case']['id']} {o['case']['sku']!r}: got bt={o['bt']!r} "
        f"expected {o['case']['expected_bt']} missing={sorted(o['missing'])} forbidden={sorted(o['forbidden'])}"
        for o in outcomes
        if not o["ok"]
    )
    print(
        f"\nGolden accuracy {accuracy:.1%} ({sum(o['ok'] for o in outcomes)}/{len(outcomes)})\n{misses}"
    )

    assert not critical, f"critical rule false positives:\n{misses}"
    assert accuracy >= GOLDEN["min_accuracy"], f"accuracy {accuracy:.1%} below threshold:\n{misses}"


# ---------------------------------------------------------------- gRPC fallback context


def test_grpc_fallback_preserves_trace(monkeypatch, loopback):
    """Redis unavailable -> Gateway dispatches via gRPC to the Engine; trace survives."""
    from backend.app.schemas.models import BaseRequest, SKUItem
    from backend.app.services import engine_client
    from engine import grpc_server
    from engine.pipeline import worker_runner

    monkeypatch.setattr(engine_client, "get_task_queue", lambda: None)  # queue disabled
    queued = []
    monkeypatch.setattr(
        grpc_server, "enqueue_batch_job", lambda p: queued.append(p) or {"status": "accepted"}
    )

    srv = grpc_server.start_grpc_server(port=0, host="127.0.0.1")
    try:
        monkeypatch.setattr(engine_client, "ENGINE_GRPC_URL", f"127.0.0.1:{srv._bound_port}")
        trace = f"fallback-{uuid.uuid4().hex[:8]}"
        req = BaseRequest(skus=[SKUItem(name="Coca Cola 1.5L")], domain="market", callback_url="")
        with observability.trace_context(trace):
            engine_client.dispatch_batch_job("fb-1", req, "pipeline")
    finally:
        srv.stop(grace=0)

    payload = queued[0]
    assert payload["trace_id"] == trace

    seen = {}

    def fake_execute(p):
        seen["trace"] = observability.TRACE_ID.get()

    monkeypatch.setattr(worker_runner, "_execute_job", fake_execute)
    worker_runner._batch_queue.put(payload)
    worker_runner._batch_queue.put(None)  # stop sentinel
    worker_runner._worker_loop()
    assert seen == {"trace": trace}


def test_grpc_fallback_defaults_when_context_missing(monkeypatch, loopback):
    import grpc

    from engine import grpc_server
    from engine.protos import engine_service_pb2 as pb2
    from engine.protos import engine_service_pb2_grpc as pb2_grpc

    queued = []
    monkeypatch.setattr(
        grpc_server, "enqueue_batch_job", lambda p: queued.append(p) or {"status": "accepted"}
    )
    srv = grpc_server.start_grpc_server(port=0, host="127.0.0.1")
    try:
        with grpc.insecure_channel(f"127.0.0.1:{srv._bound_port}") as chan:
            stub = pb2_grpc.EngineInferenceServiceStub(chan)
            stub.DispatchBatchJob(
                pb2.BatchJobRequest(job_id="x", skus=[pb2.SKUInput(name="a")]), timeout=5.0
            )
    finally:
        srv.stop(grace=0)

    assert queued[0]["trace_id"] == "-"
    with observability.trace_context("-") as bound:
        assert bound != "-" and len(bound) == 32  # placeholder gets a real trace id


# ---------------------------------------------------------------- system status


def test_system_status_endpoint(app_client, monkeypatch):
    from backend.app.api.endpoints import queue as queue_api

    stats = {"realtime": 1, "bulk": 2, "delayed": 0, "inflight": 1, "dlq": 3}
    dlq = [
        {"task_id": "t1", "payload": {"data": "item1"}},
        {"task_id": "t2", "payload": {"data": "item2"}},
        {"task_id": "t3", "payload": {"data": "item3"}},
    ]
    fake = SimpleNamespace(
        stats=lambda: stats,
        dlq_items=lambda limit=100: dlq[:limit],
        dlq_count=lambda: len(dlq),
    )
    monkeypatch.setattr(queue_api, "get_task_queue", lambda: fake)

    resp = app_client.get("/system/status").json()
    assert resp["dlq_size"] == 3
    assert resp["queue_depth"] == stats
    assert resp["queue_online"] is True
    assert isinstance(resp["processed_skus"], int)
    assert resp["rule_version"] == loader.snapshot_rules().version


def test_system_status_when_queue_offline(app_client, monkeypatch):
    from backend.app.api.endpoints import queue as queue_api

    def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(queue_api, "get_task_queue", down)
    body = app_client.get("/system/status").json()
    assert body["queue_online"] is False and body["dlq_size"] is None
    assert isinstance(body["processed_skus"], int)
