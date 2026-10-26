"""Phase 4: JWT/RBAC, multi-tenant isolation, migration 0002, observability, golden regression."""

import io
import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from backend.app.api.endpoints.auth import create_user
from backend.app.core import config as backend_config
from engine.core import observability
from engine.rules_engine import loader

PASSWORD = "correct-horse-battery"


# ---------------------------------------------------------------- shared fixtures


@pytest.fixture(scope="module")
def app_client(pg):
    """The real Gateway app (all middleware: trace ids, audit log, ETag); startup hooks not run."""
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setattr(backend_config, "AUTH_ENABLED", True)
    monkeypatch.setattr(backend_config, "JWT_SECRET", "test-secret-" + "x" * 32)


@pytest.fixture(scope="module")
def tenants(pg):
    """Two isolated tenants with users of every role, plus a platform admin."""
    suffix = uuid.uuid4().hex[:6]
    t_a, t_b = f"tenant_a_{suffix}", f"tenant_b_{suffix}"
    users = {
        "platform_admin": ("admin", "default"),
        "a_admin": ("admin", t_a),
        "a_engineer": ("rule_engineer", t_a),
        "a_reviewer": ("reviewer", t_a),
        "b_reviewer": ("reviewer", t_b),
    }
    names = {}
    with pg.begin() as conn:
        for key, (role, tenant) in users.items():
            names[key] = f"{key}_{suffix}"
            create_user(conn, names[key], PASSWORD, role, tenant)
    yield SimpleNamespace(a=t_a, b=t_b, user=names)
    with pg.begin() as conn:
        conn.execute(
            text("DELETE FROM users WHERE username = ANY(:u)"), {"u": list(names.values())}
        )
        for table in ("processed_skus", "batches", "jobs", "rules"):
            conn.execute(
                text(f"DELETE FROM {table} WHERE tenant_id IN (:a, :b)"), {"a": t_a, "b": t_b}
            )
    loader.refresh_rules_cache()


def _token(client, username: str) -> str:
    resp = client.post("/auth/token", data={"username": username, "password": PASSWORD})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _hdr(client, username: str, **extra) -> dict:
    return {"Authorization": f"Bearer {_token(client, username)}", **extra}


RULE = {
    "rule_id": "ignored",
    "domain": "market",
    "priority": 900,
    "description": "phase4 test rule",
    "reasoning": "rbac test",
    "condition_logic": "AND",
    "is_active": 1,
    "conditions": [
        {"condition_group": 1, "condition_type": "sku_contains", "value": "zzqq", "negate": 0}
    ],
    "actions": [{"action_type": "add_gk", "value": "Phase4"}],
}


# ---------------------------------------------------------------- JWT issuance


def test_jwt_issuance_and_claims(app_client, tenants, auth_on):
    resp = app_client.post(
        "/auth/token", data={"username": tenants.user["a_engineer"], "password": PASSWORD}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "rule_engineer" and body["tenant_id"] == tenants.a
    claims = jwt.decode(body["access_token"], backend_config.JWT_SECRET, algorithms=["HS256"])
    assert claims["sub"] == tenants.user["a_engineer"]
    assert claims["role"] == "rule_engineer" and claims["tenant_id"] == tenants.a
    assert claims["exp"] > claims["iat"]


def test_bad_credentials_and_tokens_are_rejected(app_client, tenants, auth_on):
    bad = app_client.post(
        "/auth/token", data={"username": tenants.user["a_admin"], "password": "wrong-password"}
    )
    assert bad.status_code == 401
    assert app_client.get("/jobs").status_code == 401  # auth on, no credentials
    assert app_client.get("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    forged = jwt.encode(
        {"sub": "x", "role": "admin", "tenant_id": "default", "exp": 9999999999},
        "not-the-real-secret-" + "y" * 32,
        algorithm="HS256",
    )
    assert app_client.get("/jobs", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    assert app_client.get("/jobs", headers={"X-API-Key": "wrong"}).status_code == 401


def test_public_endpoints_stay_open_with_auth_on(app_client, auth_on, monkeypatch):
    from backend.app.api import routes

    monkeypatch.setattr(routes, "get_engine_health", lambda: {"status": "ok"})
    assert app_client.get("/health").status_code == 200
    assert app_client.get("/metrics").status_code == 200


# ---------------------------------------------------------------- RBAC boundaries


def test_reviewer_cannot_mutate_rules_or_replay_dlq(app_client, tenants, auth_on):
    h = _hdr(app_client, tenants.user["a_reviewer"])
    assert app_client.post("/rules", json=RULE, headers=h).status_code == 403
    assert app_client.put("/rules/1", json=RULE, headers=h).status_code == 403
    assert app_client.delete("/rules/1", headers=h).status_code == 403
    assert (
        app_client.put("/rules/reorder", json={"ordered_rule_ids": ["1"]}, headers=h).status_code
        == 403
    )
    assert app_client.post("/queue/dlq/any/replay", headers=h).status_code == 403
    assert app_client.get("/queue/dlq", headers=h).status_code == 403
    # ...but can read and submit work
    assert app_client.get("/rules", headers=h).status_code == 200
    assert app_client.get("/jobs", headers=h).status_code == 200
    assert app_client.get("/processed-skus", headers=h).status_code == 200


def test_rule_engineer_manages_rules_but_not_admin_endpoints(app_client, tenants, auth_on):
    h = _hdr(app_client, tenants.user["a_engineer"])
    created = app_client.post("/rules", json=RULE, headers=h)
    assert created.status_code == 200, created.text
    rule_id = created.json()["rule_id"]
    assert (
        app_client.put(f"/rules/{rule_id}", json={**RULE, "priority": 901}, headers=h).status_code
        == 200
    )
    assert app_client.delete(f"/rules/{rule_id}", headers=h).status_code == 200

    assert app_client.get("/queue/dlq", headers=h).status_code == 403
    assert app_client.post("/queue/dlq/any/replay", headers=h).status_code == 403
    assert app_client.post("/models/reload", headers=h).status_code == 403
    assert app_client.post("/catalog/build-cache", headers=h).status_code == 403
    assert app_client.get("/api-requests", headers=h).status_code == 403
    assert (
        app_client.post(
            "/auth/users",
            json={"username": "nope_user", "password": PASSWORD, "role": "reviewer"},
            headers=h,
        ).status_code
        == 403
    )


def test_admin_succeeds_on_admin_endpoints(app_client, tenants, auth_on, monkeypatch):
    from backend.app.api.endpoints import models as models_api
    from backend.app.api.endpoints import queue as queue_api

    fake_queue = SimpleNamespace(dlq_items=lambda limit: [], replay=lambda task_id: None)
    monkeypatch.setattr(queue_api, "get_task_queue", lambda: fake_queue)
    monkeypatch.setattr(models_api, "reload_engine_models", lambda: {"status": "ok"})

    tenant_admin = _hdr(app_client, tenants.user["a_admin"])
    assert app_client.get("/queue/dlq", headers=tenant_admin).status_code == 200
    assert app_client.post("/queue/dlq/missing/replay", headers=tenant_admin).status_code == 404
    created = app_client.post("/rules", json=RULE, headers=tenant_admin)
    assert created.status_code == 200
    app_client.delete(f"/rules/{created.json()['rule_id']}", headers=tenant_admin)

    new_user = f"made_by_admin_{uuid.uuid4().hex[:6]}"
    made = app_client.post(
        "/auth/users",
        json={"username": new_user, "password": PASSWORD, "role": "reviewer"},
        headers=tenant_admin,
    )
    assert made.status_code == 201 and made.json()["tenant_id"] == tenants.a
    listed = app_client.get("/auth/users", headers=tenant_admin).json()
    assert new_user in {u["username"] for u in listed}
    assert all(u["tenant_id"] == tenants.a for u in listed)  # only their own tenant's users

    # Shared, platform-wide operations need the operator tenant's admin
    assert app_client.post("/models/reload", headers=tenant_admin).status_code == 403
    platform = _hdr(app_client, tenants.user["platform_admin"])
    assert app_client.post("/models/reload", headers=platform).status_code == 200
    assert app_client.get("/api-requests", headers=platform).status_code == 200

    from engine.core.db import get_engine

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM users WHERE username = :u"), {"u": new_user})


def test_api_key_and_dev_mode_keep_backwards_compatibility(app_client, monkeypatch):
    # Dev mode (AUTH_ENABLED off): no credentials -> platform admin of 'default'
    monkeypatch.setattr(backend_config, "AUTH_ENABLED", False)
    me = app_client.get("/auth/me").json()
    assert me == {
        "username": "anonymous-dev",
        "role": "admin",
        "tenant_id": "default",
        "platform_admin": True,
    }

    # Auth on: the service API key still works and acts as platform admin
    monkeypatch.setattr(backend_config, "AUTH_ENABLED", True)
    monkeypatch.setattr(backend_config, "JWT_SECRET", "s" * 40)
    key = {"X-API-Key": backend_config.API_KEY}
    assert app_client.get("/auth/me", headers=key).json()["platform_admin"] is True
    assert app_client.get("/jobs", headers=key).status_code == 200


def test_auth_enabled_without_secret_refuses_to_issue_tokens(monkeypatch):
    from backend.app.core import security

    monkeypatch.setattr(backend_config, "AUTH_ENABLED", True)
    monkeypatch.setattr(backend_config, "JWT_SECRET", "")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        security.create_access_token("u", "admin", "default")


# ---------------------------------------------------------------- multi-tenant isolation


@pytest.fixture
def tenant_data(pg, tenants):
    """One job, batch and result row per tenant."""
    ids = {}
    with pg.begin() as conn:
        for t in (tenants.a, tenants.b):
            job_id = f"p4-{uuid.uuid4().hex[:10]}"
            ids[t] = job_id
            conn.execute(
                text(
                    "INSERT INTO jobs (id, batch_id, type, status, current_stage, started_at, domain, "
                    "tenant_id, input_skus_json) VALUES (:id, :id, 'pipeline', 'completed', 'done', "
                    "now(), 'market', :t, '[]')"
                ),
                {"id": job_id, "t": t},
            )
            conn.execute(
                text(
                    "INSERT INTO batches (id, source, status, tenant_id) VALUES (:id, 'upload', 'completed', :t)"
                ),
                {"id": job_id, "t": t},
            )
            conn.execute(
                text(
                    "INSERT INTO processed_skus (id, batch_id, sku_index, rule_version, sku_name, domain, tenant_id) "
                    "VALUES (:id, :b, 0, 'v', :name, 'market', :t)"
                ),
                {"id": uuid.uuid4().hex, "b": job_id, "name": f"sku of {t}", "t": t},
            )
    yield ids
    with pg.begin() as conn:
        for job_id in ids.values():
            conn.execute(text("DELETE FROM processed_skus WHERE batch_id = :id"), {"id": job_id})
            conn.execute(text("DELETE FROM batches WHERE id = :id"), {"id": job_id})
            conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job_id})


def test_tenant_a_cannot_see_or_touch_tenant_b(app_client, tenants, tenant_data, auth_on):
    a = _hdr(app_client, tenants.user["a_reviewer"])
    a_job, b_job = tenant_data[tenants.a], tenant_data[tenants.b]

    listed = {j["id"] for j in app_client.get("/jobs", headers=a).json()}
    assert a_job in listed and b_job not in listed
    assert app_client.get(f"/jobs/{a_job}", headers=a).status_code == 200
    assert app_client.get(f"/jobs/{b_job}", headers=a).status_code == 404
    assert app_client.post(f"/jobs/{b_job}/cancel", headers=a).status_code == 404
    assert app_client.post(f"/jobs/{b_job}/retry", headers=a).status_code == 404
    assert app_client.get(f"/batches/{a_job}", headers=a).status_code == 200
    assert app_client.get(f"/batches/{b_job}", headers=a).status_code == 404

    skus = app_client.get("/processed-skus", headers=a, params={"limit": 0}).json()
    assert {s["batch_id"] for s in skus} >= {a_job} and b_job not in {s["batch_id"] for s in skus}
    export = app_client.get("/processed-skus/export", headers=a).text
    assert f"sku of {tenants.a}" in export and f"sku of {tenants.b}" not in export

    # Non-admins cannot switch tenant with the header
    assert app_client.get("/jobs", headers={**a, "X-Tenant-ID": tenants.b}).status_code == 403
    # Tenant admins cannot either; only the platform admin can
    a_admin = _hdr(app_client, tenants.user["a_admin"], **{"X-Tenant-ID": tenants.b})
    assert app_client.get("/jobs", headers=a_admin).status_code == 403
    platform = _hdr(app_client, tenants.user["platform_admin"], **{"X-Tenant-ID": tenants.b})
    assert {j["id"] for j in app_client.get("/jobs", headers=platform).json()} == {b_job}


def test_rules_are_isolated_per_tenant_including_in_the_engine(app_client, tenants, auth_on):
    b_admin_token = _hdr(app_client, tenants.user["platform_admin"], **{"X-Tenant-ID": tenants.b})
    created = app_client.post("/rules", json=RULE, headers=b_admin_token)
    assert created.status_code == 200
    rule_id = created.json()["rule_id"]

    a = _hdr(app_client, tenants.user["a_engineer"])
    assert rule_id not in {r["rule_id"] for r in app_client.get("/rules", headers=a).json()}
    assert app_client.put(f"/rules/{rule_id}", json=RULE, headers=a).status_code == 404
    assert app_client.delete(f"/rules/{rule_id}", headers=a).status_code == 404
    assert (
        app_client.post(f"/rules/{rule_id}/test", json={"sample_record": {}}, headers=a).status_code
        == 404
    )

    # Engine side: a job pinned to tenant A never evaluates tenant B's rule
    loader.refresh_rules_cache()
    snap_a, snap_b = loader.snapshot_rules(tenants.a), loader.snapshot_rules(tenants.b)
    assert rule_id not in {r.rule_id for r in snap_a.rules("market")}
    assert rule_id in {r.rule_id for r in snap_b.rules("market")}
    assert snap_a.version != snap_b.version  # results are versioned per tenant
    assert rule_id not in {r.rule_id for r in loader.get_rules("market")}  # default tenant

    from engine.rules_engine import run_rules_engine

    record = {"domain": "market", "sku_name": "zzqq widget", "gk": []}
    with loader.pinned_rules(snap_a):
        assert run_rules_engine(dict(record))["gk"] == []
    with loader.pinned_rules(snap_b):
        assert run_rules_engine(dict(record))["gk"] == ["Phase4"]


def test_new_jobs_and_results_are_stamped_with_the_tenant(
    pg, app_client, tenants, auth_on, monkeypatch
):
    from backend.app.services import worker
    from engine.pipeline.persistence import record_job_completion

    dispatched = []
    monkeypatch.setattr(worker, "dispatch_batch_job", lambda **kw: dispatched.append(kw))
    csv_bytes = b"name,price\nAnchor Butter 200g,850\n"
    resp = app_client.post(
        "/batches",
        data={"domain": "market", "created_by": "tester", "task": "pipeline"},
        files={"file": ("t.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers=_hdr(app_client, tenants.user["a_reviewer"]),
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    assert dispatched[0]["tenant_id"] == tenants.a

    record_job_completion(
        job_id, "market", "pipeline", [{"suggested_bt": "Butter"}], "v", 0.1, 1, 0, 0, 100.0
    )
    with pg.connect() as conn:
        assert (
            conn.execute(text("SELECT tenant_id FROM jobs WHERE id = :i"), {"i": job_id}).scalar()
            == tenants.a
        )
        assert (
            conn.execute(
                text("SELECT tenant_id FROM batches WHERE id = :i"), {"i": job_id}
            ).scalar()
            == tenants.a
        )
        assert {
            r[0]
            for r in conn.execute(
                text("SELECT tenant_id FROM processed_skus WHERE batch_id = :i"), {"i": job_id}
            )
        } == {tenants.a}


# ---------------------------------------------------------------- migration 0002


def _alembic_cfg(url: str):
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def test_migration_0002_upgrade_downgrade_is_clean(pg):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from engine.core.models import metadata

    base = make_url(pg.url.render_as_string(hide_password=False))
    name = f"matchops_mig_{uuid.uuid4().hex[:8]}"
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = base.set(database=name).render_as_string(hide_password=False)
    scratch = create_engine(url)
    try:
        cfg = _alembic_cfg(url)

        def tenant_cols():
            insp = inspect(scratch)
            return {
                t: "tenant_id" in {c["name"] for c in insp.get_columns(t)}
                for t in ("jobs", "batches", "processed_skus", "rules")
            }

        command.upgrade(cfg, "0001")
        with scratch.begin() as conn:  # pre-existing data must survive the upgrade
            conn.execute(text("INSERT INTO jobs (id, status) VALUES ('legacy', 'completed')"))

        command.upgrade(cfg, "head")
        assert all(tenant_cols().values())
        assert "users" in inspect(scratch).get_table_names()
        assert "idx_jobs_tenant_started" in {
            i["name"] for i in inspect(scratch).get_indexes("jobs")
        }
        with scratch.connect() as conn:
            assert (
                conn.execute(text("SELECT tenant_id FROM jobs WHERE id = 'legacy'")).scalar()
                == "default"
            )
            assert compare_metadata(MigrationContext.configure(conn), metadata) == []  # no drift

        command.downgrade(cfg, "0001")
        assert not any(tenant_cols().values())
        assert "users" not in inspect(scratch).get_table_names()
        with scratch.connect() as conn:
            assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0001"
            assert conn.execute(text("SELECT count(*) FROM jobs WHERE id = 'legacy'")).scalar() == 1

        command.upgrade(cfg, "head")  # and back up again
        assert all(tenant_cols().values())
    finally:
        scratch.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


# ---------------------------------------------------------------- observability


METRIC_NAMES = [
    "matchops_skus_processed_total",
    "matchops_match_confidence_bucket",
    "matchops_onnx_inference_duration_seconds",
    "matchops_queue_depth",
    "matchops_dlq_size",
]


def test_gateway_and_engine_metrics_endpoints(app_client, monkeypatch):
    from engine.server import app as engine_app

    stats = {"realtime": 2, "bulk": 5, "delayed": 1, "inflight": 3, "dlq": 4}
    monkeypatch.setattr(
        "engine.pipeline.task_queue.get_task_queue", lambda: SimpleNamespace(stats=lambda: stats)
    )
    observability.SKUS_PROCESSED.labels(domain="market", status="high", tenant_id="t_metrics").inc(
        3
    )
    observability.MATCH_CONFIDENCE.observe(0.92)
    observability.ONNX_INFERENCE_SECONDS.labels(model="embed").observe(0.01)

    for client in (app_client, TestClient(engine_app)):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        for name in METRIC_NAMES:
            assert name in body, name
        assert (
            'matchops_skus_processed_total{domain="market",status="high",tenant_id="t_metrics"} 3.0'
            in body
        )
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
    eng.score_cross_encoder([["a", "b"]] * 70, batch_size=32)
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
        task_queue._run_claimed(
            fake,
            {"task_id": "task-1", "job_id": resp.json()["job_id"], "payload": captured[0]},
            handler,
            None,
        )
        caplog.handler.removeFilter(filt)
    assert seen["trace"] == trace
    assert any(
        getattr(r, "trace_id", None) == trace
        for r in caplog.records
        if r.name == handler_logger.name
    )

    from engine.core.db import get_engine

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
def golden_rules(pg):
    """Seeds the production business rules captured in the golden dataset (default tenant)."""
    ids = [r["rule_id"] for r in GOLDEN["rules"]]
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM rules WHERE rule_id = ANY(:ids)"), {"ids": ids})
        for r in GOLDEN["rules"]:
            conn.execute(
                text(
                    "INSERT INTO rules (rule_id, domain, priority, description, reasoning, condition_logic) "
                    "VALUES (:rule_id, :domain, :priority, :description, :reasoning, :condition_logic)"
                ),
                r,
            )
            for c in r["conditions"]:
                conn.execute(
                    text(
                        "INSERT INTO conditions (rule_id, condition_group, condition_type, value, negate) "
                        "VALUES (:r, :condition_group, :condition_type, :value, :negate)"
                    ),
                    {"r": r["rule_id"], **c},
                )
            for a in r["actions"]:
                conn.execute(
                    text(
                        "INSERT INTO actions (rule_id, action_type, value) VALUES (:r, :action_type, :value)"
                    ),
                    {"r": r["rule_id"], **a},
                )
    loader.refresh_rules_cache()
    yield
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM rules WHERE rule_id = ANY(:ids)"), {"ids": ids})
    loader.refresh_rules_cache()


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


def test_golden_regression(golden_rules):
    from engine.core.resource_loader import get_classifier, get_pipeline
    from engine.pipeline.processor import process_request

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


# ---------------------------------------------------------------- HTTP fallback context (Phase 5)


def test_http_fallback_preserves_tenant_and_trace(monkeypatch):
    """Redis unavailable -> Gateway POSTs to the Engine; tenant and trace must survive the hop."""
    from backend.app.schemas.models import BaseRequest, SKUItem
    from backend.app.services import engine_client
    from engine import server as engine_server
    from engine.pipeline import worker_runner

    monkeypatch.setattr(engine_client, "get_task_queue", lambda: None)  # queue disabled
    engine_http = TestClient(engine_server.app)

    def post_to_engine(url, json=None, timeout=None):
        return engine_http.post(url.replace(engine_client.ENGINE_URL, ""), json=json)

    monkeypatch.setattr(engine_client._session, "post", post_to_engine)
    queued = []
    monkeypatch.setattr(
        engine_server, "enqueue_batch_job", lambda p: queued.append(p) or {"status": "accepted"}
    )

    trace = f"fallback-{uuid.uuid4().hex[:8]}"
    req = BaseRequest(skus=[SKUItem(name="Coca Cola 1.5L")], domain="market", callback_url="")
    with observability.trace_context(trace):
        engine_client.dispatch_batch_job("fb-1", req, "pipeline", tenant_id="tenant_fb")

    payload = queued[0]
    assert payload["tenant_id"] == "tenant_fb" and payload["trace_id"] == trace

    # Engine legacy worker loop: same tenant snapshot + trace binding as Redis-queued jobs
    seen = {}

    def fake_execute(p):
        seen["trace"] = observability.TRACE_ID.get()
        seen["tenant"] = p["tenant_id"]

    monkeypatch.setattr(worker_runner, "_execute_job", fake_execute)
    worker_runner._batch_queue.put(payload)
    worker_runner._batch_queue.put(None)  # stop sentinel
    worker_runner._worker_loop()
    assert seen == {"trace": trace, "tenant": "tenant_fb"}


def test_http_fallback_defaults_when_context_missing():
    from engine import server as engine_server

    req = engine_server.BatchProcessRequest(job_id="x", skus=[{"name": "a"}])
    assert req.tenant_id == "default" and req.trace_id == "-"
    with observability.trace_context("-") as bound:
        assert bound != "-" and len(bound) == 32  # placeholder gets a real trace id


# ---------------------------------------------------------------- system status (Phase 5)


def test_system_status_is_tenant_scoped(app_client, tenants, tenant_data, auth_on, monkeypatch):
    from backend.app.api.endpoints import queue as queue_api

    stats = {"realtime": 1, "bulk": 2, "delayed": 0, "inflight": 1, "dlq": 7}
    dlq = [
        {"task_id": "t1", "payload": {"tenant_id": tenants.a}},
        {"task_id": "t2", "payload": {"tenant_id": tenants.b}},
        {"task_id": "t3", "payload": {"tenant_id": tenants.b}},
    ]
    fake = SimpleNamespace(stats=lambda: stats, dlq_items=lambda limit: dlq)
    monkeypatch.setattr(queue_api, "get_task_queue", lambda: fake)

    a = app_client.get(
        "/system/status", headers=_hdr(app_client, tenants.user["a_reviewer"])
    ).json()
    assert a["tenant_id"] == tenants.a
    from engine.core.db import get_engine

    with get_engine().connect() as conn:
        expected = conn.execute(
            text("SELECT COUNT(*) FROM processed_skus WHERE tenant_id = :t"), {"t": tenants.a}
        ).scalar()
    assert a["processed_skus"] == expected >= 1  # tenant A's rows only
    assert a["dlq_size"] == 1
    assert a["queue_depth"] is None  # cross-tenant volume is hidden from tenant users
    assert a["rule_version"] == loader.snapshot_rules(tenants.a).version

    platform = _hdr(app_client, tenants.user["platform_admin"], **{"X-Tenant-ID": tenants.b})
    b = app_client.get("/system/status", headers=platform).json()
    assert b["dlq_size"] == 2 and b["queue_depth"] == stats and b["queue_online"] is True


def test_system_status_when_queue_offline(app_client, monkeypatch):
    from backend.app.api.endpoints import queue as queue_api

    def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(queue_api, "get_task_queue", down)
    body = app_client.get("/system/status").json()
    assert body["queue_online"] is False and body["dlq_size"] is None
    assert isinstance(body["processed_skus"], int)
