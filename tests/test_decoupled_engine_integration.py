import ast
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.api.routes import api_router
from backend.app.db.session import get_engine
from backend.app.grpc_server import clear_job_progress, start_backend_grpc_server
from backend.app.schemas.models import BaseRequest, SKUItem
from backend.app.services.worker import enqueue_job
from engine.pipeline.worker_runner import _get_backend_stub
from engine.protos import engine_service_pb2 as pb2

app = FastAPI()
app.include_router(api_router)
client = TestClient(app)


def _delete_job(job_id: str) -> None:
    clear_job_progress(job_id)
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job_id})
        conn.execute(text("DELETE FROM processed_skus WHERE batch_id = :id"), {"id": job_id})


class TestDecoupledEngineIntegration(unittest.TestCase):
    @pytest.mark.usefixtures("loopback")
    @patch("backend.app.api.routes.get_engine_health")
    def test_health_with_engine_status(self, mock_health):
        mock_health.return_value = {
            "status": "ok",
            "service": "matchops-engine",
            "loaded_domains": ["market", "food"],
            "all_models_ready": True,
        }
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "matchops-backend")
        self.assertIn("loaded_domains", data)
        self.assertEqual(data["loaded_domains"], ["market", "food"])

    @pytest.mark.usefixtures("pg")
    @patch("backend.app.services.worker.dispatch_batch_job")
    def test_enqueue_job_dispatches_to_engine(self, mock_dispatch):
        mock_dispatch.return_value = {"status": "queued", "job_id": "test_job_100"}
        req = BaseRequest(
            skus=[SKUItem(name="Test Item 1", price=10.0, description="Desc", category="Cat")],
            domain="market",
            callback_url="",
            sheet_name="Sheet1",
        )
        res = enqueue_job(req, "pipeline")
        self.assertIn("job_id", res)
        self.assertEqual(res["status"], "queued")
        self.assertEqual(res["total_skus"], 1)
        mock_dispatch.assert_called_once()

        # Clean up the job row inserted into the test DB
        _delete_job(res["job_id"])

    @pytest.mark.usefixtures("pg")
    def test_engine_callbacks_lifecycle(self):
        srv = start_backend_grpc_server(port=0, host="127.0.0.1")
        target = f"127.0.0.1:{srv._bound_port}"
        stub = _get_backend_stub(target)

        job_id = "test_cb_job_99"
        fail_job_id = "test_cb_fail_99"
        _delete_job(job_id)
        _delete_job(fail_job_id)
        try:
            with get_engine().begin() as conn:
                conn.execute(
                    text("""
                    INSERT INTO jobs (
                        id, batch_id, type, status, current_stage, total_items, completed_items, created_by, started_at, domain, input_skus_json
                    ) VALUES
                        (:id, :id, 'pipeline', 'queued', 'queued', 1, 0, 'test', now(), 'market', :skus),
                        (:fid, :fid, 'pipeline', 'queued', 'queued', 1, 0, 'test', now(), 'market', :skus)
                    """),
                    {
                        "id": job_id,
                        "fid": fail_job_id,
                        "skus": json.dumps([{"name": "Anchor Butter 200g"}]),
                    },
                )

            # 1. Engine reports progress over gRPC
            prog_resp = stub.ReportJobProgress(
                pb2.JobProgressRequest(
                    job_id=job_id,
                    current_stage="matching",
                    progress_pct=50.0,
                    eta_seconds=10,
                    has_eta=True,
                ),
                timeout=5.0,
            )
            self.assertEqual(prog_resp.status, "ok")

            # Verify progress reflected in GET /jobs/{id}
            job_get = client.get(f"/jobs/{job_id}")
            self.assertEqual(job_get.status_code, 200)
            job_data = job_get.json()
            self.assertEqual(job_data["status"], "running")
            self.assertEqual(job_data["current_stage"], "matching")
            self.assertEqual(job_data["progress_pct"], 50.0)

            # 2. Engine sends results to Gateway over gRPC; Gateway persists to PostgreSQL
            results_payload = [
                {
                    "matched_catalog_name": "Anchor Butter Salted 200g",
                    "score": 92.5,
                    "status": "High Confidence",
                    "logic_notes": "Direct catalog match",
                    "suggested_bt": "Butter",
                    "suggested_gk": "Butter, Dairy",
                    "suggested_region": "Dairy",
                }
            ]
            complete_resp = stub.CompleteJob(
                pb2.JobCompleteRequest(
                    job_id=job_id,
                    status="completed",
                    domain="market",
                    task="pipeline",
                    results_json=json.dumps(results_payload),
                    rule_version="test",
                    duration_minutes=0.25,
                    high_conf=1,
                    med_conf=0,
                    low_conf=0,
                    match_rate=100.0,
                ),
                timeout=5.0,
            )
            self.assertEqual(complete_resp.status, "ok")

            # A late progress RPC must not resurrect the finalized job
            stub.ReportJobProgress(
                pb2.JobProgressRequest(
                    job_id=job_id,
                    current_stage="late_stage",
                    progress_pct=99.0,
                    eta_seconds=1,
                    has_eta=True,
                ),
                timeout=5.0,
            )

            # Verify completed state and processed SKU row persisted by Gateway
            job_completed = client.get(f"/jobs/{job_id}")
            self.assertEqual(job_completed.status_code, 200)
            comp_data = job_completed.json()
            self.assertEqual(comp_data["status"], "completed")
            self.assertEqual(comp_data["current_stage"], "done")
            self.assertEqual(comp_data["progress_pct"], 100.0)
            self.assertEqual(comp_data["match_rate"], 100.0)

            skus = client.get("/processed-skus", params={"batch_id": job_id}).json()
            self.assertEqual(len(skus), 1)
            self.assertEqual(skus[0]["sku_name"], "Anchor Butter 200g")

            # 3. Engine reports failure over gRPC; Gateway marks job failed in PostgreSQL
            fail_resp = stub.FailJob(
                pb2.JobFailRequest(
                    job_id=fail_job_id,
                    status="failed",
                    error_message="ONNX runtime error",
                    duration_minutes=0.05,
                    is_cancelled=False,
                ),
                timeout=5.0,
            )
            self.assertEqual(fail_resp.status, "ok")
            failed_data = client.get(f"/jobs/{fail_job_id}").json()
            self.assertEqual(failed_data["status"], "failed")
            self.assertIn("ONNX runtime error", failed_data["error_message"])
        finally:
            srv.stop(grace=0)
            _delete_job(job_id)
            _delete_job(fail_job_id)

    def test_engine_has_zero_db_or_backend_imports(self):
        """Architectural boundary: engine/ must never import backend, sqlalchemy, or psycopg."""
        engine_root = Path(__file__).resolve().parent.parent / "engine"
        forbidden_roots = {"backend", "sqlalchemy", "psycopg", "psycopg2"}
        forbidden_tokens = re.compile(r"\b(DATABASE_URL|DB_POOL_SIZE|DB_MAX_OVERFLOW)\b")
        violations = []

        for py_file in sorted(engine_root.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in forbidden_roots:
                            violations.append(f"{py_file}:{node.lineno} import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if top in forbidden_roots:
                        violations.append(f"{py_file}:{node.lineno} from {node.module} import ...")
            if "protos" not in py_file.parts:
                for idx, line in enumerate(source.splitlines(), start=1):
                    if forbidden_tokens.search(line):
                        violations.append(f"{py_file}:{idx} {line.strip()}")

        self.assertEqual(violations, [], "Engine DB boundary violated:\n" + "\n".join(violations))

    def test_engine_and_backend_have_no_internal_rest_routes(self):
        """Architectural boundary: Engine <-> Backend RPCs run strictly over gRPC, never REST."""
        import engine.server as engine_server
        from backend.main import app as gateway_app

        gateway_paths = {getattr(r, "path", "") for r in gateway_app.routes}
        internal_gateway_routes = [p for p in gateway_paths if p.startswith("/api/internal/")]

        self.assertEqual(internal_gateway_routes, [])
        self.assertFalse(hasattr(engine_server, "app"))

    @pytest.mark.usefixtures("pg")
    def test_complete_job_typed_results_memory_eviction_and_db_error_abort(self):
        """CompleteJob accepts typed Protobuf results, evicts progress dicts, and aborts on DB error."""
        import grpc

        from backend.app.grpc_server import _job_eta, _job_progress, _job_stage
        from engine.core.grpc_utils import result_to_pipeline_proto

        srv = start_backend_grpc_server(port=0, host="127.0.0.1")
        stub = _get_backend_stub(f"127.0.0.1:{srv._bound_port}")
        job_id = "test_typed_cb_101"
        _delete_job(job_id)
        try:
            with get_engine().begin() as conn:
                conn.execute(
                    text("""
                    INSERT INTO jobs (
                        id, batch_id, type, status, current_stage, total_items, completed_items, created_by, started_at, domain, input_skus_json
                    ) VALUES (:id, :id, 'pipeline', 'queued', 'queued', 1, 0, 'test', now(), 'market', :skus)
                    """),
                    {"id": job_id, "skus": json.dumps([{"name": "Milo 400g"}])},
                )

            stub.ReportJobProgress(
                pb2.JobProgressRequest(
                    job_id=job_id,
                    current_stage="matching",
                    progress_pct=40.0,
                    has_eta=True,
                    eta_seconds=5,
                ),
                timeout=5.0,
            )
            self.assertIn(job_id, _job_progress)

            # Simulate transient DB outage: CompleteJob MUST abort with INTERNAL so queue retries
            with patch(
                "backend.app.grpc_server.record_job_completion",
                side_effect=RuntimeError("transient pg disconnect"),
            ):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.CompleteJob(
                        pb2.JobCompleteRequest(job_id=job_id, status="completed"), timeout=5.0
                    )
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INTERNAL)

            # Real typed completion succeeds and evicts all in-memory progress entries
            stub.CompleteJob(
                pb2.JobCompleteRequest(
                    job_id=job_id,
                    status="completed",
                    domain="market",
                    task="pipeline",
                    rule_version="v1",
                    high_conf=1,
                    match_rate=100.0,
                    results=[
                        result_to_pipeline_proto(
                            {
                                "matched_catalog_name": "Nestle Milo 400g",
                                "score": 0.96,
                                "status": "High Confidence",
                                "suggested_bt": "Malt Drink",
                                "pipeline_source": "Matcher",
                                "input_entities": {"brand": ["Milo"], "flavor": []},
                            }
                        )
                    ],
                ),
                timeout=5.0,
            )
            self.assertNotIn(job_id, _job_progress)
            self.assertNotIn(job_id, _job_eta)
            self.assertNotIn(job_id, _job_stage)

            # Late progress update after completion must also be evicted
            stub.ReportJobProgress(
                pb2.JobProgressRequest(job_id=job_id, current_stage="late", progress_pct=99.0),
                timeout=5.0,
            )
            self.assertNotIn(job_id, _job_progress)
        finally:
            srv.stop(grace=0)
            _delete_job(job_id)

    @pytest.mark.usefixtures("loopback")
    def test_grpc_secret_ssrf_pinning_typed_upload_and_error_mapping(self):
        """Verifies INTERNAL_GRPC_SECRET, SSRF pinning, typed upload, and 400/404 RPC error mapping."""
        from types import SimpleNamespace

        import grpc
        import numpy as np

        from backend.app.services import engine_client
        from engine import config as eng_cfg
        from engine.core.grpc_utils import grpc_metadata
        from engine.grpc_server import start_grpc_server
        from engine.protos import engine_service_pb2_grpc as pb2_grpc

        dispatched: list[dict] = []
        srv = start_grpc_server(port=0, host="127.0.0.1")
        target = f"127.0.0.1:{srv._bound_port}"
        old_secret = eng_cfg.INTERNAL_GRPC_SECRET
        old_url = engine_client.ENGINE_GRPC_URL
        try:
            eng_cfg.INTERNAL_GRPC_SECRET = "super-secret-token"
            engine_client.ENGINE_GRPC_URL = target
            raw_channel = grpc.insecure_channel(target)
            raw_stub = pb2_grpc.EngineInferenceServiceStub(raw_channel)

            # 1. Unauthenticated call is rejected
            with self.assertRaises(grpc.RpcError) as unauth:
                raw_stub.GetHealth(pb2.Empty(), timeout=5.0)
            self.assertEqual(unauth.exception.code(), grpc.StatusCode.UNAUTHENTICATED)

            # 2. Authenticated call succeeds; SSRF backend_grpc_url is ignored; typed upload works
            with patch("engine.grpc_server.enqueue_batch_job", side_effect=dispatched.append):
                raw_stub.DispatchUploadJob(
                    pb2.UploadJobRequest(
                        job_id="u1",
                        backend_grpc_url="evil.attacker:9999",
                        skus=[pb2.UploadSkuItem(fields={"Name": "Coke 500ml", "Price": "150"})],
                    ),
                    metadata=grpc_metadata(),
                    timeout=5.0,
                )
            self.assertEqual(len(dispatched), 1)
            self.assertEqual(dispatched[0]["backend_grpc_url"], eng_cfg.BACKEND_GRPC_URL)
            self.assertEqual(dispatched[0]["skus"][0]["Name"], "Coke 500ml")

            # 3. ValueError -> INVALID_ARGUMENT -> HTTP 400 on /interactive/run-single
            with patch("engine.grpc_server.process_request", side_effect=ValueError("bad domain")):
                resp_400 = client.post(
                    "/interactive/run-single",
                    json={"sku_name": "X", "domain": "invalid", "task": "pipeline"},
                )
            self.assertEqual(resp_400.status_code, 400)
            self.assertIn("bad domain", resp_400.json()["detail"])

            # 4. Missing Qdrant collection -> NOT_FOUND -> HTTP 404, and filters/score_threshold forwarded
            fake_qdrant = SimpleNamespace(
                get_collection=lambda collection_name: (_ for _ in ()).throw(
                    KeyError(f"Collection {collection_name} not found")
                ),
                query_points=lambda **kw: SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id="1",
                            score=0.91,
                            payload={
                                "has_filter": kw["query_filter"] is not None,
                                "threshold": kw["score_threshold"],
                            },
                        )
                    ]
                ),
            )
            fake_embed = SimpleNamespace(encode=lambda _q: {"dense": [np.zeros(4)]})
            with (
                patch("engine.grpc_server.VectorStore.get_client", return_value=fake_qdrant),
                patch(
                    "engine.core.resource_loader._get_shared_models",
                    return_value=(fake_embed, None),
                ),
            ):
                self.assertEqual(client.get("/vector-db/collections/missing").status_code, 404)
                s_resp = client.post(
                    "/vector-db/collections/market_catalog/search",
                    json={
                        "query": "milk",
                        "top_k": 5,
                        "score_threshold": 0.75,
                        "filters": {"basictype": "Milk"},
                    },
                )
                self.assertEqual(s_resp.status_code, 200)
                hit_payload = s_resp.json()["results"][0]["payload"]
                self.assertTrue(hit_payload["has_filter"])
                self.assertAlmostEqual(hit_payload["threshold"], 0.75)
            raw_channel.close()
        finally:
            eng_cfg.INTERNAL_GRPC_SECRET = old_secret
            engine_client.ENGINE_GRPC_URL = old_url
            srv.stop(grace=0)

    def test_sync_script_has_no_top_level_backend_imports(self):
        """scripts/sync.py must run across engine/backend containers without cross-service top-level imports."""
        sync_file = Path(__file__).resolve().parent.parent / "scripts" / "sync.py"
        tree = ast.parse(sync_file.read_text(encoding="utf-8"), filename=str(sync_file))
        disallowed_roots = {"backend", "engine", "torch", "transformers", "onnxruntime"}
        top_disallowed_imports = [
            node.lineno
            for node in tree.body
            if (
                isinstance(node, ast.Import)
                and any(a.name.split(".")[0] in disallowed_roots for a in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] in disallowed_roots
            )
        ]
        self.assertEqual(top_disallowed_imports, [])

    def test_model_load_lifecycle_concurrency_and_reload_cache_eviction(self):
        """trigger_model_load guards concurrent loads, evicts caches on reload, and skips duplicate classifier calls."""
        import threading
        import time

        from engine.core import resource_loader as rl

        saved_statuses = {d: dict(st) for d, st in rl._model_statuses.items()}
        saved_pipelines = dict(rl._pipelines)
        saved_classifiers = dict(rl._classifiers)
        try:
            for d in rl._model_statuses:
                rl._model_statuses[d] = {"pipeline": "ready", "classifier": "ready"}
            rl._pipelines["market"] = object()  # type: ignore[assignment]
            rl._classifiers["market"] = object()  # type: ignore[assignment]

            self.assertEqual(rl.trigger_model_load(reload=False)[0], "success")

            with rl._model_load_lock:
                self.assertTrue(rl.is_model_loading())
                self.assertEqual(rl.trigger_model_load(reload=False)[0], "ignored")

            done = threading.Event()
            loaded_domains: list[str] = []

            def fake_get_pipeline(domain, **_kw):
                loaded_domains.append(domain)
                rl._model_statuses[domain] = {"pipeline": "ready", "classifier": "ready"}
                if len(loaded_domains) == 2:
                    done.set()

            with (
                patch.object(rl, "_get_shared_models", return_value=(None, None)),
                patch.object(rl, "_rebuild_data_caches"),
                patch.object(rl, "refresh_rules_cache"),
                patch.object(rl, "start_invalidation_listener"),
                patch.object(rl, "clear_flavor_cache"),
                patch.object(rl, "get_pipeline", side_effect=fake_get_pipeline),
                patch.object(rl, "get_classifier") as mock_clf,
                patch("engine.pipeline.worker_runner.start_queue_worker", return_value=None),
            ):
                status, _ = rl.trigger_model_load(reload=True)
                self.assertEqual(status, "started")
                self.assertTrue(done.wait(timeout=5.0))
                while rl._model_load_lock.locked():
                    time.sleep(0.01)

            self.assertEqual(loaded_domains, ["market", "food"])
            mock_clf.assert_not_called()
            self.assertEqual(rl._pipelines, {})
            self.assertEqual(rl._classifiers, {})
            self.assertFalse(rl.is_model_loading())
        finally:
            rl._pipelines.clear()
            rl._pipelines.update(saved_pipelines)
            rl._classifiers.clear()
            rl._classifiers.update(saved_classifiers)
            for d, st in saved_statuses.items():
                rl._model_statuses[d] = st

    def test_engine_boot_autoload_and_immediate_worker_running_callback(self):
        """Engine serve() auto-triggers model load on boot, and _execute_job reports 0% stage before process_request."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from engine import server as eng_server
        from engine.pipeline import worker_runner
        from engine.rules_engine.loader import RulesSnapshot

        fake_srv = MagicMock()
        with (
            patch.object(eng_server, "start_grpc_server", return_value=fake_srv),
            patch.object(eng_server, "trigger_model_load") as mock_load,
            patch("engine.server.signal.signal"),
        ):
            eng_server.serve(port=0, host="127.0.0.1")
            mock_load.assert_called_once_with(reload=False)
            fake_srv.wait_for_termination.assert_called_once()

        call_order: list[str] = []
        fake_stub = SimpleNamespace(
            ReportJobProgress=lambda req, **_kw: call_order.append(
                f"progress:{req.current_stage}:{req.progress_pct}"
            ),
            CompleteJob=lambda req, **_kw: call_order.append(f"complete:{req.status}"),
        )
        with (
            patch.object(worker_runner, "_get_backend_stub", return_value=fake_stub),
            patch.object(worker_runner, "sync_rules_version"),
            patch.object(worker_runner, "snapshot_rules", return_value=RulesSnapshot("v1", {})),
            patch.object(
                worker_runner,
                "process_request",
                side_effect=lambda **_kw: (call_order.append("process_request"), {"results": []})[
                    1
                ],
            ),
        ):
            worker_runner._execute_job(
                {
                    "job_id": "j_imm_1",
                    "task": "pipeline",
                    "domain": "market",
                    "skus": [{"name": "A"}],
                }
            )

        self.assertEqual(call_order[0], "progress:embedding:0.0")
        self.assertLess(
            call_order.index("progress:embedding:0.0"), call_order.index("process_request")
        )

    @pytest.mark.usefixtures("loopback")
    def test_ready_probe_and_job_sse_stream(self):
        """GET /ready reflects model readiness (503 vs 200) and /jobs/{id}/stream streams SSE frames."""
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from backend.app import grpc_server as be_grpc
        from backend.main import app

        # Verify Redis-backed progress store + Pub/Sub publish + eviction
        published: list[tuple[str, dict]] = []
        fake_hash: dict[str, dict[str, str]] = {}

        class _FakeSyncPipe:
            def hset(self, k, mapping):
                fake_hash.setdefault(k, {}).update(mapping)

            def expire(self, _k, _ttl):
                pass

            def execute(self):
                pass

        class _FakeSyncRedis:
            def pipeline(self):
                return _FakeSyncPipe()

            def hmget(self, k, *fields):
                d = fake_hash.get(k, {})
                return [d.get(f) for f in fields]

            def publish(self, ch, data):
                published.append((ch, json.loads(data)))

            def delete(self, k):
                fake_hash.pop(k, None)

        with patch.object(be_grpc, "_progress_redis", return_value=_FakeSyncRedis()):
            be_grpc.set_job_progress("j_redis_1", pct=42.5, eta=9, stage="reranking")
            self.assertEqual(be_grpc.get_job_progress_state("j_redis_1"), (42.5, 9))
            be_grpc.publish_job_event({"id": "j_redis_1", "status": "running"})
            self.assertEqual(
                published, [("matchops:jobs:events", {"id": "j_redis_1", "status": "running"})]
            )
            be_grpc.clear_job_progress("j_redis_1")
            self.assertEqual(be_grpc.get_job_progress_state("j_redis_1"), (0.0, None))

        fake_conn_ctx = MagicMock()
        fake_db_engine = MagicMock()
        fake_db_engine.connect.return_value = fake_conn_ctx
        mock_r = MagicMock()
        mock_r.ping.return_value = True

        client = TestClient(app)
        with (
            patch("backend.app.api.routes.get_engine", return_value=fake_db_engine),
            patch("redis.Redis.from_url", return_value=mock_r),
            patch(
                "backend.app.api.routes.get_engine_health",
                return_value={"status": "ok", "all_models_ready": False},
            ),
        ):
            r_not_ready = client.get("/ready")
            self.assertEqual(r_not_ready.status_code, 503)
            self.assertEqual(r_not_ready.json()["status"], "not_ready")

        with (
            patch("backend.app.api.routes.get_engine", return_value=fake_db_engine),
            patch("redis.Redis.from_url", return_value=mock_r),
            patch(
                "backend.app.api.routes.get_engine_health",
                return_value={"status": "ok", "all_models_ready": True},
            ),
        ):
            r_ready = client.get("/ready")
            self.assertEqual(r_ready.status_code, 200)
            self.assertEqual(r_ready.json()["status"], "ready")

        with patch("backend.app.api.endpoints.jobs._single_job", return_value=None):
            self.assertEqual(client.get("/jobs/nonexistent_sse_job/stream").status_code, 404)

        job_snapshot = {
            "id": "test_sse_job_202",
            "status": "completed",
            "current_stage": "done",
            "progress_pct": 100.0,
            "eta_seconds": 0,
        }

        class _FakePubSub:
            async def subscribe(self, _ch):
                return None

            async def get_message(self, **_kw):
                return None

            async def aclose(self):
                return None

        class _FakeAsyncRedis:
            def pubsub(self, **_kw):
                return _FakePubSub()

            async def aclose(self):
                return None

        with (
            patch("backend.app.api.endpoints.jobs._single_job", return_value=job_snapshot),
            patch(
                "backend.app.api.endpoints.jobs._open_job_pubsub",
                return_value=(_FakeAsyncRedis(), _FakePubSub()),
            ),
        ):
            resp = client.get("/jobs/test_sse_job_202/stream")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
            lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
            self.assertEqual(len(lines), 1)
            frame = json.loads(lines[0][6:])
            self.assertEqual(frame["id"], "test_sse_job_202")
            self.assertEqual(frame["status"], "completed")
            self.assertEqual(frame["progress_pct"], 100.0)

    def test_sync_cli_db_mode_avoids_engine_imports(self):
        import builtins
        import sys
        from unittest.mock import MagicMock

        import scripts.sync as sync_mod

        real_import = builtins.__import__
        imported_roots = []

        def _tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "engine" or name.startswith("engine."):
                imported_roots.append(name)
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch.object(
                sys, "argv", ["sync.py", "--db", "sync", "--from-staged", "--keep-staged"]
            ),
            patch.object(sync_mod, "sync_db", MagicMock()) as mock_sync_db,
            patch.object(sync_mod, "_get_sheet_id", return_value="dummy_sheet_id"),
            patch.object(builtins, "__import__", side_effect=_tracking_import),
        ):
            sync_mod.main()

        self.assertEqual(imported_roots, [])
        self.assertEqual(mock_sync_db.call_count, 2)

    def test_grpc_suggest_tags_maps_template_fields(self):
        from unittest.mock import MagicMock

        from engine.grpc_server import EngineInferenceServicer

        servicer = EngineInferenceServicer()
        req = pb2.SuggestRequest(sku_name="Kist Mango Nectar 200ml", domain="market")
        with patch(
            "engine.grpc_server.suggest_tags_from_template",
            return_value={
                "matched": True,
                "base_sku": "Kist Nectar 200ml",
                "suggested_bt": "Juice",
                "suggested_gk": ["Juice", "Fruit Drink"],
                "suggested_region": "Beverages",
            },
        ):
            resp = servicer.SuggestTags(req, MagicMock())

        self.assertEqual(resp.status, "success")
        self.assertTrue(resp.is_match)
        self.assertEqual(resp.template_used, "Kist Nectar 200ml")
        self.assertEqual(resp.suggested_bt, "Juice")
        self.assertEqual(resp.suggested_gk, "Juice, Fruit Drink")
        self.assertEqual(resp.suggested_region, "Beverages")


if __name__ == "__main__":
    unittest.main()
