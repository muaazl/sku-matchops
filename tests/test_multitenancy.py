import json
import unittest
from unittest.mock import MagicMock, patch

from backend.app.core.security import Principal
from backend.app.schemas.models import MatchRequest, SKUItem


class TestMultiTenancyEnforcement(unittest.TestCase):
    def test_routes_propagate_tenant_id_to_enqueue_job(self):
        """TEN-01: /match, /classify, /pipeline forward principal.tenant_id to enqueue_job."""
        from backend.app.api.routes import classify, match, pipeline

        req = MatchRequest(
            skus=[SKUItem(name="Test Biscuit", price=2.5, description="Snack", category="Food")],
            domain="market",
            callback_url="http://example.com/callback",
        )
        principal = Principal(
            username="tenant_user",
            role="reviewer",
            tenant_id="tenant-alpha",
            home_tenant="tenant-alpha",
        )

        with patch("backend.app.api.routes.enqueue_job") as mock_enqueue:
            mock_enqueue.return_value = {"job_id": "job_1", "status": "queued", "total_skus": 1}

            # Test /match
            match(req, principal=principal)
            mock_enqueue.assert_called_with(req, "matcher", tenant_id="tenant-alpha")

            # Test /classify
            classify(req, principal=principal)
            mock_enqueue.assert_called_with(req, "classifier", tenant_id="tenant-alpha")

            # Test /pipeline
            pipeline(req, principal=principal)
            mock_enqueue.assert_called_with(req, "pipeline", tenant_id="tenant-alpha")

    def test_rules_cache_isolation_per_tenant(self):
        """TEN-02: Rules cache keys and versions must be namespaced per tenant."""
        from backend.app.services.rules_service import (
            get_rule_version,
            rules_data_key,
            rules_version_key,
        )
        from engine.rules_engine import loader

        self.assertEqual(rules_data_key("tenant-a"), "matchops:rules:data:tenant-a")
        self.assertEqual(rules_version_key("tenant-a"), "matchops:rules:version:tenant-a")

        # Mock Redis client
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        rules_tenant_a = [
            {
                "id": 1,
                "rule_id": "rule_a",
                "domain": "market",
                "priority": 10,
                "tenant_id": "tenant-a",
                "conditions": [],
                "actions": [],
            }
        ]

        rules_tenant_b = [
            {
                "id": 2,
                "rule_id": "rule_b",
                "domain": "market",
                "priority": 20,
                "tenant_id": "tenant-b",
                "conditions": [],
                "actions": [],
            }
        ]

        # 1. Test loader publish_rules_snapshot with tenant_id
        with patch.object(loader, "_redis", return_value=mock_redis):
            loader.publish_rules_snapshot(rules_tenant_a, tenant_id="tenant-a")
            pipe_calls = [str(call) for call in mock_pipe.mock_calls]
            self.assertTrue(any("matchops:rules:data:tenant-a" in c for c in pipe_calls))
            self.assertTrue(any("matchops:rules:version:tenant-a" in c for c in pipe_calls))

            # Publish for tenant-b
            loader.publish_rules_snapshot(rules_tenant_b, tenant_id="tenant-b")

        # 2. Verify snapshot and version are strictly isolated between tenants
        ver_a = get_rule_version("tenant-a")
        ver_b = get_rule_version("tenant-b")
        self.assertNotEqual(ver_a, ver_b)

        snap_a = loader.snapshot_rules("tenant-a")
        snap_b = loader.snapshot_rules("tenant-b")
        self.assertEqual(len(snap_a.rules("market")), 1)
        self.assertEqual(snap_a.rules("market")[0].rule_id, "rule_a")
        self.assertEqual(len(snap_b.rules("market")), 1)
        self.assertEqual(snap_b.rules("market")[0].rule_id, "rule_b")

    def test_dlq_partitioning_and_isolation(self):
        """TEN-03: TaskQueue DLQ items, replay, and drop must isolate by tenant."""
        from backend.app.services.task_queue import TaskQueue

        mock_redis = MagicMock()
        queue = TaskQueue(mock_redis)

        dlq_entry_tenant_a = json.dumps(
            {
                "task_id": "task_a",
                "job_id": "job_a",
                "queue": "bulk",
                "payload": {"tenant_id": "tenant-a"},
            }
        )
        dlq_entry_tenant_b = json.dumps(
            {
                "task_id": "task_b",
                "job_id": "job_b",
                "queue": "bulk",
                "payload": {"tenant_id": "tenant-b"},
            }
        )

        # 1. dlq_items queries tenant-specific partition first
        mock_redis.lrange.return_value = [dlq_entry_tenant_a]
        items_a = queue.dlq_items(tenant_id="tenant-a")
        self.assertEqual(len(items_a), 1)
        self.assertEqual(items_a[0]["task_id"], "task_a")
        mock_redis.lrange.assert_called_with("matchops:queue:dlq:tenant-a", 0, 99)

        # 2. Replay of task belonging to another tenant is rejected
        mock_redis.lrange.return_value = [dlq_entry_tenant_b]
        replayed = queue.replay("task_b", tenant_id="tenant-a")
        self.assertIsNone(replayed)

        # 3. Drop of task belonging to another tenant is rejected
        mock_redis.lrange.return_value = [dlq_entry_tenant_b]
        dropped = queue.drop_dlq("task_b", tenant_id="tenant-a")
        self.assertFalse(dropped)

    def test_jobs_stats_cross_tenant_request_leak_prevented(self):
        """TEN-04: Non-platform admins must only query their tenant's api_requests."""
        from backend.app.api.endpoints.jobs import get_dashboard_stats

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.fetchone.return_value = (0, 0, 0)
        mock_result.scalar.return_value = 42
        mock_db.execute.return_value = mock_result

        # 1. Tenant reviewer
        tenant_principal = Principal(
            username="tenant_user",
            role="reviewer",
            tenant_id="tenant-acme",
            home_tenant="tenant-acme",
        )
        res = get_dashboard_stats(
            domain="all", timeframe="7d", db=mock_db, principal=tenant_principal
        )
        self.assertIn("stats", res)

        # Verify SQL queries executed on db
        executed_sqls = [
            str(call[0][0].text)
            for call in mock_db.execute.call_args_list
            if hasattr(call[0][0], "text")
        ]
        # Must filter api_requests by tenant_id
        req_query = next((q for q in executed_sqls if "FROM api_requests" in q), "")
        self.assertIn("tenant_id = :tenant_id", req_query)

        # 2. Platform admin can query system-wide
        mock_db.reset_mock()
        mock_db.execute.return_value = mock_result
        platform_admin = Principal(
            username="superadmin",
            role="admin",
            tenant_id="default",
            home_tenant="default",
        )
        _ = get_dashboard_stats(domain="all", timeframe="7d", db=mock_db, principal=platform_admin)
        executed_sqls_admin = [
            str(call[0][0].text)
            for call in mock_db.execute.call_args_list
            if hasattr(call[0][0], "text")
        ]
        req_query_admin = next((q for q in executed_sqls_admin if "FROM api_requests" in q), "")
        self.assertNotIn("tenant_id = :tenant_id", req_query_admin)

    def test_principal_resolution_binds_tenant_context(self):
        """TEN-04 & OBS-02: get_principal binds principal.tenant_id to request.state and TENANT_ID."""
        from starlette.requests import Request
        from backend.app.core.observability import get_tenant_id
        from backend.app.core.security import get_principal, create_access_token

        mock_req = MagicMock(spec=Request)
        mock_req.state = MagicMock()
        token = create_access_token("alice", "reviewer", "tenant-wonderland")

        try:
            principal = get_principal(
                request=mock_req,
                bearer=MagicMock(credentials=token),
                oauth2_token=None,
                api_key=None,
                x_tenant_id=None,
            )
            self.assertEqual(principal.tenant_id, "tenant-wonderland")
            self.assertEqual(mock_req.state.tenant_id, "tenant-wonderland")
            self.assertEqual(get_tenant_id(), "tenant-wonderland")
        finally:
            from backend.app.core.observability import TENANT_ID

            TENANT_ID.set("-")

    def test_dlq_replay_resolves_batch_id_from_entry(self):
        """TEN-03: replay_dead_letter updates batches table using entry.batch_id."""
        from backend.app.api.endpoints.queue import replay_dead_letter

        mock_queue = MagicMock()
        mock_queue.dlq_items.return_value = [
            {
                "task_id": "task_dlq_1",
                "job_id": "job_123",
                "batch_id": "batch_456",
                "payload": {"tenant_id": "tenant-corp", "batch_id": "batch_456"},
            }
        ]
        mock_queue.replay.return_value = "new_task_999"

        mock_db = MagicMock()
        res_ok = MagicMock()
        res_ok.rowcount = 1
        mock_db.execute.return_value = res_ok

        principal = Principal(
            username="admin", role="admin", tenant_id="tenant-corp", home_tenant="tenant-corp"
        )
        out = replay_dead_letter(
            task_id="task_dlq_1",
            principal=principal,
            queue=mock_queue,
            db=mock_db,
        )
        self.assertEqual(out["job_id"], "job_123")

        # Verify UPDATE batches used batch_456 rather than job_123
        batch_update_calls = [
            call
            for call in mock_db.execute.call_args_list
            if "UPDATE batches" in str(call[0][0].text)
        ]
        self.assertTrue(batch_update_calls)
        params = batch_update_calls[0][0][1]
        self.assertEqual(params["id"], "batch_456")
