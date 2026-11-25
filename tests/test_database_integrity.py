import unittest
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import text


@pytest.mark.usefixtures("pg")
class TestDatabaseIntegrityAndReliability(unittest.TestCase):
    def test_startup_batch_reconciliation_logic(self):
        """DAT-01: startup_event marks orphaned queued/running jobs and batches failed while keeping live jobs."""
        from backend.core.database import get_engine
        from backend.main import startup_event

        with get_engine().begin() as conn:
            conn.execute(text("TRUNCATE processed_skus, jobs, batches CASCADE"))
            conn.execute(
                text(
                    "INSERT INTO batches (id, filename, status, total_skus) "
                    "VALUES ('b_dead', 'dead.csv', 'running', 5), "
                    "('b_alive', 'alive.csv', 'running', 5)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO jobs (id, domain, type, status, current_stage, batch_id) "
                    "VALUES ('job_dead', 'food', 'matcher', 'running', 'processing', 'b_dead'), "
                    "('job_live', 'food', 'matcher', 'queued', 'queued', 'b_alive')"
                )
            )

        fake_queue = MagicMock()
        fake_queue.ping_broker.return_value = True
        fake_queue.list_tracked_job_ids.return_value = ["job_live"]

        try:
            with (
                patch("backend.main.ensure_backend_grpc_server"),
                patch("backend.repositories.database_init_repository.initialize_database_schema"),
                patch("backend.repositories.database_init_repository.verify_required_tables"),
                patch("backend.services.rule_service.sync_rules_everywhere"),
                patch(
                    "backend.repositories.task_queue_repository.get_task_queue",
                    return_value=fake_queue,
                ),
            ):
                startup_event()

            with get_engine().connect() as conn:
                jobs = {
                    row[0]: (row[1], row[2], row[3])
                    for row in conn.execute(
                        text("SELECT id, status, current_stage, error_message FROM jobs")
                    ).fetchall()
                }
                batches = {
                    row[0]: row[1]
                    for row in conn.execute(text("SELECT id, status FROM batches")).fetchall()
                }

            self.assertEqual(jobs["job_dead"][0], "failed")
            self.assertEqual(jobs["job_dead"][1], "failed")
            self.assertIn("Use Retry", jobs["job_dead"][2])
            self.assertEqual(jobs["job_live"][0], "queued")
            self.assertEqual(batches["b_dead"], "failed")
            self.assertEqual(batches["b_alive"], "running")
        finally:
            with get_engine().begin() as conn:
                conn.execute(text("TRUNCATE processed_skus, jobs, batches CASCADE"))

    def test_streaming_export_manages_internal_connection(self):
        """DAT-02: /processed-skus/export creates and closes connection inside generator."""
        from backend.api.endpoints.history_router import export_processed_skus
        from backend.services.history_service import generate_csv_chunks

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_result = MagicMock()
        mock_mapping = MagicMock()
        mock_mapping.partitions.return_value = [
            [{"id": "sku_1", "sku_name": "Anchor Butter", "price": 450.0}]
        ]
        mock_result.mappings.return_value = mock_mapping
        mock_conn.execute.return_value = mock_result
        mock_conn.execution_options.return_value = mock_conn

        with patch("backend.repositories.sku_repository.get_engine", return_value=mock_engine):
            # Test generator directly
            chunks = list(generate_csv_chunks(ids=None))
            full_csv = "".join(chunks)
            self.assertIn("Anchor Butter", full_csv)
            self.assertIn("sku_name", full_csv)

            # Verify connection was managed and closed inside generator
            mock_engine.connect.assert_called_once()
            mock_result.close.assert_called_once()

            # Test route handler returns StreamingResponse
            resp = export_processed_skus(ids=None, format="csv")
            self.assertEqual(resp.media_type, "text/csv")

    def test_cancel_job_atomic_transition_and_concurrency_guard(self):
        """CON-01: cancel_job enforces atomic condition and returns 409 if non-cancellable."""
        from backend.api.endpoints.job_router import cancel_job

        mock_db = MagicMock()
        mock_engine = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_db

        with (
            patch("backend.repositories.job_repository.get_engine", return_value=mock_engine),
            patch("backend.services.engine_gateway_service.cancel_engine_job"),
            patch("backend.repositories.job_progress_repository.publish_job_event"),
            patch("backend.repositories.job_progress_repository.clear_job_progress"),
        ):
            # 1. Successful atomic cancellation
            res_ok = MagicMock()
            res_ok.rowcount = 1
            mock_db.execute.return_value = res_ok

            out = cancel_job(job_id="job_active")
            self.assertEqual(out["message"], "Cancellation completed")

            # 2. Job already completed -> res.rowcount == 0, fetch shows status='completed' -> 409 Conflict
            res_zero = MagicMock()
            res_zero.rowcount = 0
            mock_mapping = MagicMock()
            mock_mapping.fetchone.return_value = {"id": "job_done", "status": "completed"}
            res_zero.mappings.return_value = mock_mapping

            def execute_side_effect(stmt, params=None):
                sql = str(stmt.text) if hasattr(stmt, "text") else str(stmt)
                if "UPDATE jobs SET status = 'cancelled'" in sql:
                    return res_zero
                if "SELECT" in sql:
                    return res_zero
                return res_zero

            mock_db.execute.side_effect = execute_side_effect

            with self.assertRaises(HTTPException) as ctx:
                cancel_job(job_id="job_done")
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertIn("Cannot cancel job with status 'completed'", ctx.exception.detail)

            # 3. Job already cancelled -> idempotent success
            mock_mapping_cancelled = MagicMock()
            mock_mapping_cancelled.fetchone.return_value = {"id": "job_canc", "status": "cancelled"}
            res_canc = MagicMock()
            res_canc.rowcount = 0
            res_canc.mappings.return_value = mock_mapping_cancelled

            def execute_cancelled(stmt, params=None):
                sql = str(stmt.text) if hasattr(stmt, "text") else str(stmt)
                if "UPDATE jobs SET status = 'cancelled'" in sql:
                    return res_canc
                if "SELECT" in sql:
                    return res_canc
                return res_canc

            mock_db.execute.side_effect = execute_cancelled
            out_idem = cancel_job(job_id="job_canc")
            self.assertEqual(out_idem["message"], "Job is already cancelled")

    def test_upsert_processed_skus_uses_scoped_constraint(self):
        """DAT-03: upsert_processed_skus targets uq_processed_skus_job_sku_rule."""
        from backend.repositories.sku_repository import upsert_processed_skus

        mock_conn = MagicMock()
        rows = [{"batch_id": "job_1", "sku_index": 0, "rule_version": "v1", "sku_name": "Test SKU"}]
        upsert_processed_skus(mock_conn, rows)
        mock_conn.execute.assert_called_once()
        stmt = mock_conn.execute.call_args[0][0]
        self.assertEqual(
            stmt._post_values_clause.constraint_target, "uq_processed_skus_job_sku_rule"
        )
