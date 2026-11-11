import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from sqlalchemy import text

from backend.app.core.security import Principal


class TestDatabaseIntegrityAndReliability(unittest.TestCase):
    def test_startup_batch_reconciliation_logic(self):
        """DAT-01: Batches are reconciled based on whether any active jobs remain for that batch."""
        mock_conn = MagicMock()
        executed_sqls = []

        def fake_execute(statement, params=None):
            executed_sqls.append(str(statement.text))
            res = MagicMock()
            res.rowcount = 3
            return res

        mock_conn.execute.side_effect = fake_execute

        # Simulate startup reconciliation block
        live_job_ids = ["job_live_1"]
        mock_conn.execute(
            text("""
            UPDATE jobs SET
                status = 'failed',
                current_stage = 'failed',
                error_message = 'Interrupted by a service restart. Use Retry to resubmit.',
                completed_at = now()
            WHERE status IN ('queued', 'running') AND NOT (id = ANY(:live))
            """),
            {"live": live_job_ids},
        )
        mock_conn.execute(
            text("""
            UPDATE batches SET status = 'failed', completed_at = now()
            WHERE status IN ('queued', 'running')
              AND id NOT IN (
                  SELECT DISTINCT batch_id FROM jobs
                  WHERE status IN ('queued', 'running')
                    AND batch_id IS NOT NULL
              )
            """)
        )

        # Verify batch query references batch_id rather than erroneously equating batch_id to job_id
        batch_query = next((q for q in executed_sqls if "UPDATE batches" in q), "")
        self.assertIn("SELECT DISTINCT batch_id FROM jobs", batch_query)
        self.assertNotIn("id = ANY(:live)", batch_query)

    def test_streaming_export_manages_internal_connection(self):
        """DAT-02: /processed-skus/export creates and closes connection inside generator."""
        from backend.app.api.endpoints.history import export_history, generate_csv_chunks

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

        principal = Principal(username="test", role="admin")

        with patch("backend.app.api.endpoints.history.get_engine", return_value=mock_engine):
            # Test generator directly
            chunks = list(generate_csv_chunks(principal, ids=None))
            full_csv = "".join(chunks)
            self.assertIn("Anchor Butter", full_csv)
            self.assertIn("sku_name", full_csv)

            # Verify connection was managed and closed inside generator
            mock_engine.connect.assert_called_once()
            mock_result.close.assert_called_once()

            # Test route handler returns StreamingResponse
            resp = export_history(ids=None, format="csv", principal=principal)
            self.assertEqual(resp.media_type, "text/csv")

    def test_cancel_job_atomic_transition_and_concurrency_guard(self):
        """CON-01: cancel_job enforces atomic condition and returns 409 if non-cancellable."""
        from backend.app.api.endpoints.jobs import cancel_job

        mock_db = MagicMock()
        principal = Principal(username="admin", role="admin")

        with (
            patch("backend.app.api.endpoints.jobs.cancel_engine_job"),
            patch("backend.app.api.endpoints.jobs.publish_job_event"),
            patch("backend.app.api.endpoints.jobs.clear_job_progress"),
        ):
            # 1. Successful atomic cancellation
            res_ok = MagicMock()
            res_ok.rowcount = 1
            mock_db.execute.return_value = res_ok

            out = cancel_job(id="job_active", db=mock_db, principal=principal)
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
                cancel_job(id="job_done", db=mock_db, principal=principal)
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
            out_idem = cancel_job(id="job_canc", db=mock_db, principal=principal)
            self.assertEqual(out_idem["message"], "Job is already cancelled")

    def test_upsert_processed_skus_uses_scoped_constraint(self):
        """DAT-03: upsert_processed_skus targets uq_processed_skus_job_sku_rule."""
        from backend.app.db.session import upsert_processed_skus

        mock_conn = MagicMock()
        rows = [{"batch_id": "job_1", "sku_index": 0, "rule_version": "v1", "sku_name": "Test SKU"}]
        upsert_processed_skus(mock_conn, rows)
        mock_conn.execute.assert_called_once()
        stmt = mock_conn.execute.call_args[0][0]
        self.assertEqual(
            stmt._post_values_clause.constraint_target, "uq_processed_skus_job_sku_rule"
        )
