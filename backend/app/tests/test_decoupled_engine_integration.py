import json
import sqlite3
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import api_router
from backend.app.schemas.models import BaseRequest, SKUItem
from backend.app.services.worker import enqueue_job
from engine.rules_engine.db.seed_rules import DB_PATH

app = FastAPI()
app.include_router(api_router)
client = TestClient(app)


class TestDecoupledEngineIntegration(unittest.TestCase):

    @patch("backend.app.api.routes.get_engine_health")
    def test_health_with_engine_status(self, mock_health):
        mock_health.return_value = {
            "status": "ok",
            "service": "matchops-engine",
            "loaded_domains": ["market", "food"],
            "all_models_ready": True
        }
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "matchops-backend")
        self.assertIn("loaded_domains", data)
        self.assertEqual(data["loaded_domains"], ["market", "food"])

    @patch("backend.app.services.worker.dispatch_batch_job")
    def test_enqueue_job_dispatches_to_engine(self, mock_dispatch):
        mock_dispatch.return_value = {"status": "queued", "job_id": "test_job_100"}
        req = BaseRequest(
            skus=[SKUItem(name="Test Item 1", price=10.0, description="Desc", category="Cat")],
            domain="market",
            callback_url="",
            sheet_name="Sheet1"
        )
        res = enqueue_job(req, "pipeline")
        self.assertIn("job_id", res)
        self.assertEqual(res["status"], "queued")
        self.assertEqual(res["total_skus"], 1)
        mock_dispatch.assert_called_once()

        # Clean up the job row inserted into the real DB
        job_id = res["job_id"]
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.execute("DELETE FROM processed_skus WHERE batch_id = ?", (job_id,))
        conn.commit()
        conn.close()

    def test_engine_callbacks_lifecycle(self):
        # 1. Create a dummy queued job in DB
        job_id = "test_cb_job_99"
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs (
                id, batch_id, type, status, current_stage, total_items, completed_items, created_by, started_at, domain, input_skus_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)
            """,
            (job_id, job_id, "pipeline", "queued", "queued", 1, 0, "test", "market", json.dumps([{"name": "Anchor Butter 200g"}]))
        )
        conn.commit()
        conn.close()

        # 2. Engine reports progress
        prog_resp = client.post(
            f"/api/internal/jobs/{job_id}/progress",
            json={"current_stage": "matching", "progress_pct": 50.0, "eta_seconds": 10}
        )
        self.assertEqual(prog_resp.status_code, 200)

        # Verify progress reflected in GET /jobs/{id}
        job_get = client.get(f"/jobs/{job_id}")
        self.assertEqual(job_get.status_code, 200)
        job_data = job_get.json()
        self.assertEqual(job_data["status"], "running")
        self.assertEqual(job_data["current_stage"], "matching")
        self.assertEqual(job_data["progress_pct"], 50.0)

        # 3. Engine reports completion
        complete_resp = client.post(
            f"/api/internal/jobs/{job_id}/complete",
            json={
                "status": "completed",
                "duration_minutes": 0.25,
                "high_conf": 1,
                "med_conf": 0,
                "low_conf": 0,
                "match_rate": 100.0,
                "results": [
                    {
                        "matched_catalog_name": "Anchor Butter Salted 200g",
                        "score": 92.5,
                        "status": "High Confidence",
                        "logic_notes": "Direct catalog match",
                        "suggested_bt": "Butter",
                        "suggested_gk": "Butter, Dairy",
                        "suggested_region": "Dairy"
                    }
                ]
            }
        )
        self.assertEqual(complete_resp.status_code, 200)

        # Verify completed state and processed SKU row
        job_completed = client.get(f"/jobs/{job_id}")
        self.assertEqual(job_completed.status_code, 200)
        comp_data = job_completed.json()
        self.assertEqual(comp_data["status"], "completed")
        self.assertEqual(comp_data["current_stage"], "done")
        self.assertEqual(comp_data["progress_pct"], 100.0)
        self.assertEqual(comp_data["match_rate"], 100.0)

        # Clean up test row
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.execute("DELETE FROM processed_skus WHERE batch_id = ?", (job_id,))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    unittest.main()
