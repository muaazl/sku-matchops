import json
import unittest
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.api.routes import api_router
from backend.app.core.db import get_engine
from backend.app.schemas.models import BaseRequest, SKUItem
from backend.app.services.worker import enqueue_job
from engine.pipeline.persistence import record_job_completion

app = FastAPI()
app.include_router(api_router)
client = TestClient(app)


def _delete_job(job_id: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job_id})
        conn.execute(text("DELETE FROM processed_skus WHERE batch_id = :id"), {"id": job_id})


class TestDecoupledEngineIntegration(unittest.TestCase):
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
        # 1. Create a dummy queued job in DB
        job_id = "test_cb_job_99"
        _delete_job(job_id)
        with get_engine().begin() as conn:
            conn.execute(
                text("""
                INSERT INTO jobs (
                    id, batch_id, type, status, current_stage, total_items, completed_items, created_by, started_at, domain, input_skus_json
                ) VALUES (:id, :id, 'pipeline', 'queued', 'queued', 1, 0, 'test', now(), 'market', :skus)
                """),
                {"id": job_id, "skus": json.dumps([{"name": "Anchor Butter 200g"}])},
            )

        # 2. Engine reports progress
        prog_resp = client.post(
            f"/api/internal/jobs/{job_id}/progress",
            json={"current_stage": "matching", "progress_pct": 50.0, "eta_seconds": 10},
        )
        self.assertEqual(prog_resp.status_code, 200)

        # Verify progress reflected in GET /jobs/{id}
        job_get = client.get(f"/jobs/{job_id}")
        self.assertEqual(job_get.status_code, 200)
        job_data = job_get.json()
        self.assertEqual(job_data["status"], "running")
        self.assertEqual(job_data["current_stage"], "matching")
        self.assertEqual(job_data["progress_pct"], 50.0)

        # 3. Engine persists results directly, then notifies the gateway
        record_job_completion(
            job_id=job_id,
            domain="market",
            task="pipeline",
            results=[
                {
                    "matched_catalog_name": "Anchor Butter Salted 200g",
                    "score": 92.5,
                    "status": "High Confidence",
                    "logic_notes": "Direct catalog match",
                    "suggested_bt": "Butter",
                    "suggested_gk": "Butter, Dairy",
                    "suggested_region": "Dairy",
                }
            ],
            rule_version="test",
            duration_minutes=0.25,
            high_conf=1,
            med_conf=0,
            low_conf=0,
            match_rate=100.0,
        )
        complete_resp = client.post(
            f"/api/internal/jobs/{job_id}/complete", json={"status": "completed"}
        )
        self.assertEqual(complete_resp.status_code, 200)

        # A late progress POST must not resurrect the finalized job
        client.post(
            f"/api/internal/jobs/{job_id}/progress",
            json={"current_stage": "late_stage", "progress_pct": 99.0},
        )

        # Verify completed state and processed SKU row
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

        _delete_job(job_id)


if __name__ == "__main__":
    unittest.main()
