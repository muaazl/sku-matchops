"""
SKU MatchOps Backend - Internal Engine Callbacks API
Receives progress, completion, and failure webhook events from the dedicated ML Engine microservice.
"""

import json
import logging
import sqlite3
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.rules_engine.db.seed_rules import DB_PATH

logger = logging.getLogger("matchops.engine_callbacks")
router = APIRouter(prefix="/api/internal/jobs", tags=["engine_internal"])

# In-memory store for quick polling access
_job_progress: Dict[str, float] = {}
_job_eta: Dict[str, Optional[int]] = {}
_job_stage: Dict[str, str] = {}


class JobProgressPayload(BaseModel):
    current_stage: str
    progress_pct: float
    eta_seconds: Optional[int] = None


class JobCompletePayload(BaseModel):
    status: str = "completed"
    duration_minutes: float
    high_conf: int
    med_conf: int
    low_conf: int
    match_rate: float
    results: List[Dict[str, Any]]
    raw_payload: Optional[Dict[str, Any]] = None
    callback_url: Optional[str] = None


class JobFailPayload(BaseModel):
    status: str = "failed"
    error_message: str
    duration_minutes: float
    callback_url: Optional[str] = None


@router.post("/{job_id}/progress")
def update_progress(job_id: str, payload: JobProgressPayload):
    """Updates job progress and stage in DB and in-memory cache."""
    j_id = str(job_id)
    _job_progress[j_id] = payload.progress_pct
    _job_eta[j_id] = payload.eta_seconds

    # Only perform SQLite disk write when the stage changes
    last_stage = _job_stage.get(j_id)
    if last_stage != payload.current_stage:
        _job_stage[j_id] = payload.current_stage
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                "UPDATE jobs SET status = 'running', current_stage = ?, updated_at = datetime('now') WHERE id = ?",
                (payload.current_stage, j_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update progress for job {job_id} in DB: {e}")

    return {"status": "ok"}


@router.post("/{job_id}/complete")
def complete_job(job_id: str, payload: JobCompletePayload):
    """Saves processed SKUs and final job metrics in DB upon ML Engine completion."""
    j_id = str(job_id)
    _job_progress[j_id] = 100.0
    _job_eta[j_id] = 0

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")

        # Fetch job metadata to get domain and task
        job_row = conn.execute("SELECT domain, type FROM jobs WHERE id = ?", (j_id,)).fetchone()
        domain = job_row[0] if job_row else "market"
        task_lower = (job_row[1] if job_row else "pipeline").lower()

        # Fetch input SKUs from job record to associate names
        input_row = conn.execute("SELECT input_skus_json FROM jobs WHERE id = ?", (j_id,)).fetchone()
        input_skus = []
        if input_row and input_row[0]:
            try:
                input_skus = json.loads(input_row[0])
            except Exception:
                pass

        sku_rows = []
        res_list = payload.results

        for i, res in enumerate(res_list):
            sku_name = input_skus[i].get("name", "") if i < len(input_skus) else ""
            logic_notes = res.get("logic_notes", "")
            matched_catalog_name = res.get("matched_catalog_name", "")
            match_score = res.get("score", 0.0)
            bt_confidence = res.get("bt_confidence", 0.0)
            gk_confidence = res.get("gk_confidence", 0.0)
            region_confidence = res.get("region_confidence", res.get("category_confidence", 0.0))

            if task_lower == "matcher":
                bt = res.get("suggested_bt", "")
                gk = res.get("suggested_gk", "")
                if gk is None or gk == "None":
                    gk = ""
                region = res.get("suggested_region", "")
                conf = res.get("score", 0.0)
                source = "matcher"
                rules = res.get("rules_applied", "")
            elif task_lower == "classifier":
                bt = res.get("suggested_bt", "")
                gk = res.get("suggested_gk", "")
                if gk is None or gk == "None":
                    gk = ""
                region = res.get("suggested_region", "")
                conf = res.get("bt_confidence", 0.0)
                source = "classifier"
                rules = res.get("rules_applied", "")
            else: # pipeline
                bt = res.get("suggested_bt", "")
                gk = res.get("suggested_gk", "")
                if gk is None or gk == "None":
                    gk = ""
                region = res.get("suggested_region", "")
                conf = res.get("bt_confidence", res.get("score", 0.0))
                source = res.get("pipeline_source", "")
                rules = res.get("rules_applied", "")

            sku_id = str(uuid.uuid4())
            gk_val = gk
            if isinstance(gk, str):
                gk_val = [x.strip() for x in gk.split(",") if x.strip()]

            sku_rows.append((
                sku_id, j_id, sku_name, domain, bt,
                json.dumps(gk_val) if gk_val else "[]",
                region, conf, source, rules, logic_notes,
                matched_catalog_name, match_score, bt_confidence,
                gk_confidence, region_confidence
            ))

        if sku_rows:
            conn.executemany(
                """
                INSERT INTO processed_skus (
                    id, batch_id, sku_name, domain, bt, gk_json, region,
                    confidence, match_source, rules_applied_json, logic_notes,
                    matched_catalog_name, match_score, bt_confidence,
                    gk_confidence, region_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                sku_rows
            )

        conn.execute(
            """
            UPDATE jobs SET
                status = 'completed',
                current_stage = 'done',
                completed_items = ?,
                completed_at = datetime('now'),
                duration_minutes = ?,
                high_conf = ?,
                med_conf = ?,
                low_conf = ?,
                match_rate = ?
            WHERE id = ?
            """,
            (len(res_list), payload.duration_minutes, payload.high_conf, payload.med_conf, payload.low_conf, payload.match_rate, j_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Successfully recorded completion for job {job_id} ({len(sku_rows)} SKUs saved).")
    except Exception as e:
        logger.error(f"Failed to record job {job_id} completion in DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}


@router.post("/{job_id}/fail")
def fail_job(job_id: str, payload: JobFailPayload):
    """Updates job to failed state with error message."""
    j_id = str(job_id)
    _job_eta[j_id] = None

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            UPDATE jobs SET
                status = 'failed',
                current_stage = 'failed',
                error_message = ?,
                duration_minutes = ?,
                completed_at = datetime('now')
            WHERE id = ?
            """,
            (payload.error_message, payload.duration_minutes, j_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Recorded failure for job {job_id}: {payload.error_message}")
    except Exception as e:
        logger.error(f"Failed to record job {job_id} failure in DB: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}
