"""
SKU MatchOps Backend - Job Dispatcher
Persists job records in the database and dispatches async tasks to the dedicated ML Engine microservice.
"""

import json
import logging
import sqlite3
import uuid

from engine.rules_engine.db.seed_rules import DB_PATH
from backend.app.core.db import get_next_job_id
from backend.app.schemas.models import BaseRequest
from backend.app.services.engine_client import dispatch_batch_job
from backend.app.api.endpoints.engine_callbacks import _job_progress, _job_eta

logger = logging.getLogger("matchops.backend.worker")

# In-memory status caches for fast endpoint lookups
_jobs: dict[str, str] = {}


def enqueue_job(request: BaseRequest, task: str) -> dict:
    """
    Registers a new matching/classification job in the DB and dispatches it
    to the dedicated ML Engine microservice.
    """
    if not request.skus:
        raise ValueError("'skus' list is empty.")

    domain = request.domain or "market"

    # 1. Get sequential job ID
    try:
        conn = sqlite3.connect(DB_PATH)
        job_id = get_next_job_id(conn)
        conn.close()
    except Exception:
        job_id = "1"

    _jobs[job_id] = "queued"
    _job_progress[job_id] = 0.0
    _job_eta[job_id] = max(3, int(len(request.skus) * 0.04))

    target_sheet = request.sheet_name or "N/A"
    skus_data = [sku.model_dump() for sku in request.skus]
    input_skus_json = json.dumps(skus_data)

    # 2. Write initial job state to SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            INSERT INTO jobs (
                id, batch_id, type, status, current_stage,
                total_items, completed_items, created_by,
                started_at, domain, sheet_name, target_sheet, input_skus_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?)
            """,
            (job_id, job_id, task, 'queued', 'queued', len(request.skus), 0, 'system', domain, request.sheet_name, target_sheet, input_skus_json)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to insert job {job_id} into DB: {e}")
        raise RuntimeError(f"Database error registering job: {e}")

    logger.info(f"[JOB {job_id}] Queued {len(request.skus)} SKUs (task={task}, domain={domain})")

    # 3. Dispatch to ML Engine microservice
    try:
        dispatch_batch_job(job_id=job_id, request=request, task=task)
    except Exception as dispatch_err:
        logger.error(f"[JOB {job_id}] Dispatch to ML Engine failed: {dispatch_err}")
        # Mark as failed in DB
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                "UPDATE jobs SET status = 'failed', current_stage = 'failed', error_message = ? WHERE id = ?",
                (str(dispatch_err), job_id)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        raise RuntimeError(f"Failed to dispatch job to ML Engine: {dispatch_err}")

    return {"job_id": job_id, "status": "queued", "total_skus": len(request.skus)}


def log_outbound_request(url: str, method: str, payload: dict, response_status: int, response_text: str, duration_ms: int):
    """Logs outbound HTTP requests (e.g. Google Sheets webhooks) to SQLite."""
    try:
        req_id = str(uuid.uuid4())
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            INSERT INTO api_requests (id, method, path, payload_json_redacted, response_json, status_code, duration_ms, ip_address, headers_json, query_params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req_id,
                method,
                "Google Sheets Callback",
                json.dumps(payload),
                response_text,
                response_status,
                duration_ms,
                "outbound",
                json.dumps({"Content-Type": "application/json"}),
                json.dumps({"callback_url": url})
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log outbound request to DB: {e}")
