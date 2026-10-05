"""
SKU MatchOps Backend - Job Dispatcher
Persists job records in the database and dispatches async tasks to the dedicated ML Engine microservice.
"""

import json
import logging
import sqlite3

from backend.app.api.endpoints.engine_callbacks import _job_eta, _job_progress
from backend.app.core.db import DB_PATH, get_next_job_id
from backend.app.schemas.models import BaseRequest
from backend.app.services.engine_client import dispatch_batch_job

logger = logging.getLogger("matchops.backend.worker")


def enqueue_job(request: BaseRequest, task: str) -> dict:
    """
    Registers a new matching/classification job in the DB and dispatches it
    to the dedicated ML Engine microservice.
    """
    if not request.skus:
        raise ValueError("'skus' list is empty.")

    domain = request.domain or "market"
    target_sheet = request.sheet_name or "N/A"
    skus_data = [sku.model_dump() for sku in request.skus]
    input_skus_json = json.dumps(skus_data)

    # Allocate a job ID and write the initial job row in the same transaction, retrying on
    # a rare id collision (two near-simultaneous submissions both computing the same
    # "next id" before either commits) instead of silently falling back to a hardcoded "1".
    max_attempts = 5
    job_id = None
    for attempt in range(1, max_attempts + 1):
        conn = sqlite3.connect(DB_PATH, timeout=60.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            job_id = get_next_job_id(conn)
            conn.execute(
                """
                INSERT INTO jobs (
                    id, batch_id, type, status, current_stage,
                    total_items, completed_items, created_by,
                    started_at, domain, sheet_name, target_sheet, input_skus_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_id,
                    task,
                    "queued",
                    "queued",
                    len(request.skus),
                    0,
                    "system",
                    domain,
                    request.sheet_name,
                    target_sheet,
                    input_skus_json,
                ),
            )
            conn.commit()
            break
        except sqlite3.IntegrityError as e:
            conn.rollback()
            job_id = None
            if attempt == max_attempts:
                logger.error(
                    f"Failed to allocate a unique job id after {max_attempts} attempts: {e}"
                )
                raise RuntimeError(
                    f"Could not allocate a unique job id after {max_attempts} attempts: {e}"
                )
            logger.warning(
                f"Job id collided with an existing row (attempt {attempt}/{max_attempts}); retrying with a new id."
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to insert job into DB: {e}")
            raise RuntimeError(f"Database error registering job: {e}")
        finally:
            conn.close()

    _job_progress[job_id] = 0.0
    _job_eta[job_id] = max(3, int(len(request.skus) * 0.04))

    logger.info(f"[JOB {job_id}] Queued {len(request.skus)} SKUs (task={task}, domain={domain})")

    # 3. Dispatch to ML Engine microservice
    try:
        dispatch_batch_job(job_id=job_id, request=request, task=task)
    except Exception as dispatch_err:
        logger.error(f"[JOB {job_id}] Dispatch to ML Engine failed: {dispatch_err}")
        # Mark as failed in DB
        try:
            conn = sqlite3.connect(DB_PATH, timeout=60.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            conn.execute(
                "UPDATE jobs SET status = 'failed', current_stage = 'failed', error_message = ? WHERE id = ?",
                (str(dispatch_err), job_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        raise RuntimeError(f"Failed to dispatch job to ML Engine: {dispatch_err}")

    return {"job_id": job_id, "status": "queued", "total_skus": len(request.skus)}
