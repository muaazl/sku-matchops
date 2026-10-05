import json
import sqlite3
import logging
from fastapi import APIRouter, Depends, HTTPException
from backend.app.core.db import get_db_connection, get_next_job_id
from engine.config import DB_PATH
from backend.app.schemas.models import UploadRequest, EnqueueJobResponse
from backend.app.services.engine_client import dispatch_upload_job

router = APIRouter()
logger = logging.getLogger("matchops.api.upload")

@router.post("/upload", response_model=EnqueueJobResponse)
def trigger_upload(request: UploadRequest, db: sqlite3.Connection = Depends(get_db_connection)):
    if not request.skus:
        raise HTTPException(status_code=400, detail="'skus' list is empty.")

    domain = request.domain or "market"
    input_skus_json = json.dumps(request.skus)

    max_attempts = 5
    job_id = None
    
    # We use a new connection here to handle the get_next_job_id retry logic 
    # similar to the worker enqueue_job logic.
    for attempt in range(1, max_attempts + 1):
        conn = sqlite3.connect(DB_PATH, timeout=60.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            job_id = get_next_job_id(conn)
            
            # Using target_sheet to store the outlet_id_or_name as requested
            conn.execute(
                """
                INSERT INTO jobs (
                    id, batch_id, type, status, current_stage,
                    total_items, completed_items, created_by,
                    started_at, domain, target_sheet, input_skus_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
                """,
                (job_id, job_id, 'upload', 'queued', 'queued', len(request.skus), 0, 'system', domain, request.outlet_id_or_name, input_skus_json)
            )
            conn.commit()
            break
        except sqlite3.IntegrityError as e:
            conn.rollback()
            job_id = None
            if attempt == max_attempts:
                logger.error(f"Failed to allocate unique job id for upload: {e}")
                raise HTTPException(status_code=500, detail="Could not allocate job id.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to insert upload job into DB: {e}")
            raise HTTPException(status_code=500, detail="Database error registering job.")
        finally:
            conn.close()

    logger.info(f"[JOB {job_id}] Queued upload for {len(request.skus)} SKUs to outlet {request.outlet_id_or_name}")

    try:
        dispatch_upload_job(job_id=job_id, request=request, task="upload")
    except Exception as dispatch_err:
        logger.error(f"[JOB {job_id}] Dispatch to ML Engine failed: {dispatch_err}")
        try:
            conn = sqlite3.connect(DB_PATH, timeout=60.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                "UPDATE jobs SET status = 'failed', current_stage = 'failed', error_message = ? WHERE id = ?",
                (str(dispatch_err), job_id)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to dispatch job to ML Engine: {dispatch_err}")

    return {"job_id": job_id, "status": "queued", "total_skus": len(request.skus)}
