"""
SKU MatchOps Backend - Job Dispatcher
Persists job records in the database and dispatches async tasks to the dedicated ML Engine microservice.
"""

import json
import logging
from collections.abc import Callable

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import DEFAULT_TENANT
from backend.app.db.session import get_engine, get_next_job_id
from backend.app.grpc_server import clear_job_progress, publish_job_event, set_job_progress
from backend.app.schemas.models import BaseRequest
from backend.app.services.engine_client import dispatch_batch_job

logger = logging.getLogger("matchops.backend.worker")


def register_and_dispatch_job(
    *,
    job_type: str,
    domain: str,
    total_items: int,
    input_skus_json: str,
    tenant_id: str,
    sheet_name: str | None,
    target_sheet: str | None,
    dispatch_fn: Callable[[str], object],
) -> dict:
    """Allocates a sequential job row, tracks initial progress, and dispatches to the Engine."""
    max_attempts = 5
    job_id = None
    for attempt in range(1, max_attempts + 1):
        conn = get_engine().connect()
        try:
            job_id = get_next_job_id(conn)
            conn.execute(
                text("""
                INSERT INTO jobs (
                    id, batch_id, type, status, current_stage,
                    total_items, completed_items, created_by,
                    started_at, domain, sheet_name, target_sheet, input_skus_json, tenant_id
                ) VALUES (
                    :id, :id, :type, 'queued', 'queued', :total_items, 0, 'system',
                    now(), :domain, :sheet_name, :target_sheet, :input_skus_json, :tenant_id
                )
                """),
                {
                    "id": job_id,
                    "type": job_type,
                    "total_items": total_items,
                    "domain": domain,
                    "sheet_name": sheet_name,
                    "target_sheet": target_sheet,
                    "input_skus_json": input_skus_json,
                    "tenant_id": tenant_id,
                },
            )
            conn.commit()
            break
        except IntegrityError as e:
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

    assert job_id is not None
    initial_eta = max(3, int(total_items * 0.04))
    set_job_progress(job_id, pct=0.0, eta=initial_eta, stage="queued", tenant_id=tenant_id)
    publish_job_event(
        tenant_id,
        {
            "id": job_id,
            "status": "queued",
            "current_stage": "queued",
            "progress_pct": 0.0,
            "eta_seconds": initial_eta,
        },
    )
    logger.info(f"[JOB {job_id}] Queued {total_items} SKUs (task={job_type}, domain={domain})")

    try:
        dispatch_fn(job_id)
    except Exception as dispatch_err:
        clear_job_progress(job_id)
        logger.error(f"[JOB {job_id}] Dispatch to ML Engine failed: {dispatch_err}")
        try:
            with get_engine().begin() as conn:
                conn.execute(
                    text(
                        "UPDATE jobs SET status = 'failed', current_stage = 'failed', error_message = :err WHERE id = :id"
                    ),
                    {"err": str(dispatch_err), "id": job_id},
                )
        except Exception as db_err:
            logger.error(f"[JOB {job_id}] Failed to record failure state in database: {db_err}")
        publish_job_event(
            tenant_id,
            {
                "id": job_id,
                "status": "failed",
                "current_stage": "failed",
                "error_message": str(dispatch_err),
                "eta_seconds": None,
            },
        )
        raise RuntimeError(f"Failed to dispatch job to ML Engine: {dispatch_err}")

    return {"job_id": job_id, "status": "queued", "total_skus": total_items}


def enqueue_job(request: BaseRequest, task: str, tenant_id: str = DEFAULT_TENANT) -> dict:
    """Registers a matching/classification/pipeline job and dispatches it to the ML Engine."""
    if not request.skus:
        raise ValueError("'skus' list is empty.")

    return register_and_dispatch_job(
        job_type=task,
        domain=request.domain or "market",
        total_items=len(request.skus),
        input_skus_json=json.dumps([sku.model_dump() for sku in request.skus]),
        tenant_id=tenant_id,
        sheet_name=request.sheet_name,
        target_sheet=request.sheet_name or "N/A",
        dispatch_fn=lambda job_id: dispatch_batch_job(
            job_id=job_id, request=request, task=task, tenant_id=tenant_id
        ),
    )
