"""
SKU MatchOps Backend - Internal Engine Callbacks API
Receives progress, completion, and failure webhook events from the dedicated ML Engine microservice.
The Gateway persists results and terminal job status directly in PostgreSQL; these callbacks
keep the Gateway's in-memory progress/ETA state in sync.
"""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from backend.app.core.db import get_engine, log_outbound_request
from backend.app.services.persistence import record_job_completion, record_job_failure

logger = logging.getLogger("matchops.engine_callbacks")
router = APIRouter(prefix="/api/internal/jobs", tags=["engine_internal"])

# In-memory store for quick polling access
_job_progress: dict[str, float] = {}
_job_eta: dict[str, int | None] = {}
_job_stage: dict[str, str] = {}


class JobProgressPayload(BaseModel):
    current_stage: str
    progress_pct: float
    eta_seconds: int | None = None


class JobCompletePayload(BaseModel):
    status: str = "completed"
    domain: str | None = None
    task: str | None = None
    results: list[dict[str, Any]] | None = None
    rule_version: str = ""
    duration_minutes: float = 0.0
    high_conf: int = 0
    med_conf: int = 0
    low_conf: int = 0
    match_rate: float = 0.0
    webhook_log: dict[str, Any] | None = None


class JobFailPayload(BaseModel):
    status: str = "failed"
    error_message: str = "Job execution failed"
    duration_minutes: float = 0.0
    is_cancelled: bool = False


@router.post("/{job_id}/progress")
def update_progress(job_id: str, payload: JobProgressPayload):
    """Updates job progress and stage in DB and in-memory cache."""
    j_id = str(job_id)
    _job_progress[j_id] = payload.progress_pct
    _job_eta[j_id] = payload.eta_seconds

    # Only perform a DB write when the stage changes
    last_stage = _job_stage.get(j_id)
    if last_stage != payload.current_stage:
        _job_stage[j_id] = payload.current_stage
        try:
            with get_engine().begin() as conn:
                # Never resurrect a job the engine already finalized (late/retried progress POST)
                conn.execute(
                    text(
                        "UPDATE jobs SET status = 'running', current_stage = :stage, updated_at = now() "
                        "WHERE id = :id AND status IN ('queued', 'running')"
                    ),
                    {"stage": payload.current_stage, "id": j_id},
                )
        except Exception as e:
            logger.error(f"Failed to update progress for job {job_id} in DB: {e}")

    return {"status": "ok"}


@router.post("/{job_id}/complete")
def complete_job(job_id: str, payload: JobCompletePayload):
    """Persists results in PostgreSQL and finalizes in-memory progress."""
    j_id = str(job_id)
    _job_progress[j_id] = 100.0
    _job_eta[j_id] = 0

    if payload.results is not None:
        try:
            record_job_completion(
                job_id=j_id,
                domain=payload.domain or "market",
                task=payload.task or "pipeline",
                results=payload.results,
                rule_version=payload.rule_version,
                duration_minutes=payload.duration_minutes,
                high_conf=payload.high_conf,
                med_conf=payload.med_conf,
                low_conf=payload.low_conf,
                match_rate=payload.match_rate,
            )
        except Exception as e:
            logger.error(f"Failed to record job completion in DB for job {job_id}: {e}")

    if payload.webhook_log:
        try:
            log_outbound_request(**payload.webhook_log)
        except Exception as e:
            logger.error(f"Failed to log outbound webhook for job {job_id}: {e}")

    return {"status": "ok"}


@router.post("/{job_id}/fail")
def fail_job(job_id: str, payload: JobFailPayload):
    """Records failure in PostgreSQL and clears the in-memory ETA."""
    j_id = str(job_id)
    _job_eta[j_id] = None
    try:
        record_job_failure(
            job_id=j_id,
            error_message=payload.error_message,
            duration_minutes=payload.duration_minutes,
            is_cancelled=payload.is_cancelled,
        )
    except Exception as e:
        logger.error(f"Failed to record job failure in DB for job {job_id}: {e}")

    return {"status": "ok"}
