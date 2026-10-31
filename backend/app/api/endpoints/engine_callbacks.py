"""
SKU MatchOps Backend - Internal Engine Callbacks API
Receives progress, completion, and failure webhook events from the dedicated ML Engine microservice.
The Engine persists results and terminal job status directly in PostgreSQL; these callbacks
keep the Gateway's in-memory progress/ETA state in sync.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from backend.app.core.db import get_engine

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


class JobStatusPayload(BaseModel):
    status: str


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
def complete_job(job_id: str, payload: JobStatusPayload):
    """Finalizes in-memory progress after the Engine has persisted the job's results."""
    j_id = str(job_id)
    _job_progress[j_id] = 100.0
    _job_eta[j_id] = 0
    return {"status": "ok"}


@router.post("/{job_id}/fail")
def fail_job(job_id: str, payload: JobStatusPayload):
    """Clears the in-memory ETA after the Engine has recorded the job as failed."""
    _job_eta[str(job_id)] = None
    return {"status": "ok"}
