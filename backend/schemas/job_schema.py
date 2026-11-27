"""Pydantic schemas for job status, enqueueing responses, and job details."""

from datetime import datetime

from pydantic import BaseModel


class JobResponse(BaseModel):
    id: str
    batch_id: str | None = None
    type: str
    status: str
    current_stage: str
    total_items: int
    completed_items: int
    eta_seconds: int | None = None
    error_message: str | None = None
    created_by: str | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    domain: str | None = None
    sheet_name: str | None = None
    target_sheet: str | None = None
    duration_minutes: float | None = None
    high_conf: int | None = None
    med_conf: int | None = None
    low_conf: int | None = None
    match_rate: float | None = None
    input_skus_json: str | None = None
    progress_pct: float | None = 0.0


class EnqueueJobResponse(BaseModel):
    job_id: str
    status: str
    total_skus: int
