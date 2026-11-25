"""Pydantic schemas for outlet catalog upload jobs and host-managed progress updates."""

from typing import Literal

from pydantic import BaseModel

from backend.core.constants import DOMAIN_MARKET


class UploadRequest(BaseModel):
    outlet_id_or_name: str
    skus: list[dict]
    domain: str = DOMAIN_MARKET
    sheet_name: str | None = None
    host_managed: bool = False


class UploadJobStateUpdate(BaseModel):
    status: Literal["running", "completed", "failed", "cancelled"]
    progress_pct: float | None = None
    stage: str = "uploading"
    duration_minutes: float = 0.0
    error_message: str | None = None
