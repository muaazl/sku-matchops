"""Outlet portal upload dispatch and host-managed progress state update endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import RateLimiter
from backend.core.exceptions import ResourceNotFoundError, ServiceValidationError
from backend.schemas.upload_schema import UploadJobStateUpdate, UploadRequest
from backend.services import upload_service

router = APIRouter(tags=["Upload"])


@router.post(
    "/upload",
    dependencies=[Depends(RateLimiter(max_requests=20, window_seconds=60))],
)
def trigger_upload(request: UploadRequest) -> dict[str, Any]:
    """Registers an outlet portal upload job and dispatches unless host_managed."""
    try:
        return upload_service.dispatch_upload(request)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.patch("/upload/{job_id}")
def update_host_upload_job(job_id: str, body: UploadJobStateUpdate) -> dict[str, Any]:
    """Updates progress or terminal completion state for a host-managed upload job."""
    try:
        return upload_service.update_host_upload_state(job_id, body)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
