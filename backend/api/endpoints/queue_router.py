"""Distributed Redis task queue DLQ management and system status endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.core.exceptions import (
    ResourceConflictError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from backend.services import queue_service

router = APIRouter(tags=["Queue & Operations"])


@router.get("/queue/dlq")
def list_dead_letters(limit: int = 100) -> list[dict[str, Any]]:
    """Lists dead-lettered tasks waiting for inspection or replay."""
    try:
        return queue_service.list_dead_letters(limit=limit)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/queue/dlq/{task_id}/replay")
def replay_dead_letter(task_id: str) -> dict[str, Any]:
    """Resets database job status and re-enqueues a dead-lettered task."""
    try:
        return queue_service.replay_dead_letter(task_id)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ResourceConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/system/status")
def get_system_status() -> dict[str, Any]:
    """Returns processed SKU count, active rule version, queue depth, and DLQ count."""
    return queue_service.fetch_system_status()
