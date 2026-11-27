"""Async job querying, live SSE streaming, cancellation, retry, and analytics endpoints."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from backend.api.dependencies import RateLimiter
from backend.core.exceptions import (
    ResourceConflictError,
    ResourceNotFoundError,
    ServiceUnavailableError,
    ServiceValidationError,
)
from backend.schemas.job_schema import JobResponse
from backend.services import job_service

router = APIRouter(tags=["Jobs"])

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_cancel_rate_limiter = [Depends(RateLimiter(max_requests=30, window_seconds=60))]
_retry_rate_limiter = [Depends(RateLimiter(max_requests=10, window_seconds=60))]


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(
    response: Response,
    status: str | None = None,
    type: str | None = None,
    created_by: str | None = None,
    domain: str | None = None,
    sheet_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    limit: int | None = 10000,
) -> list[JobResponse]:
    """Lists filtered jobs enriched with live Redis progress and sets `X-Total-Count`."""
    total, items = job_service.list_jobs(
        status=status,
        job_type=type,
        created_by=created_by,
        domain=domain,
        sheet_name=sheet_name,
        date_from=date_from,
        date_to=date_to,
        page=page,
        limit=limit,
    )
    response.headers["X-Total-Count"] = str(total)
    return items


@router.get("/jobs/dashboard-stats")
def get_dashboard_stats(
    domain: str | None = "all", timeframe: str | None = "30d"
) -> dict[str, Any]:
    """Returns KPI cards, confidence distribution, match sources, and volume trends."""
    return job_service.compute_dashboard_stats(domain=domain, timeframe=timeframe)


@router.get("/jobs/tag-stats")
def get_tag_stats(domain: str | None = "all", timeframe: str | None = "30d") -> dict[str, Any]:
    """Returns top Basic Types, Regions/Categories, and tag confidence tier metrics."""
    return job_service.compute_tag_stats(domain=domain, timeframe=timeframe)


async def _close_pubsub_pair(client: Any, pubsub: Any) -> None:
    try:
        await pubsub.unsubscribe()
    except Exception:
        pass
    for conn in (pubsub, client):
        if hasattr(conn, "aclose"):
            try:
                await conn.aclose()
            except Exception:
                pass


@router.get("/jobs/stream")
async def stream_jobs_events() -> StreamingResponse:
    """Streams live updates for all active and transitioning jobs over Server-Sent Events."""
    try:
        client, pubsub = await job_service.open_job_event_stream()
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def event_generator():
        try:
            snapshot = job_service.fetch_recent_job_snapshots(limit=25)
            yield f"data: {json.dumps({'type': 'snapshot', 'jobs': snapshot})}\n\n"
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if msg is None:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0)
                    continue
                raw_data = msg.get("data")
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8", errors="replace")
                if isinstance(raw_data, str) and raw_data:
                    yield f"data: {raw_data}\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            await _close_pubsub_pair(client, pubsub)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Fetches detailed metadata and live progress for a single job."""
    try:
        return job_service.fetch_job_detail(job_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/stream")
async def stream_job_events(job_id: str) -> StreamingResponse:
    """Streams live Server-Sent Events for a single job until terminal completion."""
    snapshot = job_service.fetch_single_job_snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if snapshot.get("status") in _TERMINAL_STATUSES:

        async def terminal_only():
            yield f"data: {json.dumps(snapshot)}\n\n"

        return StreamingResponse(
            terminal_only(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        client, pubsub = await job_service.open_job_event_stream()
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def single_event_generator():
        try:
            yield f"data: {json.dumps(snapshot)}\n\n"
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if msg is None:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0)
                    continue
                raw_data = msg.get("data")
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8", errors="replace")
                if not isinstance(raw_data, str) or not raw_data:
                    continue
                try:
                    event = json.loads(raw_data)
                except ValueError:
                    continue
                if str(event.get("id")) != str(job_id):
                    continue
                yield f"data: {raw_data}\n\n"
                if event.get("status") in _TERMINAL_STATUSES:
                    break
        except asyncio.CancelledError:
            raise
        finally:
            await _close_pubsub_pair(client, pubsub)

    return StreamingResponse(
        single_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/jobs/{job_id}/cancel", dependencies=_cancel_rate_limiter)
def cancel_job(job_id: str) -> dict[str, str]:
    """Requests atomic cancellation for a queued or running job."""
    try:
        return job_service.cancel_job(job_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ResourceConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/retry", dependencies=_retry_rate_limiter)
def retry_job(job_id: str) -> dict[str, Any]:
    """Re-enqueues a stored job using its saved `input_skus_json` payload."""
    try:
        return job_service.retry_job(job_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
