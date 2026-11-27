"""Outlet portal upload dispatch and host-managed Playwright progress state transitions."""

import json
from typing import Any

from backend.core import config
from backend.core.exceptions import ResourceNotFoundError, ServiceValidationError
from backend.core.observability import get_trace_id
from backend.repositories import job_progress_repository, job_repository
from backend.schemas.upload_schema import UploadJobStateUpdate, UploadRequest
from backend.services import engine_gateway_service, job_callback_service, job_service

ALLOWED_UPLOAD_COLUMNS = (
    "Name",
    "RefID",
    "Description",
    "Price",
    "Category",
    "Individually Sellable",
    "Brands",
    "Categories",
    "Generic keywords",
    "Promotions",
    "Special Properties and Ingredients",
    "Uncategrized",
    "basictype",
    "dietarypreference",
    "foodstate",
    "mealtime",
    "other",
    "region",
)


def dispatch_upload_payload(job_id: str, request: UploadRequest, task: str) -> dict[str, Any]:
    """Dispatches a portal upload job via Redis queue, falling back to direct Engine gRPC."""
    filtered_skus = [
        {col: sku.get(col, "") for col in ALLOWED_UPLOAD_COLUMNS} for sku in request.skus
    ]
    payload = {
        "job_id": str(job_id),
        "task": task,
        "domain": request.domain or "market",
        "skus": filtered_skus,
        "backend_grpc_url": config.BACKEND_GRPC_URL,
        "sheet_name": request.outlet_id_or_name,
        "callback_url": None,
        "spreadsheet_id": None,
        "trace_id": get_trace_id(),
    }
    queued = job_service.try_enqueue_redis_task(job_id, payload, "upload job")
    if queued is not None:
        return queued
    return engine_gateway_service.dispatch_upload_job(
        job_id=job_id,
        skus_payload=filtered_skus,
        domain=request.domain or "market",
        outlet_id_or_name=request.outlet_id_or_name or "",
        task=task,
    )


def dispatch_upload(request: UploadRequest) -> dict[str, Any]:
    """Validates and registers an upload job, dispatching remotely unless host_managed."""
    if not request.skus:
        raise ServiceValidationError("'skus' list is empty.")

    dispatch_fn = (
        (lambda _job_id: None)
        if request.host_managed
        else lambda job_id: dispatch_upload_payload(job_id=job_id, request=request, task="upload")
    )
    return job_service.register_and_dispatch_job(
        job_type="upload",
        domain=request.domain or "market",
        total_items=len(request.skus),
        input_skus_json=json.dumps(request.skus),
        sheet_name=request.sheet_name,
        target_sheet=request.outlet_id_or_name,
        dispatch_fn=dispatch_fn,
    )


def update_host_upload_state(job_id: str, body: UploadJobStateUpdate) -> dict[str, Any]:
    """Transitions progress or terminal status for a host-executed Playwright upload job."""
    if body.status == "running":
        row = job_repository.activate_job_if_queued(job_id, body.stage)
    else:
        row = job_repository.fetch_job_status_and_inputs(job_id)

    if not row:
        raise ResourceNotFoundError("Job not found")

    cancel_requested = bool(row["cancel_requested"]) or row["status"] == "cancelled"

    if body.status == "running":
        if cancel_requested or row["status"] in ("completed", "failed", "cancelled"):
            return {"status": row["status"], "cancel_requested": cancel_requested}

        pct = float(body.progress_pct or 0.0)
        job_progress_repository.set_job_progress(job_id, pct=pct, stage=body.stage)
        job_progress_repository.publish_job_event(
            {
                "id": job_id,
                "status": "running",
                "current_stage": body.stage,
                "progress_pct": pct,
                "eta_seconds": None,
            }
        )
        return {"status": "running", "cancel_requested": False}

    if body.status == "completed":
        input_skus = json.loads(row["input_skus_json"]) if row["input_skus_json"] else []
        results = [
            {
                "status": "High (Uploaded)",
                "sku_name": s.get("Name", s.get("sku_name", s.get("name", "Unknown"))),
            }
            for s in input_skus
        ]
        total = len(results)
        match_rate = 100.0 if total else 0.0
        job_callback_service.record_job_completion(
            job_id=job_id,
            domain=row["domain"] or "food",
            task="upload",
            results=results,
            rule_version="",
            duration_minutes=float(body.duration_minutes),
            high_conf=total,
            med_conf=0,
            low_conf=0,
            match_rate=match_rate,
        )
        return {"status": "completed", "cancel_requested": False}

    is_cancelled = body.status == "cancelled"
    err_msg = body.error_message or ("Job cancelled" if is_cancelled else "Upload failed")
    job_callback_service.record_job_failure(
        job_id=job_id,
        error_message=err_msg,
        duration_minutes=float(body.duration_minutes),
        is_cancelled=is_cancelled,
    )
    return {"status": body.status, "cancel_requested": is_cancelled}
