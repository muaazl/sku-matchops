import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from backend.app.core.security import Principal, get_principal
from backend.app.db.session import get_engine
from backend.app.grpc_server import clear_job_progress, publish_job_event, set_job_progress
from backend.app.schemas.models import EnqueueJobResponse, UploadJobStateUpdate, UploadRequest
from backend.app.services.engine_client import dispatch_upload_job
from backend.app.services.persistence import record_job_completion, record_job_failure
from backend.app.services.worker import register_and_dispatch_job

router = APIRouter()


@router.post("/upload", response_model=EnqueueJobResponse)
def trigger_upload(request: UploadRequest, principal: Principal = Depends(get_principal)):
    if not request.skus:
        raise HTTPException(status_code=400, detail="'skus' list is empty.")

    try:
        dispatch_fn = (
            (lambda _job_id: None)
            if request.host_managed
            else lambda job_id: dispatch_upload_job(job_id=job_id, request=request, task="upload")
        )
        return register_and_dispatch_job(
            job_type="upload",
            domain=request.domain or "market",
            total_items=len(request.skus),
            input_skus_json=json.dumps(request.skus),
            sheet_name=request.sheet_name,
            target_sheet=request.outlet_id_or_name,
            dispatch_fn=dispatch_fn,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/upload/{job_id}")
def update_host_upload_job(
    job_id: str,
    body: UploadJobStateUpdate,
    principal: Principal = Depends(get_principal),
):
    """Updates progress or terminal status for a host-executed Playwright upload job."""
    with get_engine().begin() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT id, status, cancel_requested, domain, input_skus_json "
                    "FROM jobs WHERE id = :id"
                ),
                {"id": job_id},
            )
            .mappings()
            .fetchone()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")

        cancel_requested = bool(row["cancel_requested"]) or row["status"] == "cancelled"
        if body.status == "running" and not (
            cancel_requested or row["status"] in ("completed", "failed", "cancelled")
        ):
            if row["status"] != "running":
                conn.execute(
                    text(
                        "UPDATE jobs SET status = 'running', current_stage = :stage, updated_at = now() "
                        "WHERE id = :id AND status IN ('queued', 'running')"
                    ),
                    {"stage": body.stage, "id": job_id},
                )

    if body.status == "running":
        if cancel_requested or row["status"] in ("completed", "failed", "cancelled"):
            return {"status": row["status"], "cancel_requested": cancel_requested}

        pct = float(body.progress_pct or 0.0)
        set_job_progress(job_id, pct=pct, stage=body.stage)
        publish_job_event(
            {
                "id": job_id,
                "status": "running",
                "current_stage": body.stage,
                "progress_pct": pct,
                "eta_seconds": None,
            },
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
        record_job_completion(
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
        clear_job_progress(job_id)
        publish_job_event(
            {
                "id": job_id,
                "status": "completed",
                "current_stage": "done",
                "progress_pct": 100.0,
                "eta_seconds": 0,
                "high_conf": total,
                "med_conf": 0,
                "low_conf": 0,
                "match_rate": match_rate,
                "duration_minutes": float(body.duration_minutes),
            },
        )
        return {"status": "completed", "cancel_requested": False}

    is_cancelled = body.status == "cancelled"
    err_msg = body.error_message or ("Job cancelled" if is_cancelled else "Upload failed")
    record_job_failure(
        job_id=job_id,
        error_message=err_msg,
        duration_minutes=float(body.duration_minutes),
        is_cancelled=is_cancelled,
    )
    clear_job_progress(job_id)
    publish_job_event(
        {
            "id": job_id,
            "status": "cancelled" if is_cancelled else "failed",
            "current_stage": "cancelled" if is_cancelled else "failed",
            "error_message": err_msg,
            "duration_minutes": float(body.duration_minutes),
            "eta_seconds": None,
        },
    )
    return {"status": body.status, "cancel_requested": is_cancelled}
