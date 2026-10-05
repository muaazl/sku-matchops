import sqlite3

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.app.core.db import get_db_connection
from backend.app.schemas.models import (
    BaseRequest,
    BatchResponse,
    MerchantFetchRequest,
)
from backend.app.services.portal_service import fetch_merchant_csv, parse_csv_text_to_skus
from backend.app.services.worker import enqueue_job

router = APIRouter()

# Generous backstop against an accidental multi-GB upload consuming all backend memory —
# not meant to be a restrictive limit for realistic CSV/TSV exports.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


@router.post("/batches")
def create_batch(
    domain: str = Form(...),
    created_by: str = Form(...),
    task: str = Form("pipeline"),
    file: UploadFile = File(...),
    db: sqlite3.Connection = Depends(get_db_connection),
):
    # A plain `def` endpoint runs in FastAPI's threadpool rather than on the event loop,
    # so the blocking SQLite writes and the blocking dispatch POST to the engine below
    # (both inside enqueue_job) don't stall every other request while this one is in flight.
    if task not in ("pipeline", "matcher", "classifier"):
        raise HTTPException(status_code=400, detail="Invalid task type.")

    filename = file.filename or "Job"
    if filename:
        for ext in (".csv", ".tsv", ".txt"):
            if filename.lower().endswith(ext):
                filename = filename[: -len(ext)]
                break

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    # Parse CSV content
    try:
        skus = parse_csv_text_to_skus(content.decode("utf-8", errors="ignore"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if not skus:
        raise HTTPException(status_code=400, detail="CSV file has no valid SKUs")

    # Enqueue a pipeline job
    request = BaseRequest(
        skus=skus,
        domain=domain,
        callback_url="",
        sheet_name=filename,  # filename serves as target sheet name
    )

    res = enqueue_job(request, task=task)
    job_id = res["job_id"]

    # Insert batch entry
    db.execute(
        """
        INSERT INTO batches (id, source, filename, domain, status, created_by)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, "upload", filename, domain, "queued", created_by),
    )
    db.commit()

    return {"batch_id": job_id, "job_id": job_id}


@router.post("/merchant-fetch")
def merchant_fetch(request: MerchantFetchRequest):
    """
    Fetches a merchant's SKU CSV from the Food Portal API and returns parsed rows for preview.
    Does NOT create a job — mirrors the CSV upload flow, where a separate call to
    POST /batches (via the "Start Processing" confirm step) actually enqueues the job.
    """
    if not request.merchant_id or not request.merchant_id.strip():
        raise HTTPException(status_code=400, detail="Merchant ID is required.")
    if not request.bearer_token or not request.bearer_token.strip():
        raise HTTPException(status_code=400, detail="Bearer token is required.")

    result = fetch_merchant_csv(request.merchant_id, request.bearer_token, request.portal_url)

    if result["auth_failed"]:
        # 401 signals the frontend to clear the stored token and re-prompt for a fresh one.
        raise HTTPException(
            status_code=401,
            detail=result["error"] or "Portal token is invalid, blacklisted, or expired.",
        )
    if result["error"]:
        raise HTTPException(status_code=502, detail=result["error"])

    return {
        "merchant_id": request.merchant_id,
        "rows": [s.model_dump() for s in result["skus"]],
    }


@router.get("/batches/{id}", response_model=BatchResponse)
def get_batch(id: str, db: sqlite3.Connection = Depends(get_db_connection)):
    row = db.execute("SELECT * FROM batches WHERE id = ?", (id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Batch not found")
    return dict(row)
