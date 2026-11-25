"""CSV batch upload parsing, validation, job enqueueing, and batch retrieval."""

from typing import Any

from backend.core.exceptions import (
    PayloadTooLargeError,
    ResourceNotFoundError,
    ServiceValidationError,
)
from backend.repositories import batch_repository
from backend.schemas.batch_schema import BatchResponse
from backend.schemas.sku_schema import BaseRequest
from backend.services import job_service, portal_service

# Generous backstop against accidental multi-GB upload exhausting gateway memory
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def _strip_supported_extension(filename: str) -> str:
    for ext in (".csv", ".tsv", ".txt"):
        if filename.lower().endswith(ext):
            return filename[: -len(ext)]
    return filename


def create_upload_batch(
    *,
    domain: str,
    created_by: str,
    task: str,
    filename: str | None,
    content_stream: Any,
) -> dict[str, str]:
    """Validates an uploaded CSV/TSV file, enqueues its processing job, and records the batch."""
    if task not in ("pipeline", "matcher", "classifier"):
        raise ServiceValidationError("Invalid task type.")

    clean_filename = _strip_supported_extension(filename or "Job")
    content = content_stream.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )

    try:
        skus = portal_service.parse_csv_text_to_skus(content.decode("utf-8", errors="ignore"))
    except Exception as e:
        raise ServiceValidationError(f"Failed to parse CSV file: {str(e)}") from e

    if not skus:
        raise ServiceValidationError("CSV file has no valid SKUs")

    request = BaseRequest(
        skus=skus,
        domain=domain,
        callback_url="",
        sheet_name=clean_filename,
    )
    res = job_service.enqueue_job(request, task=task)
    job_id = res["job_id"]

    batch_repository.insert_batch(
        batch_id=job_id,
        source="upload",
        filename=clean_filename,
        domain=domain,
        status="queued",
        created_by=created_by,
    )
    return {"batch_id": job_id, "job_id": job_id}


def fetch_batch_detail(batch_id: str) -> BatchResponse:
    """Fetches a batch record by ID or raises ResourceNotFoundError."""
    row = batch_repository.fetch_batch_by_id(batch_id)
    if not row:
        raise ResourceNotFoundError("Batch not found")
    return BatchResponse(**row)
