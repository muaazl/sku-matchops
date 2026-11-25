"""CSV upload batches and merchant portal preview endpoints."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.core.constants import DOMAIN_MARKET
from backend.core.exceptions import (
    PayloadTooLargeError,
    ResourceNotFoundError,
    ServiceValidationError,
    UnauthorizedPortalError,
    UpstreamGatewayError,
)
from backend.schemas.batch_schema import BatchResponse, MerchantFetchRequest
from backend.services import batch_service, portal_service

router = APIRouter(tags=["Batches"])


@router.post("/batches")
def create_batch(
    file: UploadFile = File(...),
    domain: str = Form(DOMAIN_MARKET),
    created_by: str = Form("web-user"),
    task: str = Form("pipeline"),
) -> dict[str, str]:
    """Uploads a CSV/TSV file, parses valid SKUs, and enqueues an async processing job."""
    try:
        return batch_service.create_upload_batch(
            domain=domain,
            created_by=created_by,
            task=task,
            filename=file.filename,
            content_stream=file.file,
        )
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/merchant-fetch")
def fetch_merchant_skus(request: MerchantFetchRequest) -> dict:
    """Proxies an authenticated fetch of merchant SKU rows from the Food Portal."""
    try:
        return portal_service.fetch_merchant_preview(
            merchant_id=request.merchant_id,
            bearer_token=request.bearer_token,
            portal_url=request.portal_url,
        )
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnauthorizedPortalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except UpstreamGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/batches/{batch_id}", response_model=BatchResponse)
def get_batch(batch_id: str) -> BatchResponse:
    """Fetches a batch upload metadata record by ID."""
    try:
        return batch_service.fetch_batch_detail(batch_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
