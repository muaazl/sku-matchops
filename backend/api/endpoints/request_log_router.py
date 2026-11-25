"""HTTP and outbound request audit log inspection endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.core.exceptions import ResourceNotFoundError
from backend.core.security import require_platform_admin
from backend.schemas.audit_schema import ApiRequestDetailResponse, ApiRequestResponse
from backend.services import audit_service

router = APIRouter(
    tags=["Audit"],
    dependencies=[Depends(require_platform_admin)],
)


@router.get("/api-requests", response_model=list[ApiRequestResponse])
def list_api_requests(
    path: str | None = None,
    status_code: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
) -> list[ApiRequestResponse]:
    """Fetches paginated HTTP/outbound request audit logs."""
    rows = audit_service.list_api_requests(
        path=path,
        status_code=status_code,
        date_from=date_from,
        date_to=date_to,
        page=page,
    )
    return [ApiRequestResponse(**row) for row in rows]


@router.get("/api-requests/{request_id}", response_model=ApiRequestDetailResponse)
def get_api_request(request_id: str) -> ApiRequestDetailResponse:
    """Fetches redacted payload, header, and response details for an audit record."""
    try:
        row = audit_service.fetch_api_request_detail(request_id)
        return ApiRequestDetailResponse(**row)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
