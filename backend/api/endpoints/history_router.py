"""Processed SKU audit history query and CSV export endpoints."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from backend.core.exceptions import ResourceNotFoundError
from backend.services import history_service

router = APIRouter(tags=["History"])


@router.get("/processed-skus")
def list_processed_skus(
    response: Response,
    batch_id: str | None = None,
    domain: str | None = None,
    min_confidence: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sku_name: str | None = None,
    bt: str | None = None,
    gk: str | None = None,
    match_source: str | None = None,
    page: int = 1,
    limit: int | None = 50,
) -> list[dict]:
    """Queries paginated processed SKU audit rows and sets `X-Total-Count`."""
    total, rows = history_service.list_processed_skus(
        batch_id=batch_id,
        domain=domain,
        min_confidence=min_confidence,
        date_from=date_from,
        date_to=date_to,
        sku_name=sku_name,
        bt=bt,
        gk=gk,
        match_source=match_source,
        page=page,
        limit=limit,
    )
    response.headers["X-Total-Count"] = str(total)
    return rows


@router.get("/processed-skus/export")
def export_processed_skus(ids: str | None = None, format: str = "csv") -> StreamingResponse:
    """Streams formula-sanitized processed SKU history rows as CSV."""
    if format.lower() != "csv":
        raise HTTPException(
            status_code=400, detail="Unsupported export format. Only 'csv' is supported."
        )
    return StreamingResponse(
        history_service.generate_csv_chunks(ids=ids),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="processed_skus.csv"'},
    )


@router.get("/processed-skus/{sku_id}")
def get_processed_sku(sku_id: str) -> dict:
    """Fetches detailed persisted inference outputs for a single processed SKU."""
    try:
        return history_service.fetch_processed_sku_detail(sku_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
