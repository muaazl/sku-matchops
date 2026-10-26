from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Connection, text

from backend.app.core.db import get_db_connection
from backend.app.core.security import require_platform_admin
from backend.app.schemas.models import ApiRequestDetailResponse, ApiRequestResponse

# The raw request log spans every tenant (payloads included): platform admins only
router = APIRouter(dependencies=[Depends(require_platform_admin)])


@router.get("/api-requests", response_model=list[ApiRequestResponse])
def get_api_requests(
    path: str | None = None,
    status_code: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    db: Connection = Depends(get_db_connection),
):
    query = "SELECT id, method, path, status_code, duration_ms, ip_address, created_at FROM api_requests WHERE 1=1"
    params = {}

    if path:
        query += " AND path LIKE :path"
        params["path"] = f"%{path}%"
    if status_code:
        query += " AND status_code = :status_code"
        params["status_code"] = status_code
    if date_from:
        query += " AND created_at >= CAST(:date_from AS timestamptz)"
        params["date_from"] = date_from
    if date_to:
        query += " AND created_at <= CAST(:date_to AS timestamptz)"
        params["date_to"] = date_to

    query += " ORDER BY created_at DESC LIMIT 50 OFFSET :offset"
    params["offset"] = (page - 1) * 50

    rows = db.execute(text(query), params).mappings().fetchall()
    return [dict(row) for row in rows]


@router.get("/api-requests/{id}", response_model=ApiRequestDetailResponse)
def get_api_request(id: str, db: Connection = Depends(get_db_connection)):
    row = (
        db.execute(
            text("""
        SELECT id, method, path, status_code, duration_ms, ip_address, headers_json, query_params_json, payload_json_redacted, response_json, created_at
        FROM api_requests
        WHERE id = :id
        """),
            {"id": id},
        )
        .mappings()
        .fetchone()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Request not found")
    return dict(row)
