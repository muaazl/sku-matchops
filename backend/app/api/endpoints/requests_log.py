from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List
import sqlite3
from backend.app.core.db import get_db_connection
from backend.app.schemas.models import ApiRequestResponse, ApiRequestDetailResponse

router = APIRouter()

@router.get("/api-requests", response_model=List[ApiRequestResponse])
def get_api_requests(
    path: Optional[str] = None,
    status_code: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    query = "SELECT id, method, path, status_code, duration_ms, ip_address, created_at FROM api_requests WHERE 1=1"
    params = []
    
    if path:
        query += " AND path LIKE ?"
        params.append(f"%{path}%")
    if status_code:
        query += " AND status_code = ?"
        params.append(status_code)
    if date_from:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= ?"
        params.append(date_to)
        
    query += " ORDER BY created_at DESC LIMIT 50 OFFSET ?"
    params.append((page - 1) * 50)
    
    rows = db.execute(query, params).fetchall()
    return [dict(row) for row in rows]

@router.get("/api-requests/{id}", response_model=ApiRequestDetailResponse)
def get_api_request(id: str, db: sqlite3.Connection = Depends(get_db_connection)):
    row = db.execute(
        """
        SELECT id, method, path, status_code, duration_ms, ip_address, headers_json, query_params_json, payload_json_redacted, response_json, created_at
        FROM api_requests
        WHERE id = ?
        """,
        (id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Request not found")
    return dict(row)

