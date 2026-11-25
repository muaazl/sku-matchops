"""SQLAlchemy repository for inbound and outbound API request audit logs."""

from typing import Any

from sqlalchemy import text

from backend.core.database import get_engine
from backend.models.audit_model import api_requests


def insert_api_request(row: dict[str, Any]) -> None:
    """Inserts a single audit record into `api_requests`."""
    with get_engine().begin() as conn:
        conn.execute(api_requests.insert().values(**row))


def fetch_api_requests(
    *,
    path: str | None = None,
    status_code: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    """Fetches paginated `api_requests` summary rows matching optional filters."""
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if path:
        clauses.append("path LIKE :path")
        params["path"] = f"%{path}%"
    if status_code:
        clauses.append("status_code = :status_code")
        params["status_code"] = status_code
    if date_from:
        clauses.append("created_at >= CAST(:date_from AS timestamptz)")
        params["date_from"] = date_from
    if date_to:
        clauses.append("created_at <= CAST(:date_to AS timestamptz)")
        params["date_to"] = date_to

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params["limit"] = page_size
    params["offset"] = (page - 1) * page_size

    query = (
        f"SELECT id, method, path, status_code, duration_ms, ip_address, created_at "
        f"FROM api_requests {where} ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
    )
    with get_engine().connect() as conn:
        rows = conn.execute(text(query), params).mappings().fetchall()
    return [dict(row) for row in rows]


def fetch_api_request_by_id(request_id: str) -> dict[str, Any] | None:
    """Fetches full audit details for a single `api_requests` row by ID."""
    with get_engine().connect() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT id, method, path, status_code, duration_ms, ip_address,
                           headers_json, query_params_json, payload_json_redacted,
                           response_json, created_at
                    FROM api_requests
                    WHERE id = :id
                    """
                ),
                {"id": request_id},
            )
            .mappings()
            .fetchone()
        )
    return dict(row) if row else None


def count_api_requests_last_24h() -> int:
    """Counts recorded API requests within the last 24 hours."""
    with get_engine().connect() as conn:
        return int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM api_requests WHERE created_at >= now() - interval '24 hours'"
                )
            ).scalar()
            or 0
        )
