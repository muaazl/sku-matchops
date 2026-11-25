"""Audit logging service for inbound HTTP traffic and outbound portal/webhook requests."""

import json
import logging
import uuid
from typing import Any

from backend.core.exceptions import ResourceNotFoundError
from backend.repositories import audit_repository

logger = logging.getLogger("matchops.services.audit")


def record_api_request(row: dict[str, Any]) -> None:
    """Persists a redacted inbound HTTP request/response audit row."""
    audit_repository.insert_api_request(row)


def record_outbound_request(
    *,
    url: str,
    method: str,
    payload: object,
    response_status: int,
    response_text: str,
    duration_ms: int,
    path: str = "/doPost",
) -> None:
    """Logs outbound HTTP requests (e.g. Google Sheets webhooks, Portal API) to `api_requests`."""
    try:
        payload_str = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
        resp_str = str(response_text) if response_text is not None else ""
        audit_repository.insert_api_request(
            {
                "id": str(uuid.uuid4()),
                "method": method,
                "path": path,
                "payload_json_redacted": payload_str,
                "response_json": resp_str,
                "status_code": response_status,
                "duration_ms": duration_ms,
                "ip_address": "outbound",
                "headers_json": json.dumps({"Content-Type": "application/json"}),
                "query_params_json": json.dumps({"callback_url": url}),
            }
        )
        logger.info(
            f"[DB] Logged outbound {method} request to {path} (status {response_status}) in api_requests."
        )
    except Exception as e:
        logger.error(f"[DB] Failed to log outbound request to DB: {e}")


def list_api_requests(
    *,
    path: str | None = None,
    status_code: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Lists paginated API request audit logs."""
    return audit_repository.fetch_api_requests(
        path=path,
        status_code=status_code,
        date_from=date_from,
        date_to=date_to,
        page=page,
    )


def fetch_api_request_detail(request_id: str) -> dict[str, Any]:
    """Fetches detailed payload/header audit record or raises ResourceNotFoundError."""
    row = audit_repository.fetch_api_request_by_id(request_id)
    if not row:
        raise ResourceNotFoundError("Request not found")
    return row
