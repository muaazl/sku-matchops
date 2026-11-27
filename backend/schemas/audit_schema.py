"""Pydantic schemas for HTTP/outbound request audit log responses."""

from datetime import datetime

from pydantic import BaseModel


class ApiRequestResponse(BaseModel):
    id: str
    method: str
    path: str
    status_code: int
    duration_ms: int | None = None
    ip_address: str | None = None
    created_at: datetime | None = None


class ApiRequestDetailResponse(ApiRequestResponse):
    headers_json: str | None = None
    query_params_json: str | None = None
    payload_json_redacted: str | None = None
    response_json: str | None = None
