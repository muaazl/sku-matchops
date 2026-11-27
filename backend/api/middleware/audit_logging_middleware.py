"""Audit logging middleware capturing redacted HTTP request and response records."""

import json
import logging
import time
import uuid
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.services import audit_service

logger = logging.getLogger("matchops.middleware.audit")

SENSITIVE_SUBSTRINGS = (
    "password",
    "token",
    "secret",
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "cookie",
    "bearer",
)

_SKIPPED_PATH_SUFFIXES = ("/stream",)
_SKIPPED_EXACT_PATHS = {"/health", "/ready", "/metrics"}
_AUDIT_MAX_BODY_BYTES = 64 * 1024  # 64 KB


def redact_sensitive_data(obj: Any) -> Any:
    """Recursively masks secret/token fields within nested dicts and lists."""
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            key_lower = str(k).lower()
            if any(sub in key_lower for sub in SENSITIVE_SUBSTRINGS):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = redact_sensitive_data(v)
        return sanitized
    if isinstance(obj, list):
        return [redact_sensitive_data(item) for item in obj]
    return obj


def format_audit_response(
    body: bytes | None,
    content_type: str | None = None,
    content_length: int | None = None,
) -> str:
    """Redacts and formats a captured HTTP response payload within a 64KB memory bound."""
    if content_length is not None and content_length > _AUDIT_MAX_BODY_BYTES:
        return f"<omitted: response body {content_length} bytes > 64KB>"
    if body is None:
        return ""
    if len(body) > _AUDIT_MAX_BODY_BYTES:
        return f"<omitted: response body {len(body)} bytes > 64KB>"
    if not body:
        return ""
    if content_type and ("application/octet-stream" in content_type or "text/csv" in content_type):
        return f"<{len(body)} bytes {content_type}>"

    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        return json.dumps(redact_sensitive_data(parsed), default=str)
    except (ValueError, TypeError):
        return text


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Persists redacted request/response metadata for non-streaming HTTP traffic."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path in _SKIPPED_EXACT_PATHS or path.endswith(_SKIPPED_PATH_SUFFIXES):
            return await call_next(request)

        start = time.perf_counter()
        raw_body = b""
        req_content_type = request.headers.get("content-type", "")
        is_multipart = "multipart/form-data" in req_content_type.lower()
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and not is_multipart:
            try:
                raw_body = await request.body()
            except Exception:
                raw_body = b""

        response = await call_next(request)
        if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
            return response

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return response

        raw_cl = response.headers.get("content-length")
        resp_content_length: int | None = None
        if raw_cl is not None:
            try:
                resp_content_length = int(raw_cl)
            except ValueError:
                resp_content_length = None

        resp_body: bytes | None = b""
        body_iterator = getattr(response, "body_iterator", None)
        if resp_content_length is not None and resp_content_length > _AUDIT_MAX_BODY_BYTES:
            resp_body = None
        elif body_iterator is not None:
            chunks = []
            async for raw_chunk in body_iterator:
                chunks.append(
                    raw_chunk if isinstance(raw_chunk, bytes) else raw_chunk.encode("utf-8")
                )
            resp_body = b"".join(chunks)
            headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
            response = Response(
                content=resp_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )

        duration_ms = int((time.perf_counter() - start) * 1000)

        payload_str = ""
        if is_multipart:
            payload_str = f"<multipart/form-data omitted for {path}>"
        elif raw_body:
            raw_text = raw_body.decode("utf-8", errors="replace")
            try:
                payload_str = json.dumps(redact_sensitive_data(json.loads(raw_text)), default=str)
            except (ValueError, TypeError):
                payload_str = raw_text[:8000]

        redacted_headers = redact_sensitive_data(dict(request.headers))
        query_params = redact_sensitive_data(dict(request.query_params))

        audit_row = {
            "id": str(uuid.uuid4()),
            "method": request.method,
            "path": path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "ip_address": request.client.host if request.client else None,
            "headers_json": json.dumps(redacted_headers, default=str),
            "query_params_json": json.dumps(query_params, default=str),
            "payload_json_redacted": payload_str[:16000],
            "response_json": format_audit_response(
                resp_body, content_type=content_type, content_length=resp_content_length
            ),
        }
        try:
            await run_in_threadpool(audit_service.record_api_request, audit_row)
        except Exception as err:
            logger.warning(f"Failed to record API audit log for {request.method} {path}: {err}")

        return response
