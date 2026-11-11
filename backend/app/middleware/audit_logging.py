import json
import logging
import time
import uuid

from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.db.schema import api_requests
from backend.app.db.session import get_engine

logger = logging.getLogger("matchops.middleware")


def _insert_api_request(row: dict) -> None:
    with get_engine().begin() as conn:
        conn.execute(api_requests.insert().values(**row))


SENSITIVE_SUBSTRINGS = ("token", "auth", "bearer", "password", "secret", "key", "x-api-key")


def _redact_sensitive_data(obj):
    """Recursively redacts sensitive keys in dictionaries and lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _redact_sensitive_data(v)
            else:
                key_lower = k.lower()
                if any(sub in key_lower for sub in SENSITIVE_SUBSTRINGS):
                    obj[k] = "***REDACTED***"
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _redact_sensitive_data(item)


def _format_audit_response(body_bytes: bytes | None, content_length: int | None = None) -> str:
    """Format and redact response body for audit trail, truncating responses exceeding 64KB."""
    if content_length is not None and content_length > 65536:
        return f"<omitted: response body {content_length} bytes > 64KB>"
    if body_bytes is not None and len(body_bytes) > 65536:
        return f"<omitted: response body {len(body_bytes)} bytes > 64KB>"
    if not body_bytes:
        return ""
    text = body_bytes.decode("utf-8", errors="replace")
    try:
        resp_data = json.loads(text)
        _redact_sensitive_data(resp_data)
        return json.dumps(resp_data)
    except Exception:
        return text


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Determine if we should log this request
        path = request.url.path
        method = request.method

        # Only log matcher, classify, and pipeline calls, exclude GETs (polling)
        log_paths = ("/match", "/classify", "/pipeline", "/upload")
        should_log = path.startswith(log_paths) and method != "GET"

        if not should_log:
            return await call_next(request)

        start_time = time.time()
        req_id = str(uuid.uuid4())

        # Read body only for logged requests.
        body = await request.body()

        # Restore body for the next handlers
        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive

        payload_redacted = ""
        ip_address = ""
        headers_json = ""
        query_params_json = ""

        # 1. Get client IP (support proxies/cloudflare tunnel)
        ip_address = request.headers.get("x-forwarded-for")
        if not ip_address:
            ip_address = request.headers.get("x-real-ip")
        if not ip_address:
            ip_address = request.client.host if request.client else "127.0.0.1"
        if ip_address and "," in ip_address:
            ip_address = ip_address.split(",")[0].strip()

        # 2. Capture and redact request headers
        try:
            headers_dict = dict(request.headers)
            for key in list(headers_dict.keys()):
                key_lower = key.lower()
                if any(sub in key_lower for sub in SENSITIVE_SUBSTRINGS):
                    headers_dict[key] = "***REDACTED***"
            headers_json = json.dumps(headers_dict)
        except Exception:
            headers_json = "{}"

        # 3. Capture query parameters
        try:
            query_params_dict = dict(request.query_params)
            query_params_json = json.dumps(query_params_dict)
        except Exception:
            query_params_json = "{}"

        # 4. Redact payload/body
        try:
            if body:
                payload_json = json.loads(body.decode("utf-8"))
                _redact_sensitive_data(payload_json)
                payload_redacted = json.dumps(payload_json)
        except Exception:
            payload_redacted = "<non-json body or decode error>"

        # Process the request
        response = await call_next(request)

        duration_ms = int((time.time() - start_time) * 1000)
        status_code = response.status_code

        # 5. Capture response body safely (capped at 64KB to avoid memory blowup)
        response_json = ""
        try:
            content_length = response.headers.get("content-length")
            cl_int = int(content_length) if content_length and content_length.isdigit() else None
            if cl_int and cl_int > 65536:
                response_json = f"<omitted: response body {cl_int} bytes > 64KB>"
            elif hasattr(response, "body_iterator"):
                response_body = []
                total_bytes = 0
                too_large = False
                async for section in response.body_iterator:
                    response_body.append(section)
                    total_bytes += len(section)
                    if total_bytes > 65536:
                        too_large = True

                async def async_iter():
                    for chunk in response_body:
                        yield chunk

                response.body_iterator = async_iter()
                if too_large:
                    response_json = f"<omitted: response body {total_bytes} bytes > 64KB>"
                else:
                    response_json = _format_audit_response(b"".join(response_body))
            elif hasattr(response, "body"):
                response_json = _format_audit_response(response.body, cl_int)
        except Exception as e:
            logger.error(f"Failed to capture response body: {e}")

        # Write to DB (off the event loop: the pooled connection is blocking)
        try:
            await run_in_threadpool(
                _insert_api_request,
                {
                    "id": req_id,
                    "method": method,
                    "path": path,
                    "payload_json_redacted": payload_redacted,
                    "response_json": response_json,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "ip_address": ip_address,
                    "headers_json": headers_json,
                    "query_params_json": query_params_json,
                },
            )
        except Exception as e:
            logger.error(f"Failed to log API request: {e}")

        logger.info(f"API {method} {path} - {status_code} - {duration_ms}ms")
        return response
