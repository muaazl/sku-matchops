"""Middleware that binds per-request trace IDs and records HTTP Prometheus metrics."""

import re
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.core.observability import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    TRACE_ID,
    VALID_TRACE_PATTERN,
    generate_trace_id,
)

_NUMERIC_SEGMENT_PATTERN = re.compile(r"/\d+(?=/|$)")


def _extract_incoming_trace_id(headers) -> str | None:
    for candidate in ("x-request-id", "X-Request-ID", "x-trace-id", "X-Trace-ID"):
        val = headers.get(candidate)
        if val:
            return str(val)
    for k, v in headers.items():
        if str(k).lower() in ("x-request-id", "x-trace-id") and v:
            return str(v)
    return None


def _normalize_path_label(request: Request) -> str:
    scope = getattr(request, "scope", None)
    route = scope.get("route") if isinstance(scope, dict) else None
    route_path = getattr(route, "path", None)
    if route_path:
        return str(route_path)
    return _NUMERIC_SEGMENT_PATTERN.sub("/{id}", request.url.path)


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Propagates `X-Request-ID` trace identifiers and records request latency metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = _extract_incoming_trace_id(request.headers)
        trace_id = (
            incoming if (incoming and VALID_TRACE_PATTERN.match(incoming)) else generate_trace_id()
        )
        token = TRACE_ID.set(trace_id)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = trace_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            path_label = _normalize_path_label(request)
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method, path=path_label, status=str(status_code)
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=path_label).observe(
                elapsed
            )
            TRACE_ID.reset(token)
