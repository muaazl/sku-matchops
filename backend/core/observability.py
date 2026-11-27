"""Trace ID propagation, structured logging, and Prometheus metrics for the Gateway."""

import contextvars
import json
import logging
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
VALID_TRACE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

TEXT_LOG_FORMAT = (
    "%(asctime)s - [%(levelname)s] - %(name)s - [trace=%(trace_id)s] - "
    "[%(filename)s:%(lineno)d] - %(message)s"
)


def generate_trace_id() -> str:
    """Generates a 32-character hexadecimal trace identifier."""
    return uuid.uuid4().hex


@contextmanager
def bind_trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Binds a trace id to the current execution context so every log line carries it."""
    token = TRACE_ID.set(trace_id if trace_id and trace_id != "-" else generate_trace_id())
    try:
        yield TRACE_ID.get()
    finally:
        TRACE_ID.reset(token)


def get_trace_id() -> str:
    """Returns the active trace id, falling back to an in-process engine context if loaded."""
    tid = TRACE_ID.get()
    if tid == "-" and "engine.core.observability" in sys.modules:
        try:
            engine_tid = sys.modules["engine.core.observability"].TRACE_ID.get()
            if engine_tid and engine_tid != "-":
                return engine_tid
        except Exception:
            pass
    return tid


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        tid = get_trace_id()
        existing = getattr(record, "trace_id", None)
        if tid != "-" or not existing or existing == "-":
            record.trace_id = tid
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", get_trace_id()),
            "module": f"{record.filename}:{record.lineno}",
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(logger: logging.Logger | None = None, log_format: str = "text") -> None:
    """Attaches TraceIdFilter and configures text or JSON formatting on logger handlers."""
    logger = logger or logging.getLogger()
    formatter = JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_LOG_FORMAT)
    for handler in logger.handlers:
        if not any(isinstance(f, TraceIdFilter) for f in handler.filters):
            handler.addFilter(TraceIdFilter())
        handler.setFormatter(formatter)


# --- Metrics --------------------------------------------------------------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "matchops_backend_http_requests_total",
    "Total HTTP requests handled by the Gateway",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "matchops_backend_http_request_duration_seconds",
    "HTTP request latency at the Gateway in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
MATCH_CONFIDENCE = Histogram(
    "matchops_backend_match_confidence",
    "Confidence of persisted SKU results",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)


def refresh_queue_gauges() -> None:
    """Refreshes queue gauges via an in-process engine observability module when loaded."""
    if "engine.core.observability" in sys.modules:
        try:
            sys.modules["engine.core.observability"].refresh_queue_gauges()
        except Exception:
            pass


def render_metrics_payload() -> tuple[bytes, str]:
    """Serializes Prometheus metrics for exposition."""
    refresh_queue_gauges()
    return generate_latest(), CONTENT_TYPE_LATEST
