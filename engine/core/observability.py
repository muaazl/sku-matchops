"""
Shared observability for the Gateway and the Engine: request/job trace ids, structured (JSON)
logging, and Prometheus metrics. Each process exposes its own registry on /metrics.
"""

import contextvars
import json
import logging
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from engine import config

# --- Trace ids ------------------------------------------------------------------------------
TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def trace_context(trace_id: str | None) -> Iterator[str]:
    """Binds a trace id to the current thread/context so every log line carries it."""
    token = TRACE_ID.set(trace_id if trace_id and trace_id != "-" else new_trace_id())
    try:
        yield TRACE_ID.get()
    finally:
        TRACE_ID.reset(token)


# --- Logging --------------------------------------------------------------------------------
TEXT_LOG_FORMAT = (
    "%(asctime)s - [%(levelname)s] - %(name)s - [trace=%(trace_id)s] - "
    "[%(filename)s:%(lineno)d] - %(message)s"
)


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = TRACE_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", TRACE_ID.get()),
            "module": f"{record.filename}:{record.lineno}",
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(logger: logging.Logger | None = None) -> None:
    """Adds trace ids to every handler on `logger` (root by default); LOG_FORMAT=json switches
    the output to one JSON object per line."""
    logger = logger or logging.getLogger()
    formatter = (
        JsonFormatter() if config.LOG_FORMAT == "json" else logging.Formatter(TEXT_LOG_FORMAT)
    )
    for handler in logger.handlers:
        if not any(isinstance(f, TraceIdFilter) for f in handler.filters):
            handler.addFilter(TraceIdFilter())
        handler.setFormatter(formatter)


# --- Metrics --------------------------------------------------------------------------------
SKUS_PROCESSED = Counter(
    "matchops_skus_processed_total",
    "SKUs processed by batch jobs",
    ["domain", "status", "tenant_id"],
)
MATCH_CONFIDENCE = Histogram(
    "matchops_match_confidence",
    "Confidence of persisted SKU results",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)
ONNX_INFERENCE_SECONDS = Histogram(
    "matchops_onnx_inference_duration_seconds",
    "Wall time of one model inference call (one micro-batch)",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
QUEUE_DEPTH = Gauge("matchops_queue_depth", "Tasks waiting in the job queue", ["queue"])
DLQ_SIZE = Gauge("matchops_dlq_size", "Tasks in the dead-letter queue")


def refresh_queue_gauges() -> None:
    from engine.pipeline.task_queue import get_task_queue

    try:
        queue = get_task_queue()
        if queue is None:
            return
        stats = queue.stats()
    except Exception:
        return  # Redis down: keep the last known values rather than fail the scrape
    for name in ("realtime", "bulk", "delayed", "inflight"):
        QUEUE_DEPTH.labels(queue=name).set(stats[name])
    DLQ_SIZE.set(stats["dlq"])


def metrics_payload() -> tuple[bytes, str]:
    """Prometheus exposition for this process (queue gauges refreshed at scrape time)."""
    refresh_queue_gauges()
    return generate_latest(), CONTENT_TYPE_LATEST


# --- HTTP trace propagation -----------------------------------------------------------------
_VALID_TRACE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class TraceIdMiddleware:
    """Pure ASGI middleware: adopts a valid incoming X-Request-ID (or mints one), binds it for the
    request's logs and downstream dispatch, and echoes it on the response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        trace_id = incoming if _VALID_TRACE.match(incoming) else new_trace_id()

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", trace_id.encode()),
                ]
            await send(message)

        token = TRACE_ID.set(trace_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            TRACE_ID.reset(token)
