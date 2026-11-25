"""
Engine observability: request/job trace IDs, structured (JSON) logging,
and Prometheus metrics without upward pipeline dependencies.
"""

import contextvars
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from engine.core import config

# --- Trace IDs ------------------------------------------------------------------------------
TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
VALID_TRACE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def generate_trace_id() -> str:
    """Generates a 32-character hexadecimal trace identifier."""
    return uuid.uuid4().hex


@contextmanager
def bind_trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Binds a trace id to the current thread/context so every log line carries it."""
    token = TRACE_ID.set(trace_id if trace_id and trace_id != "-" else generate_trace_id())
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


def configure_logging(
    logger: logging.Logger | None = None,
    ensure_handlers: bool = False,
) -> None:
    """Adds trace IDs and structured formatting to `logger` (root by default)."""
    logger = logger or logging.getLogger()
    if ensure_handlers:
        logger.setLevel(logging.INFO)
        has_stream = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )
        has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        if not has_stream:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        if not has_file:
            try:
                os.makedirs(config.LOG_DIR, exist_ok=True)
                logger.addHandler(logging.FileHandler(config.LOG_FILE, encoding="utf-8"))
            except Exception:
                pass

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
    ["domain", "status"],
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

QueueStatsProvider = Callable[[], dict[str, int] | None]
_queue_stats_provider: QueueStatsProvider | None = None


def register_queue_stats_provider(
    provider: QueueStatsProvider | None,
) -> QueueStatsProvider | None:
    """Registers a queue statistics provider callback without importing pipeline modules."""
    global _queue_stats_provider
    previous = _queue_stats_provider
    _queue_stats_provider = provider
    return previous


def refresh_queue_gauges() -> None:
    """Refreshes queue depth and DLQ gauges using the registered stats provider."""
    if _queue_stats_provider is None:
        return
    try:
        stats = _queue_stats_provider()
        if not stats:
            return
    except Exception:
        return  # Redis down: keep the last known values rather than fail the scrape
    for name in ("realtime", "bulk", "delayed", "inflight"):
        QUEUE_DEPTH.labels(queue=name).set(stats[name])
    DLQ_SIZE.set(stats["dlq"])


def render_metrics_payload() -> tuple[bytes, str]:
    """Prometheus exposition for this process (queue gauges refreshed at scrape time)."""
    refresh_queue_gauges()
    return generate_latest(), CONTENT_TYPE_LATEST
