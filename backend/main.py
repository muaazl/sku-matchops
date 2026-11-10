"""
SKU MatchOps – FastAPI Application Entry Point (Lightweight API Gateway)
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.app.api.routes import api_router
from backend.app.core import config
from backend.app.core.observability import TraceIdMiddleware, configure_logging
from backend.app.db.init import init_db
from backend.app.db.session import get_engine
from backend.app.grpc_server import ensure_backend_grpc_server, stop_backend_grpc_server
from backend.app.middleware.audit_logging import AuditLoggingMiddleware
from backend.app.middleware.etag import ETagMiddleware
from backend.app.services.rules_service import sync_rules_everywhere
from backend.app.services.task_queue import get_task_queue

# Configure logging with standard stream and file handlers
LOG_DIR = config.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = config.LOG_FILE

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Preserve existing test/framework handlers (e.g. pytest LogCaptureHandler)
has_stream = any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    for h in root_logger.handlers
)
has_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

formatter = logging.Formatter(
    "%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
)

if not has_stream:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

if not has_file:
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except Exception:
        pass
configure_logging()  # trace ids on every line; LOG_FORMAT=json for structured output

logger = logging.getLogger("matchops.server")


def startup_event():
    """Startup sequence: Run schema migrations, start callback gRPC server, and sync rules."""
    logger.info("Executing Startup Sequence for Backend Gateway...")

    ensure_backend_grpc_server()

    # Auto-run schema migrations on startup
    try:
        init_db()
        logger.info("Database schemas verified/migrated.")
    except Exception as e:
        logger.error(f"Failed to execute database migrations on startup: {e}")

    try:
        sync_rules_everywhere()
    except Exception as e:
        logger.warning(f"Failed to sync rules on startup: {e}")

    # Jobs left in 'queued'/'running' state belong to in-memory queue/progress state that
    # doesn't survive a restart of the backend or the engine. Flip them to 'failed' so they
    # show up as actionable unless still held by the Redis queue.
    try:
        queue = get_task_queue()
        live_job_ids = list(queue.tracked_job_ids()) if queue is not None else []
    except Exception as e:
        logger.warning(f"Redis queue unreachable at startup ({e}); treating no jobs as queued.")
        live_job_ids = []

    try:
        with get_engine().begin() as conn:
            cur = conn.execute(
                text("""
                UPDATE jobs SET
                    status = 'failed',
                    current_stage = 'failed',
                    error_message = 'Interrupted by a service restart. Use Retry to resubmit.',
                    completed_at = now()
                WHERE status IN ('queued', 'running') AND NOT (id = ANY(:live))
                """),
                {"live": live_job_ids},
            )
            stuck_count = cur.rowcount
            # Reconcile batches: only mark failed if no remaining active jobs exist for the batch
            conn.execute(
                text("""
                UPDATE batches SET status = 'failed', completed_at = now()
                WHERE status IN ('queued', 'running')
                  AND id NOT IN (
                      SELECT DISTINCT batch_id FROM jobs
                      WHERE status IN ('queued', 'running')
                        AND batch_id IS NOT NULL
                  )
                """)
            )
        if stuck_count:
            logger.warning(
                f"Marked {stuck_count} job(s) left in queued/running state as failed after restart."
            )
    except Exception as e:
        logger.error(f"Failed to reconcile stuck jobs on startup: {e}")


def shutdown_event():
    stop_backend_grpc_server()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup_event()
    try:
        yield
    finally:
        shutdown_event()
        try:
            from backend.app.api.endpoints.jobs import close_async_redis_pool

            await close_async_redis_pool()
        except Exception:
            pass


app = FastAPI(
    title="SKU MatchOps API Gateway",
    description="Domain-aware SKU matching control plane and API gateway.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "https://sku-matchops.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "If-None-Match",
        "X-API-Key",
        "Authorization",
        "X-Tenant-ID",
        "X-Request-ID",
    ],
    expose_headers=["X-Total-Count", "X-Request-ID"],
)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(ETagMiddleware)
# Outermost: the trace id is bound before audit logging and every handler run
app.add_middleware(TraceIdMiddleware)

app.include_router(api_router)
