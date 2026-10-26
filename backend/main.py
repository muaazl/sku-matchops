"""
SKU MatchOps – FastAPI Application Entry Point (Lightweight API Gateway)
"""

import logging
import os
import sys
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.app.api.routes import api_router
from backend.app.core.db import get_engine
from backend.app.middleware.audit_logging import AuditLoggingMiddleware
from backend.app.middleware.etag import ETagMiddleware
from backend.app.services.meilisearch_service import check_and_sync_meilisearch
from engine import config
from engine.core.db import init_db
from engine.pipeline.task_queue import get_task_queue
from engine.rules_engine import refresh_rules_cache

# Configure logging with standard stream and file handlers
LOG_DIR = config.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = config.LOG_FILE

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

formatter = logging.Formatter(
    "%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
root_logger.addHandler(sh)

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(formatter)
root_logger.addHandler(fh)

logger = logging.getLogger("matchops.server")

app = FastAPI(
    title="SKU MatchOps API Gateway",
    description="Domain-aware SKU matching control plane and API gateway.",
    version="2.0.0",
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
    allow_headers=["Content-Type", "If-None-Match", "X-API-Key"],
    expose_headers=["X-Total-Count"],
)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(ETagMiddleware)

app.include_router(api_router)


@app.on_event("startup")
def startup_event():
    """Startup sequence: Run schema migrations and sync search engine."""
    logger.info("Executing Startup Sequence for Backend Gateway...")

    # Auto-run schema migrations on startup
    try:
        init_db()
        logger.info("Database schemas verified/migrated.")
    except Exception as e:
        logger.error(f"Failed to execute database migrations on startup: {e}")

    try:
        refresh_rules_cache()
    except Exception as e:
        logger.warning(f"Failed to refresh rules cache on startup: {e}")

    # Jobs left in 'queued'/'running' state belong to in-memory queue/progress state that
    # doesn't survive a restart of the backend or the engine. Without this, such a job would
    # sit stuck in that state forever with no progress updates ever arriving again. Flip them
    # to 'failed' with a clear message so they show up as actionable — the existing "Retry"
    # button re-submits them from their stored input SKUs.
    # Jobs still held by the Redis queue survive restarts and must not be failed here.
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
            conn.execute(
                text("""
                UPDATE batches SET status = 'failed', completed_at = now()
                WHERE status IN ('queued', 'running') AND NOT (id = ANY(:live))
                """),
                {"live": live_job_ids},
            )
        if stuck_count:
            logger.warning(
                f"Marked {stuck_count} job(s) left in queued/running state as failed after restart."
            )
    except Exception as e:
        logger.error(f"Failed to reconcile stuck jobs on startup: {e}")

    # Start Meilisearch verification and sync in a background daemon thread
    try:
        threading.Thread(target=check_and_sync_meilisearch, daemon=True).start()
    except Exception as e:
        logger.error(f"Failed to initiate startup Meilisearch check: {e}")
