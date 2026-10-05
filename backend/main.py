"""
SKU MatchOps – FastAPI Application Entry Point (Lightweight API Gateway)
"""

import logging
import os
import sqlite3
import sys
import threading
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import api_router
from backend.app.middleware.audit_logging import AuditLoggingMiddleware
from backend.app.middleware.etag import ETagMiddleware
from backend.app.services.meilisearch_service import check_and_sync_meilisearch
from scripts.database.migrate import migrate
from engine import config
from engine.rules_engine import refresh_rules_cache
from backend.app.core.db import DB_PATH

# Configure logging with standard stream and file handlers
LOG_DIR = getattr(config, "LOG_DIR", config.DB_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = getattr(config, "LOG_FILE", os.path.join(LOG_DIR, "app.log"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s")

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
root_logger.addHandler(sh)

fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
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
    allow_origins=["http://localhost:5173", "http://localhost:4173", "http://127.0.0.1:5173", "https://sku-matchops.vercel.app"],
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
        migrate()
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
    try:
        conn = sqlite3.connect(DB_PATH, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        cur = conn.execute(
            """
            UPDATE jobs SET
                status = 'failed',
                current_stage = 'failed',
                error_message = 'Interrupted by a service restart. Use Retry to resubmit.',
                completed_at = datetime('now')
            WHERE status IN ('queued', 'running')
            """
        )
        stuck_count = cur.rowcount
        conn.execute(
            """
            UPDATE batches SET status = 'failed', completed_at = datetime('now')
            WHERE status IN ('queued', 'running')
            """
        )
        conn.commit()
        conn.close()
        if stuck_count:
            logger.warning(f"Marked {stuck_count} job(s) left in queued/running state as failed after restart.")
    except Exception as e:
        logger.error(f"Failed to reconcile stuck jobs on startup: {e}")

    # Start Meilisearch verification and sync in a background daemon thread
    try:
        threading.Thread(target=check_and_sync_meilisearch, daemon=True).start()
    except Exception as e:
        logger.error(f"Failed to initiate startup Meilisearch check: {e}")
