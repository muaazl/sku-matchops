"""
SKU MatchOps – FastAPI Application Entry Point (Lightweight API Gateway)
"""

import logging
import os
import sys
import threading
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import api_router
from backend.app.middleware.audit_logging import AuditLoggingMiddleware
from backend.app.middleware.etag import ETagMiddleware
from backend.app.services.meilisearch_service import check_and_sync_meilisearch
from backend.migrate_db import migrate
from engine import config
from engine.rules_engine import refresh_rules_cache

# Configure logging with standard stream and file handlers
LOG_DIR = config.DB_DIR
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://sku-matchops.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    # Start Meilisearch verification and sync in a background daemon thread
    try:
        threading.Thread(target=check_and_sync_meilisearch, daemon=True).start()
    except Exception as e:
        logger.error(f"Failed to initiate startup Meilisearch check: {e}")
