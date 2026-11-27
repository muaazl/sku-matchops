"""Backend API Gateway runtime configuration and paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

# --- Authentication & Access Control ---
API_KEY = os.getenv("MATCHOPS_API_KEY", "bismillah")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"
ENVIRONMENT = "production"

if not API_KEY:
    raise RuntimeError("A non-empty MATCHOPS_API_KEY must be configured in environment variables.")

# --- External Integrations ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
DEFAULT_PORTAL_URL = (
    "https://uni-portal-api.pickme.lk/food/v1/t0001/food/place/skus/csv/{merchantid}"
)
PORTAL_URL = os.getenv("PORTAL_URL", DEFAULT_PORTAL_URL)

# --- PostgreSQL (owned exclusively by the Backend Gateway) ---
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))
# Rows per INSERT statement when bulk-writing per-SKU results
DB_BULK_CHUNK_SIZE = int(os.getenv("DB_BULK_CHUNK_SIZE", "1000"))

# --- gRPC Endpoints (Backend <-> Engine) ---
ENGINE_GRPC_URL = os.getenv("ENGINE_GRPC_URL", "127.0.0.1:50051")
BACKEND_GRPC_PORT = int(os.getenv("BACKEND_GRPC_PORT", "50052"))
BACKEND_GRPC_URL = os.getenv("BACKEND_GRPC_URL", f"127.0.0.1:{BACKEND_GRPC_PORT}")
INTERNAL_GRPC_SECRET = os.getenv("INTERNAL_GRPC_SECRET", "")

# --- Redis & Task Queue ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TASK_QUEUE_ENABLED = os.getenv("TASK_QUEUE_ENABLED", "true").lower() == "true"

# --- Thresholds ---
AUTO_THRESHOLD = float(os.getenv("AUTO_THRESHOLD", "0.80"))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.50"))

# --- Persistent Data & Logging Paths ---
DATA_DIR = ROOT_DIR / "data"
STAGING_DIR = Path(os.getenv("STAGING_DIR", str(DATA_DIR / "staged_sheets")))
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
