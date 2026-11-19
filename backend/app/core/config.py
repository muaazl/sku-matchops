import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# System configuration (always running in standard secure mode)
API_KEY = os.getenv("MATCHOPS_API_KEY", "bismillah")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

# --- PostgreSQL & PgBouncer (owned exclusively by the Backend Gateway) ---
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://matchops:matchops@localhost:5432/matchops"
)
USE_PGBOUNCER = os.getenv("USE_PGBOUNCER", "false").lower() == "true"
PGBOUNCER_PORT = int(os.getenv("PGBOUNCER_PORT", "6432"))
PGBOUNCER_URL = os.getenv("PGBOUNCER_URL", "")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
# Rows per INSERT statement when bulk-writing per-SKU results
DB_BULK_CHUNK_SIZE = int(os.getenv("DB_BULK_CHUNK_SIZE", "1000"))

# --- gRPC Endpoints (Backend <-> Engine) ---
ENGINE_GRPC_URL = os.getenv("ENGINE_GRPC_URL", "127.0.0.1:50051")
BACKEND_GRPC_PORT = int(os.getenv("BACKEND_GRPC_PORT", "50052"))
BACKEND_GRPC_URL = os.getenv("BACKEND_GRPC_URL", f"127.0.0.1:{BACKEND_GRPC_PORT}")
INTERNAL_GRPC_SECRET = os.getenv("INTERNAL_GRPC_SECRET", "")

# --- Redis & Task Queue ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Thresholds ---
AUTO_THRESHOLD = float(os.getenv("AUTO_THRESHOLD", "0.80"))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.50"))

# --- Logging ---
LOG_DIR = Path(
    os.getenv(
        "LOG_DIR",
        str(Path(__file__).resolve().parents[3] / "data" / "logs"),
    )
)
LOG_FILE = Path(os.getenv("LOG_FILE", str(LOG_DIR / "backend.log")))
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")

# --- Authentication & Access Control ---
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"
ENVIRONMENT = "production"

if not API_KEY:
    raise RuntimeError("A non-empty MATCHOPS_API_KEY must be configured in environment variables.")
