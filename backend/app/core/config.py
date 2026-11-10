import os

from dotenv import load_dotenv

load_dotenv()

# Override in .env for production
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
API_KEY = os.getenv("MATCHOPS_API_KEY", "bismillah")
APPSCRIPT_WEBHOOK_URL = os.getenv("APPSCRIPT_WEBHOOK_URL", "")
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
AUTO_THRESHOLD = float(os.getenv("AUTO_THRESHOLD", "0.85"))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.60"))

# --- Logging ---
LOG_DIR = os.getenv(
    "LOG_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "logs"
    ),
)
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "backend.log"))
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")

# --- Authentication / RBAC ---
# Off (dev default): unauthenticated requests act as an admin of the X-Tenant-ID tenant
# ('default' if omitted). On: every non-public endpoint needs a Bearer JWT or X-API-Key.
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
# Required when AUTH_ENABLED; otherwise a random per-process secret is used (tokens die on restart)
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
DEFAULT_TENANT = "default"

if ENVIRONMENT in ("production", "prod"):
    if not AUTH_ENABLED:
        raise RuntimeError("AUTH_ENABLED must be true in production environments.")
    if not JWT_SECRET or JWT_SECRET == "change_me_in_production":
        raise RuntimeError("A secure JWT_SECRET must be configured in production environments.")
    if not API_KEY or API_KEY == "bismillah":
        raise RuntimeError(
            "A secure, non-default MATCHOPS_API_KEY must be configured in production environments."
        )
    if not INTERNAL_GRPC_SECRET:
        raise RuntimeError("INTERNAL_GRPC_SECRET must be configured in production environments.")
