import os

# Override in .env for production
API_KEY = os.getenv("MATCHOPS_API_KEY", "meilimasterkey")
APPSCRIPT_WEBHOOK_URL = os.getenv("APPSCRIPT_WEBHOOK_URL", "")

# --- Authentication / RBAC ---
# Off (dev default): unauthenticated requests act as an admin of the X-Tenant-ID tenant
# ('default' if omitted). On: every non-public endpoint needs a Bearer JWT or X-API-Key.
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
# Required when AUTH_ENABLED; otherwise a random per-process secret is used (tokens die on restart)
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
DEFAULT_TENANT = "default"
