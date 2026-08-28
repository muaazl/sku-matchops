import os

# Override in .env for production
API_KEY = os.getenv("MATCHOPS_API_KEY", "meilimasterkey")
APPSCRIPT_WEBHOOK_URL = os.getenv("APPSCRIPT_WEBHOOK_URL", "")
