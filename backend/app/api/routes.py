import hmac
import os
from collections import deque

from fastapi import APIRouter, Depends, HTTPException, Query, Response, Security
from fastapi.security.api_key import APIKeyHeader

from backend.app.api.endpoints.auth import router as auth_router
from backend.app.api.endpoints.auth import users_router
from backend.app.api.endpoints.batches import router as batches_router
from backend.app.api.endpoints.catalog import router as catalog_router
from backend.app.api.endpoints.engine_callbacks import router as engine_callbacks_router
from backend.app.api.endpoints.history import router as history_router
from backend.app.api.endpoints.interactive import router as interactive_router
from backend.app.api.endpoints.jobs import router as jobs_router
from backend.app.api.endpoints.models import router as models_router
from backend.app.api.endpoints.qdrant_proxy import router as qdrant_proxy_router
from backend.app.api.endpoints.queue import router as queue_router
from backend.app.api.endpoints.requests_log import router as requests_log_router
from backend.app.api.endpoints.rules_api import router as rules_router
from backend.app.api.endpoints.upload import router as upload_router
from backend.app.core import config
from backend.app.core.security import get_principal, require_platform_admin
from backend.app.schemas.models import (
    ClassifyRequest,
    EnqueueJobResponse,
    MatchRequest,
    PipelineRequest,
)
from backend.app.services.engine_client import get_engine_health
from backend.app.services.worker import enqueue_job
from engine import config as engine_config
from engine.core.observability import metrics_payload

api_router = APIRouter()

# Every operational router requires a resolved principal (a no-op in AUTH_ENABLED=false dev mode).
# Role and platform-admin checks are added per endpoint on top of this.
_authenticated = [Depends(get_principal)]
for _router, _tag in (
    (jobs_router, "jobs"),
    (batches_router, "batches"),
    (history_router, "history"),
    (interactive_router, "interactive"),
    (requests_log_router, "requests_log"),
    (qdrant_proxy_router, "qdrant_proxy"),
    (rules_router, "rules_engine"),
    (catalog_router, "catalog"),
    (models_router, "models"),
    (upload_router, "upload"),
    (queue_router, "queue"),
    (users_router, "auth"),
):
    api_router.include_router(_router, tags=[_tag], dependencies=_authenticated)
api_router.include_router(auth_router, tags=["auth"])  # public: login
# Internal Engine -> Gateway callbacks (private network, no end-user auth)
api_router.include_router(engine_callbacks_router)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key_header: str = Security(api_key_header)):
    if not hmac.compare_digest(api_key_header or "", config.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return api_key_header


@api_router.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus metrics for the Gateway process (queue depth / DLQ gauges included)."""
    body, content_type = metrics_payload()
    return Response(content=body, media_type=content_type)


@api_router.get("/health")
def health():
    engine_status = get_engine_health()
    loaded_domains = engine_status.get("loaded_domains", [])
    return {
        "status": "ok",
        "service": "matchops-backend",
        "loaded_domains": loaded_domains,
        "engine": engine_status,
    }


@api_router.post("/match", response_model=EnqueueJobResponse)
def match(request: MatchRequest, api_key: str = Depends(get_api_key)):
    """
    Match a batch of SKUs — fully async.
    Dispatched to the ML Engine microservice.
    """
    try:
        return enqueue_job(request, "matcher")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/classify", response_model=EnqueueJobResponse)
def classify(request: ClassifyRequest, api_key: str = Depends(get_api_key)):
    """
    Classify a batch of SKUs — fully async.
    Dispatched to the ML Engine microservice.
    """
    try:
        return enqueue_job(request, "classifier")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/pipeline", response_model=EnqueueJobResponse)
def pipeline(request: PipelineRequest, api_key: str = Depends(get_api_key)):
    """
    Run pipeline (Match + Classify fallback) on a batch of SKUs — fully async.
    Dispatched to the ML Engine microservice.
    """
    try:
        return enqueue_job(request, "pipeline")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/logs", dependencies=[Depends(require_platform_admin)])
def get_logs(lines: int = Query(500, ge=1, le=20000)):
    """
    Returns the last N lines of the application logs.
    """
    log_file = engine_config.LOG_FILE

    if not os.path.exists(log_file):
        return {"logs": "Log file not found."}

    try:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            last_lines = deque(f, maxlen=lines)
            return {"logs": "".join(last_lines)}
    except Exception as e:
        return {"logs": f"Error reading log file: {str(e)}"}
