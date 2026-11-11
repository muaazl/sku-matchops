import hmac
import logging
import os
from collections import deque

from fastapi import APIRouter, Depends, HTTPException, Query, Response, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import text

from backend.app.api.endpoints.batches import router as batches_router
from backend.app.api.endpoints.catalog import router as catalog_router
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
from backend.app.core.observability import metrics_payload
from backend.app.core.rate_limit import RateLimiter
from backend.app.core.security import Principal, get_principal, require_platform_admin
from backend.app.db.session import get_engine
from backend.app.schemas.models import (
    ClassifyRequest,
    EnqueueJobResponse,
    MatchRequest,
    PipelineRequest,
)
from backend.app.services.engine_client import get_engine_health
from backend.app.services.worker import enqueue_job

logger = logging.getLogger("matchops.api.routes")

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
):
    api_router.include_router(_router, tags=[_tag], dependencies=_authenticated)

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


@api_router.get("/ready")
def ready(response: Response):
    """Deep readiness probe checking database, Redis broker, and ML Engine health."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    redis_ok = True
    try:
        import redis

        from backend.app.core import config as backend_config

        if backend_config.REDIS_URL:
            r = redis.Redis.from_url(backend_config.REDIS_URL, socket_timeout=1.0)
            redis_ok = bool(r.ping())
    except Exception:
        redis_ok = False

    engine_status = get_engine_health()
    engine_ok = bool(engine_status.get("all_models_ready"))

    is_ready = db_ok and redis_ok and engine_ok
    if not is_ready:
        response.status_code = 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "db": db_ok,
        "redis": redis_ok,
        "engine": engine_status,
    }


_expensive_rate_limiter = RateLimiter(max_requests=120, window_seconds=60)


@api_router.post(
    "/match",
    response_model=EnqueueJobResponse,
    dependencies=[Depends(_expensive_rate_limiter)],
)
def match(request: MatchRequest, principal: Principal = Depends(get_principal)):
    """
    Match a batch of SKUs — fully async.
    Dispatched to the ML Engine microservice.
    """
    try:
        return enqueue_job(request, "matcher")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error enqueuing matcher job: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred while processing request."
        )


@api_router.post(
    "/classify",
    response_model=EnqueueJobResponse,
    dependencies=[Depends(_expensive_rate_limiter)],
)
def classify(request: ClassifyRequest, principal: Principal = Depends(get_principal)):
    """
    Classify a batch of SKUs — fully async.
    Dispatched to the ML Engine microservice.
    """
    try:
        return enqueue_job(request, "classifier")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error enqueuing classifier job: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred while processing request."
        )


@api_router.post(
    "/pipeline",
    response_model=EnqueueJobResponse,
    dependencies=[Depends(_expensive_rate_limiter)],
)
def pipeline(request: PipelineRequest, principal: Principal = Depends(get_principal)):
    """
    Run pipeline (Match + Classify fallback) on a batch of SKUs — fully async.
    Dispatched to the ML Engine microservice.
    """
    try:
        return enqueue_job(request, "pipeline")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error enqueuing pipeline job: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred while processing request."
        )


@api_router.get("/logs", dependencies=[Depends(require_platform_admin)])
def get_logs(lines: int = Query(500, ge=1, le=20000)):
    """
    Returns the last N lines of the application logs.
    """
    log_file = config.LOG_FILE

    if not os.path.exists(log_file):
        return {"logs": "Log file not found."}

    try:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            last_lines = deque(f, maxlen=lines)
            return {"logs": "".join(last_lines)}
    except Exception as e:
        return {"logs": f"Error reading log file: {str(e)}"}
