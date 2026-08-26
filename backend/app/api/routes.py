import os
from collections import deque
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from backend.app.api.endpoints.batches import router as batches_router
from backend.app.api.endpoints.catalog import router as catalog_router
from backend.app.api.endpoints.engine_callbacks import router as engine_callbacks_router
from backend.app.api.endpoints.history import router as history_router
from backend.app.api.endpoints.interactive import router as interactive_router
from backend.app.api.endpoints.jobs import router as jobs_router
from backend.app.api.endpoints.models import router as models_router
from backend.app.api.endpoints.qdrant_proxy import router as qdrant_proxy_router
from backend.app.api.endpoints.requests_log import router as requests_log_router
from backend.app.api.endpoints.rules_api import router as rules_router
from backend.app.core import config
from backend.app.schemas.models import (
    ClassifyRequest,
    EnqueueJobResponse,
    MatchRequest,
    PipelineRequest,
)
from backend.app.services.engine_client import get_engine_health
from backend.app.services.worker import enqueue_job
from engine.rules_engine.db.seed_rules import DB_PATH

api_router = APIRouter()

api_router.include_router(jobs_router, tags=["jobs"])
api_router.include_router(batches_router, tags=["batches"])
api_router.include_router(history_router, tags=["history"])
api_router.include_router(interactive_router, tags=["interactive"])
api_router.include_router(requests_log_router, tags=["requests_log"])
api_router.include_router(qdrant_proxy_router, tags=["qdrant_proxy"])
api_router.include_router(rules_router, tags=["rules_engine"])
api_router.include_router(catalog_router, tags=["catalog"])
api_router.include_router(models_router, tags=["models"])
api_router.include_router(engine_callbacks_router)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return api_key_header


@api_router.get("/health")
def health():
    engine_status = get_engine_health()
    loaded_domains = engine_status.get("loaded_domains", [])
    return {
        "status": "ok",
        "service": "matchops-backend",
        "loaded_domains": loaded_domains,
        "engine": engine_status
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


@api_router.get("/logs")
def get_logs(lines: int = 500):
    """
    Returns the last N lines of the application logs.
    """
    log_file = os.path.join(os.path.dirname(DB_PATH), "app.log")
    
    if not os.path.exists(log_file):
        return {"logs": "Log file not found."}
        
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            last_lines = deque(f, maxlen=lines)
            return {"logs": "".join(last_lines)}
    except Exception as e:
        return {"logs": f"Error reading log file: {str(e)}"}
