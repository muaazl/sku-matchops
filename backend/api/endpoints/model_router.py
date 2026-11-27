"""ML Engine model warmup, status, and hot-reload endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.core.exceptions import UpstreamGatewayError
from backend.services import model_service

router = APIRouter(tags=["Models"])


@router.post("/load-models")
def load_models() -> dict[str, Any]:
    """Triggers background ML model loading on the Engine microservice."""
    try:
        return model_service.load_engine_models()
    except UpstreamGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models-status")
def get_models_status() -> dict[str, Any]:
    """Fetches domain model readiness from the ML Engine microservice."""
    return model_service.fetch_models_status()


@router.post("/models/reload")
def reload_models() -> dict[str, Any]:
    """Triggers cache invalidation and hot-reload of ML models on the Engine."""
    try:
        return model_service.reload_engine_models()
    except UpstreamGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
