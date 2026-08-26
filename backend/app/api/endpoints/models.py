import logging
from fastapi import APIRouter

from backend.app.services.engine_client import (
    get_models_status,
    reload_engine_models,
    trigger_load_models,
)

logger = logging.getLogger("matchops.models_api")
router = APIRouter()


@router.post("/load-models")
def load_models():
    """Triggers ML model pre-loading on the ML Engine microservice."""
    return trigger_load_models()


@router.get("/models-status")
def check_models_status():
    """Checks model loading statuses from the ML Engine microservice."""
    return get_models_status()


@router.post("/models/reload")
def reload_models():
    """Forces cache clearing and model reload on the ML Engine microservice."""
    return reload_engine_models()
