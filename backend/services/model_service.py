"""ML Engine model lifecycle orchestration service."""

from typing import Any

from backend.services import engine_gateway_service


def load_engine_models() -> dict[str, Any]:
    """Triggers background ML model pre-loading on the Engine microservice."""
    return engine_gateway_service.trigger_load_models()


def fetch_models_status() -> dict[str, Any]:
    """Fetches domain model load status from the Engine microservice."""
    return engine_gateway_service.get_models_status()


def reload_engine_models() -> dict[str, Any]:
    """Forces cache invalidation and model reload on the Engine microservice."""
    return engine_gateway_service.reload_engine_models()
