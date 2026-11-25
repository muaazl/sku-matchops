"""System observability, health, readiness, and log-tail endpoints."""

from fastapi import APIRouter, Depends, Response

from backend.core.observability import render_metrics_payload
from backend.core.security import require_platform_admin
from backend.services import system_service

router = APIRouter(tags=["System"])


@router.get("/metrics")
def get_prometheus_metrics() -> Response:
    """Exposes Gateway Prometheus metrics in exposition format."""
    payload, content_type = render_metrics_payload()
    return Response(content=payload, media_type=content_type)


@router.get("/health")
def get_health_status() -> dict:
    """Returns basic liveness status alongside ML Engine domain readiness."""
    return system_service.check_system_health()


@router.get("/ready")
def get_readiness_status(response: Response) -> dict:
    """Verifies PostgreSQL, Redis, and ML Engine model readiness."""
    is_ready, payload = system_service.check_system_readiness()
    response.status_code = 200 if is_ready else 503
    return payload


@router.get("/logs", dependencies=[Depends(require_platform_admin)])
def get_application_logs(lines: int = 500) -> dict[str, str]:
    """Reads trailing lines from the Backend application log file."""
    return system_service.read_application_logs(lines=lines)
