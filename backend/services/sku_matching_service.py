"""SKU matching, classification, pipeline dispatch, and interactive inference orchestration."""

from typing import Any

from backend.schemas.sku_schema import BaseRequest
from backend.services import engine_gateway_service, job_service


def enqueue_sku_job(request: BaseRequest, task: str) -> dict[str, Any]:
    """Enqueues an async matcher, classifier, or pipeline SKU batch job."""
    return job_service.enqueue_job(request, task)


def run_single_sku_inference(
    *,
    sku_name: str,
    domain: str,
    task: str,
    price: float = 0.0,
    description: str = "",
    category: str = "",
) -> dict[str, Any]:
    """Runs synchronous single-SKU inference via the ML Engine gRPC gateway."""
    return engine_gateway_service.run_single(
        sku_name=sku_name,
        domain=domain,
        task=task,
        price=price,
        description=description,
        category=category,
    )


def suggest_sku_tags(
    *,
    sku_name: str,
    domain: str,
    current_bt: str = "",
    exclude_bt: str = "",
    exclude_gk: str = "",
) -> dict[str, Any]:
    """Requests template-aware tag suggestions via the ML Engine gRPC gateway."""
    return engine_gateway_service.suggest_tags(
        sku_name=sku_name,
        domain=domain,
        current_bt=current_bt,
        exclude_bt=exclude_bt,
        exclude_gk=exclude_gk,
    )


def audit_single_sku(
    *,
    sku_name: str,
    domain: str,
    task: str,
    price: float = 0.0,
    description: str = "",
    category: str = "",
) -> dict[str, Any]:
    """Runs diagnostic single-SKU audit via the ML Engine gRPC gateway."""
    return engine_gateway_service.run_sku_audit(
        sku_name=sku_name,
        domain=domain,
        task=task,
        price=price,
        description=description,
        category=category,
    )
