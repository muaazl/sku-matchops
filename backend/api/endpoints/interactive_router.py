"""Interactive single-SKU inference, rule re-evaluation, suggestion, and audit endpoints."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.core.exceptions import (
    ResourceNotFoundError,
    ServiceValidationError,
    UpstreamGatewayError,
)
from backend.schemas.interactive_schema import (
    RerunRulesRequest,
    RunSingleRequest,
    SuggestRequest,
)
from backend.services import rule_service, sku_matching_service

router = APIRouter(prefix="/interactive", tags=["Interactive"])


def _handle_interactive_call(fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    try:
        return fn(**kwargs)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UpstreamGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/run-single")
def run_single_sku(request: RunSingleRequest) -> dict[str, Any]:
    """Runs synchronous single-SKU matcher, classifier, or hybrid pipeline inference."""
    return _handle_interactive_call(
        sku_matching_service.run_single_sku_inference,
        sku_name=request.sku_name,
        domain=request.domain,
        task=request.task,
        price=float(request.price or 0.0),
        description=request.description or "",
        category=request.category or "",
    )


@router.post("/rerun-rules")
def rerun_sku_rules(request: RerunRulesRequest) -> dict[str, Any]:
    """Re-applies active business rules on modified interactive classification outputs."""
    return _handle_interactive_call(
        rule_service.rerun_rules,
        sku_name=request.sku_name,
        domain=request.domain,
        bt=request.bt,
        gk=request.gk,
        region=request.region or "",
        category=request.category or "",
        price=float(request.price or 0.0),
        confidence=float(request.confidence if request.confidence is not None else 1.0),
        match_source=request.match_source or "Classifier",
        matched_sku=request.matched_sku or "",
        reasoning=request.reasoning or "",
    )


@router.post("/suggest")
def suggest_sku_tags(request: SuggestRequest) -> dict[str, Any]:
    """Suggests next Basic Type and Generic Keyword tags for a single SKU."""
    return _handle_interactive_call(
        sku_matching_service.suggest_sku_tags,
        sku_name=request.sku_name,
        domain=request.domain,
        current_bt=request.current_bt or "",
        exclude_bt=request.exclude_bt or "",
        exclude_gk=request.exclude_gk or "",
    )


@router.post("/audit")
def audit_single_sku(request: RunSingleRequest) -> dict[str, Any]:
    """Runs diagnostic stage-by-stage audit telemetry for a single SKU."""
    return _handle_interactive_call(
        sku_matching_service.audit_single_sku,
        sku_name=request.sku_name,
        domain=request.domain,
        task=request.task,
        price=float(request.price or 0.0),
        description=request.description or "",
        category=request.category or "",
    )
