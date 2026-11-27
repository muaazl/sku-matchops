"""Deterministic business rules CRUD, priority reordering, and test endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.core.exceptions import ResourceNotFoundError, ServiceValidationError
from backend.core.security import require_rule_editor
from backend.schemas.rule_schema import (
    RuleDraftTestRequest,
    RuleModel,
    RuleOperationResponse,
    RuleReorderRequest,
    RuleTestRequest,
)
from backend.services import rule_service

router = APIRouter(tags=["Rules"])

_rule_editor = [Depends(require_rule_editor)]


@router.get("/rules")
def list_rules(domain: str | None = None) -> list[dict[str, Any]]:
    """Lists stored business rules with nested conditions and actions."""
    return rule_service.list_rules(domain=domain)


@router.post("/rules", response_model=RuleOperationResponse, dependencies=_rule_editor)
def create_rule(rule: RuleModel) -> RuleOperationResponse:
    """Creates a new business rule and broadcasts snapshot invalidation."""
    try:
        rule_id = rule_service.create_rule(rule)
        return RuleOperationResponse(message="Rule created", rule_id=rule_id)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/rules/reorder", response_model=RuleOperationResponse, dependencies=_rule_editor)
def reorder_rules(request: RuleReorderRequest) -> RuleOperationResponse:
    """Reorders rule priorities sequentially and syncs updated rules."""
    reordered = rule_service.reorder_rules(request.ordered_rule_ids)
    if not reordered:
        raise HTTPException(status_code=400, detail="No rule IDs provided to reorder.")
    return RuleOperationResponse(message="Rules reordered")


@router.post("/rules/test-draft", dependencies=_rule_editor)
def test_rule_draft(request: RuleDraftTestRequest) -> dict[str, Any]:
    """Tests an unsaved rule definition against a sample SKU record."""
    try:
        return rule_service.test_rule_draft(request.rule, request.sample_record)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/rules/{rule_id}", response_model=RuleOperationResponse, dependencies=_rule_editor)
def update_rule(rule_id: str, rule: RuleModel) -> RuleOperationResponse:
    """Updates a persisted rule definition and broadcasts snapshot invalidation."""
    try:
        rule_service.update_rule(rule_id, rule)
        return RuleOperationResponse(message="Rule updated", rule_id=rule_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/rules/{rule_id}", response_model=RuleOperationResponse, dependencies=_rule_editor)
def delete_rule(rule_id: str) -> RuleOperationResponse:
    """Deletes a persisted rule definition and broadcasts snapshot invalidation."""
    try:
        rule_service.delete_rule(rule_id)
        return RuleOperationResponse(message="Rule deleted", rule_id=rule_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/rules/{rule_id}/test", dependencies=_rule_editor)
def test_rule(rule_id: str, request: RuleTestRequest) -> dict[str, Any]:
    """Tests an existing stored rule against a sample SKU record."""
    try:
        return rule_service.test_stored_rule(rule_id, request.sample_record)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
