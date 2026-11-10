from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Connection

from backend.app.core.security import Principal, get_principal, require_rule_editor
from backend.app.db.session import get_db_connection
from backend.app.schemas.models import (
    RuleDraftTestRequest,
    RuleModel,
    RuleOperationResponse,
    RuleReorderRequest,
    RuleTestRequest,
)
from backend.app.services import rules_service
from backend.app.services.engine_client import test_rule_via_engine

router = APIRouter()


@router.get("/rules", response_model=list[dict])
def get_rules(
    domain: str | None = None,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    return rules_service.get_rules_list(db, domain=domain, tenant_id=principal.tenant_id)


@router.post("/rules", response_model=RuleOperationResponse)
def create_rule(
    rule: RuleModel,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(require_rule_editor),
):
    try:
        new_id = rules_service.create_rule_entry(db, rule, tenant_id=principal.tenant_id)
        return RuleOperationResponse(message=f"Rule {new_id} created successfully.", rule_id=new_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {str(e)}")


@router.put("/rules/reorder", response_model=RuleOperationResponse)
def reorder_rules(
    payload: RuleReorderRequest,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(require_rule_editor),
):
    ordered_ids = payload.ordered_rule_ids
    if not ordered_ids:
        return RuleOperationResponse(message="No order specified.")

    try:
        rules_service.reorder_rules_entries(db, ordered_ids, tenant_id=principal.tenant_id)
        return RuleOperationResponse(message="Rules reordered successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reorder rules: {str(e)}")


@router.post("/rules/test-draft")
def test_rule_draft(request: RuleDraftTestRequest):
    rule_dict = request.rule.model_dump()
    rule_dict["conditions"] = [
        c if isinstance(c, dict) else c.model_dump() if hasattr(c, "model_dump") else dict(c)
        for c in rule_dict.get("conditions", [])
    ]
    rule_dict["actions"] = [
        a if isinstance(a, dict) else a.model_dump() if hasattr(a, "model_dump") else dict(a)
        for a in rule_dict.get("actions", [])
    ]
    return test_rule_via_engine(rule_dict, dict(request.sample_record))


@router.put("/rules/{rule_id}", response_model=RuleOperationResponse)
def update_rule(
    rule_id: str,
    rule: RuleModel,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(require_rule_editor),
):
    try:
        updated = rules_service.update_rule_entry(db, rule_id, rule, tenant_id=principal.tenant_id)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found.")
        return RuleOperationResponse(
            message=f"Rule {rule_id} updated successfully.", rule_id=rule_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update rule: {str(e)}")


@router.delete("/rules/{rule_id}", response_model=RuleOperationResponse)
def delete_rule(
    rule_id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(require_rule_editor),
):
    try:
        deleted = rules_service.delete_rule_entry(db, rule_id, tenant_id=principal.tenant_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found.")
        return RuleOperationResponse(
            message=f"Rule {rule_id} deleted successfully.", rule_id=rule_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete rule: {str(e)}")


@router.post("/rules/{rule_id}/test")
def test_rule(
    rule_id: str,
    request: RuleTestRequest,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    rule_dict = rules_service.get_rule_by_id(db, rule_id, tenant_id=principal.tenant_id)
    if not rule_dict:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found.")

    return test_rule_via_engine(
        rule_dict, dict(request.sample_record), tenant_id=principal.tenant_id
    )
