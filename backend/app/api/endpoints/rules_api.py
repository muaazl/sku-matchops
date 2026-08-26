from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List
import sqlite3

from backend.app.core.db import get_db_connection
from backend.app.schemas.models import (
    RuleModel,
    RuleTestRequest,
    RuleDraftTestRequest,
    RuleReorderRequest,
    RuleOperationResponse,
)
from backend.app.services import rules_service
from engine.rules_engine.loader import Rule
from engine.rules_engine.evaluator import evaluate_conditions
from engine.rules_engine.actions import apply_actions

router = APIRouter()


@router.get("/rules", response_model=List[dict])
def get_rules(
    domain: Optional[str] = None,
    module: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    return rules_service.get_rules_list(db, domain=domain, module=module)


@router.post("/rules", response_model=RuleOperationResponse)
def create_rule(
    rule: RuleModel,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    try:
        new_id = rules_service.create_rule_entry(db, rule)
        return RuleOperationResponse(
            message=f"Rule {new_id} created successfully.",
            rule_id=new_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {str(e)}")


@router.put("/rules/reorder", response_model=RuleOperationResponse)
def reorder_rules(
    payload: RuleReorderRequest,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    ordered_ids = payload.ordered_rule_ids
    if not ordered_ids:
        return RuleOperationResponse(message="No order specified.")
        
    try:
        rules_service.reorder_rules_entries(db, ordered_ids)
        return RuleOperationResponse(message="Rules reordered successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reorder rules: {str(e)}")


@router.post("/rules/test-draft")
def test_rule_draft(request: RuleDraftTestRequest):
    rule_dict = request.rule.model_dump()
    row_values = (
        0,
        rule_dict["rule_id"],
        rule_dict["domain"],
        rule_dict["module"],
        rule_dict["priority"],
        rule_dict["description"],
        rule_dict["reasoning"],
        rule_dict["condition_logic"],
        rule_dict["is_active"]
    )
    engine_rule = Rule(row_values)
    engine_rule.conditions = [
        c if isinstance(c, dict) else c.model_dump() if hasattr(c, 'model_dump') else dict(c)
        for c in rule_dict.get("conditions", [])
    ]
    engine_rule.actions = [
        a if isinstance(a, dict) else a.model_dump() if hasattr(a, 'model_dump') else dict(a)
        for a in rule_dict.get("actions", [])
    ]
    
    record = dict(request.sample_record)
    fires = evaluate_conditions(engine_rule, record)
    changes = ""
    if fires:
        changes = apply_actions(engine_rule, record)
        
    return {
        "rule_id": rule_dict["rule_id"],
        "fires": fires,
        "matched_conditions": engine_rule.conditions if fires else [],
        "actions_applied": engine_rule.actions if fires else [],
        "changes": changes,
        "sample_record_after": record
    }


@router.put("/rules/{rule_id}", response_model=RuleOperationResponse)
def update_rule(
    rule_id: str,
    rule: RuleModel,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    try:
        updated = rules_service.update_rule_entry(db, rule_id, rule)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found.")
        return RuleOperationResponse(
            message=f"Rule {rule_id} updated successfully.",
            rule_id=rule_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update rule: {str(e)}")


@router.delete("/rules/{rule_id}", response_model=RuleOperationResponse)
def delete_rule(
    rule_id: str,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    try:
        deleted = rules_service.delete_rule_entry(db, rule_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found.")
        return RuleOperationResponse(
            message=f"Rule {rule_id} deleted successfully.",
            rule_id=rule_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete rule: {str(e)}")


@router.post("/rules/{rule_id}/test")
def test_rule(
    rule_id: str,
    request: RuleTestRequest,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    rule_dict = rules_service.get_rule_by_id(db, rule_id)
    if not rule_dict:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found.")
    
    row_values = (
        rule_dict.get("id", 0),
        rule_dict["rule_id"],
        rule_dict["domain"],
        rule_dict["module"],
        rule_dict["priority"],
        rule_dict["description"],
        rule_dict["reasoning"],
        rule_dict["condition_logic"],
        rule_dict["is_active"]
    )
    engine_rule = Rule(row_values)
    engine_rule.conditions = rule_dict.get("conditions", [])
    engine_rule.actions = rule_dict.get("actions", [])
    
    record = dict(request.sample_record)
    fires = evaluate_conditions(engine_rule, record)
    changes = ""
    if fires:
        changes = apply_actions(engine_rule, record)
        
    return {
        "rule_id": rule_id,
        "fires": fires,
        "matched_conditions": rule_dict["conditions"] if fires else [],
        "actions_applied": rule_dict["actions"] if fires else [],
        "changes": changes,
        "sample_record_after": record
    }
