import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.engine_client import (
    run_single as engine_run_single,
    suggest_tags as engine_suggest_tags,
    run_sku_audit as engine_run_sku_audit
)
from engine.rules_engine import run_rules_engine

router = APIRouter(prefix="/interactive")


class RunSingleRequest(BaseModel):
    sku_name: str
    domain: str
    task: str
    price: float | None = 0.0
    description: str | None = ""
    category: str | None = ""


class RerunRulesRequest(BaseModel):
    sku_name: str
    domain: str
    bt: str
    gk: str
    region: str | None = ""
    category: str | None = ""
    price: float | None = 0.0
    confidence: float | None = 1.0
    match_source: str | None = "classifier"
    matched_sku: str | None = ""
    reasoning: str | None = ""


@router.post("/run-single")
def run_single(request: RunSingleRequest):
    try:
        res = engine_run_single(
            sku_name=request.sku_name,
            domain=request.domain,
            task=request.task,
            price=request.price or 0.0,
            description=request.description or "",
            category=request.category or ""
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rerun-rules")
def rerun_rules(request: RerunRulesRequest):
    try:
        gk_list = [x.strip() for x in request.gk.split(",") if x.strip()]
        
        # Prepare record for rules engine
        record = {
            "sku_name": request.sku_name,
            "domain": request.domain,
            "bt": request.bt,
            "gk": gk_list,
            "region": request.region if request.domain == "food" else None,
            "category": request.category if request.domain == "market" else None,
            "price": request.price,
            "confidence": request.confidence,
            "match_source": request.match_source,
            "matched_sku": request.matched_sku,
            "reasoning": request.reasoning,
            "rules_applied": []
        }
        
        aug_record = run_rules_engine(record)
        
        rules_applied = aug_record.get("rules_applied", [])
        rules_applied_str = json.dumps(rules_applied) if rules_applied else ""
        
        return {
            "suggested_bt": aug_record.get("bt") or "",
            "suggested_gk": ", ".join(aug_record.get("gk", [])),
            "suggested_region": (aug_record.get("region") or aug_record.get("category") or ""),
            "rules_applied": rules_applied_str
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SuggestRequest(BaseModel):
    sku_name: str
    domain: str
    current_bt: str | None = ""
    exclude_bt: str | None = ""
    exclude_gk: str | None = ""


@router.post("/suggest")
def suggest_tags(request: SuggestRequest):
    try:
        return engine_suggest_tags(
            sku_name=request.sku_name,
            domain=request.domain,
            current_bt=request.current_bt or "",
            exclude_bt=request.exclude_bt or "",
            exclude_gk=request.exclude_gk or ""
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit")
def run_audit(request: RunSingleRequest):
    try:
        return engine_run_sku_audit(
            sku_name=request.sku_name,
            domain=request.domain,
            task=request.task,
            price=request.price or 0.0,
            description=request.description or "",
            category=request.category or ""
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
