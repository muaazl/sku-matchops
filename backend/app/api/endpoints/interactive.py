from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.core.security import Principal, get_principal
from backend.app.services.engine_client import rerun_rules_via_engine
from backend.app.services.engine_client import run_single as engine_run_single
from backend.app.services.engine_client import run_sku_audit as engine_run_sku_audit
from backend.app.services.engine_client import suggest_tags as engine_suggest_tags

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
    match_source: str | None = "Classifier"
    matched_sku: str | None = ""
    reasoning: str | None = ""


@router.post("/run-single")
def run_single(request: RunSingleRequest, principal: Principal = Depends(get_principal)):
    try:
        res = engine_run_single(
            tenant_id=principal.tenant_id,
            sku_name=request.sku_name,
            domain=request.domain,
            task=request.task,
            price=request.price or 0.0,
            description=request.description or "",
            category=request.category or "",
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rerun-rules")
def rerun_rules(request: RerunRulesRequest, principal: Principal = Depends(get_principal)):
    try:
        return rerun_rules_via_engine(
            sku_name=request.sku_name,
            domain=request.domain,
            bt=request.bt,
            gk=request.gk,
            region=request.region or "",
            category=request.category or "",
            price=request.price or 0.0,
            confidence=request.confidence or 1.0,
            match_source=request.match_source or "Classifier",
            matched_sku=request.matched_sku or "",
            reasoning=request.reasoning or "",
            tenant_id=principal.tenant_id,
        )
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
            exclude_gk=request.exclude_gk or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit")
def run_audit(request: RunSingleRequest, principal: Principal = Depends(get_principal)):
    try:
        return engine_run_sku_audit(
            tenant_id=principal.tenant_id,
            sku_name=request.sku_name,
            domain=request.domain,
            task=request.task,
            price=request.price or 0.0,
            description=request.description or "",
            category=request.category or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
