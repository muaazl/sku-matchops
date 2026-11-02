import json

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.security import Principal, get_principal
from backend.app.schemas.models import EnqueueJobResponse, UploadRequest
from backend.app.services.engine_client import dispatch_upload_job
from backend.app.services.worker import register_and_dispatch_job

router = APIRouter()


@router.post("/upload", response_model=EnqueueJobResponse)
def trigger_upload(request: UploadRequest, principal: Principal = Depends(get_principal)):
    if not request.skus:
        raise HTTPException(status_code=400, detail="'skus' list is empty.")

    try:
        return register_and_dispatch_job(
            job_type="upload",
            domain=request.domain or "market",
            total_items=len(request.skus),
            input_skus_json=json.dumps(request.skus),
            tenant_id=principal.tenant_id,
            sheet_name=None,
            target_sheet=request.outlet_id_or_name,
            dispatch_fn=lambda job_id: dispatch_upload_job(
                job_id=job_id, request=request, task="upload", tenant_id=principal.tenant_id
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
