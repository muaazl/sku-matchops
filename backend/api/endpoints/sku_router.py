"""SKU matching, classification, and pipeline dispatch endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import RateLimiter
from backend.core.exceptions import ServiceValidationError
from backend.schemas.job_schema import EnqueueJobResponse
from backend.schemas.sku_schema import (
    BaseRequest,
    ClassifyRequest,
    MatchRequest,
    PipelineRequest,
)
from backend.services import sku_matching_service

router = APIRouter(tags=["Inference"])

_inference_rate_limiter = [Depends(RateLimiter(max_requests=20, window_seconds=60))]


def _enqueue_batch_task(request: BaseRequest, task: str) -> EnqueueJobResponse:
    try:
        result = sku_matching_service.enqueue_sku_job(request, task=task)
        return EnqueueJobResponse(**result)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/match", response_model=EnqueueJobResponse, dependencies=_inference_rate_limiter)
def match_skus(request: MatchRequest) -> EnqueueJobResponse:
    """Enqueues an asynchronous catalog matching job."""
    return _enqueue_batch_task(request, "matcher")


@router.post("/classify", response_model=EnqueueJobResponse, dependencies=_inference_rate_limiter)
def classify_skus(request: ClassifyRequest) -> EnqueueJobResponse:
    """Enqueues an asynchronous taxonomy classification job."""
    return _enqueue_batch_task(request, "classifier")


@router.post("/pipeline", response_model=EnqueueJobResponse, dependencies=_inference_rate_limiter)
def run_sku_pipeline(request: PipelineRequest) -> EnqueueJobResponse:
    """Enqueues an asynchronous hybrid matching and classification pipeline job."""
    return _enqueue_batch_task(request, "pipeline")
