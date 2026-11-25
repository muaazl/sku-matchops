"""Root FastAPI API router assembling all domain endpoint routers and global auth."""

from fastapi import APIRouter, Depends

from backend.api.endpoints import (
    batch_router,
    catalog_router,
    history_router,
    interactive_router,
    job_router,
    model_router,
    queue_router,
    request_log_router,
    rule_router,
    sku_router,
    system_router,
    upload_router,
    vector_db_router,
)
from backend.core.security import get_principal

_authenticated = [Depends(get_principal)]

api_router = APIRouter()

api_router.include_router(system_router.router)
api_router.include_router(sku_router.router, dependencies=_authenticated)
api_router.include_router(interactive_router.router, dependencies=_authenticated)
api_router.include_router(job_router.router, dependencies=_authenticated)
api_router.include_router(batch_router.router, dependencies=_authenticated)
api_router.include_router(catalog_router.router, dependencies=_authenticated)
api_router.include_router(history_router.router, dependencies=_authenticated)
api_router.include_router(model_router.router, dependencies=_authenticated)
api_router.include_router(vector_db_router.router, dependencies=_authenticated)
api_router.include_router(queue_router.router, dependencies=_authenticated)
api_router.include_router(request_log_router.router, dependencies=_authenticated)
api_router.include_router(rule_router.router, dependencies=_authenticated)
api_router.include_router(upload_router.router, dependencies=_authenticated)
