"""Vector database collection inspection and search proxy endpoints."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.core.exceptions import (
    ResourceNotFoundError,
    ServiceValidationError,
    UpstreamGatewayError,
)
from backend.core.security import require_platform_admin
from backend.schemas.vector_db_schema import VectorSearchRequest
from backend.services import vector_db_service

router = APIRouter(
    prefix="/vector-db",
    tags=["Vector DB"],
    dependencies=[Depends(require_platform_admin)],
)


def _execute_vector_operation(fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
    try:
        return fn(*args)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (UpstreamGatewayError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/collections")
def list_vector_collections() -> dict[str, Any]:
    """Lists vector collections registered in Qdrant via the ML Engine."""
    return _execute_vector_operation(vector_db_service.list_collections)


@router.get("/collections/{name}")
def get_vector_collection(name: str) -> dict[str, Any]:
    """Fetches metadata and vector configuration for a Qdrant collection."""
    return _execute_vector_operation(vector_db_service.fetch_collection_detail, name)


@router.post("/collections/{name}/search")
def search_vector_collection(name: str, payload: VectorSearchRequest) -> dict[str, Any]:
    """Executes dense, sparse, or hybrid vector similarity search via the ML Engine."""
    return _execute_vector_operation(
        vector_db_service.search_collection, name, payload.model_dump()
    )
