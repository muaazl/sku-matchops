from fastapi import APIRouter, HTTPException
from backend.app.schemas.models import VectorSearchRequest
from backend.app.services.engine_client import (
    list_vector_collections,
    get_vector_collection,
    search_vector_collection,
)

router = APIRouter()


@router.get("/vector-db/collections")
def list_collections():
    try:
        return list_vector_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector-db/collections/{name}")
def get_collection(name: str):
    try:
        return get_vector_collection(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector-db/collections/{name}/search")
def search_collection(name: str, request: VectorSearchRequest):
    try:
        payload = request.model_dump()
        return search_vector_collection(name, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
