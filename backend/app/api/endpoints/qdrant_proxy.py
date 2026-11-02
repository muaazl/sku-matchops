from fastapi import APIRouter, HTTPException

from backend.app.schemas.models import VectorSearchRequest
from backend.app.services.engine_client import (
    get_vector_collection,
    list_vector_collections,
    search_vector_collection,
)

router = APIRouter()


def _call_vector_rpc(fn, *args):
    try:
        return fn(*args)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector-db/collections")
def list_collections():
    return _call_vector_rpc(list_vector_collections)


@router.get("/vector-db/collections/{name}")
def get_collection(name: str):
    return _call_vector_rpc(get_vector_collection, name)


@router.post("/vector-db/collections/{name}/search")
def search_collection(name: str, request: VectorSearchRequest):
    return _call_vector_rpc(search_vector_collection, name, request.model_dump())
