"""Vector database inspection and search proxy service."""

from typing import Any

from backend.services import engine_gateway_service


def list_collections() -> dict[str, Any]:
    """Lists available Qdrant collections via the Engine gRPC gateway."""
    return engine_gateway_service.list_vector_collections()


def fetch_collection_detail(name: str) -> dict[str, Any]:
    """Fetches metadata for a specific Qdrant collection via the Engine gRPC gateway."""
    return engine_gateway_service.get_vector_collection(name)


def search_collection(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Executes dense/sparse/hybrid vector search via the Engine gRPC gateway."""
    return engine_gateway_service.search_vector_collection(name, payload)
