"""Pydantic schemas for Qdrant vector search proxy requests."""

from typing import Literal

from pydantic import BaseModel


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict | None = None
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
