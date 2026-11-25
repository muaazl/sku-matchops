"""Vector collection inspection and query pipeline."""

from __future__ import annotations

import json
from typing import Any

from engine.pipelines.runtime_pipeline import _get_shared_models
from engine.preprocessing.text_normalizer import normalize_sku_text, standardize_sku_units
from engine.vector_store.vector_store import VectorStore


def _create_vector_store_handle() -> VectorStore:
    vs = VectorStore.__new__(VectorStore)
    vs.client = VectorStore.get_qdrant_client()
    return vs


def list_vector_collections() -> list[dict[str, Any]]:
    """Lists vector collections with point counts and status."""
    return _create_vector_store_handle().list_collection_details()


def get_vector_collection_detail(name: str) -> dict[str, Any]:
    """Fetches schema and point-count metadata for one collection."""
    return _create_vector_store_handle().fetch_collection_details(name)


def search_vector_collection(
    collection_name: str,
    query: str,
    mode: str = "dense",
    filters_json: str = "",
    score_threshold: float = 0.0,
    has_score_threshold: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    """Runs embedding/normalization and searches a Qdrant collection."""
    effective_mode = mode or "dense"
    filters_dict = json.loads(filters_json) if filters_json else {}
    if isinstance(filters_dict, dict) and "_search_mode" in filters_dict:
        effective_mode = filters_dict.pop("_search_mode")
        cleaned_filters_json = json.dumps(filters_dict) if filters_dict else ""
    else:
        cleaned_filters_json = filters_json

    VectorStore.build_qdrant_filter(cleaned_filters_json)

    applied_threshold = (
        score_threshold if (has_score_threshold and effective_mode in ("dense", "sparse")) else None
    )
    vs = _create_vector_store_handle()
    embed_engine, _ = _get_shared_models()

    raw_query = (query or "").strip()
    normalized_query = normalize_sku_text(standardize_sku_units(raw_query))
    is_catalog = collection_name.endswith("_catalog") or not collection_name.endswith("_tags")
    dense_target = normalized_query if (is_catalog and normalized_query) else raw_query

    encoded = embed_engine.encode([dense_target])
    dense_vec = encoded["dense"][0]
    bm25_list = encoded.get("bm25") or [{}]
    bm25_dict = bm25_list[0] if bm25_list else {}
    sparse_list = encoded.get("sparse") or []
    sparse_dict = sparse_list[0] if sparse_list else {}

    hits = vs.search_collection_points(
        collection_name=collection_name,
        dense_vec=dense_vec,
        bm25_vec_dict=bm25_dict,
        sparse_vec_dict=sparse_dict,
        mode=effective_mode,
        normalized_query=normalized_query,
        filters_json=cleaned_filters_json,
        score_threshold=applied_threshold,
        limit=limit or 10,
    )
    return {"results": hits}
