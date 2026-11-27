"""Unified VectorStore facade composing client, collection, and query stores."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, Prefetch, ScoredPoint, SparseVector

from engine.core import config
from engine.vector_store.collection_store import (
    QdrantCollectionStore,
    cleanup_local_collection_dir,
    inspect_schema_mismatch,
    resolve_catalog_collection_name,
    resolve_tags_collection_name,
)
from engine.vector_store.qdrant_client_store import QdrantClientManager
from engine.vector_store.vector_query_store import (
    SEARCH_PAYLOAD,
    VectorQueryStore,
    build_hybrid_prefetch,
    build_point_vectors,
    build_qdrant_filter,
    convert_hit_to_dict,
    parse_sparse_dict,
    sanitize_catalog_payload,
)


class VectorStore:
    """Facade for Qdrant connection management, schema lifecycle, and hybrid retrieval."""

    _SEARCH_PAYLOAD = SEARCH_PAYLOAD

    def __init__(self) -> None:
        QdrantClientManager.ensure_local_storage_dir()
        self.client: QdrantClient = QdrantClientManager.get_qdrant_client()
        self._init_stores()

    def _init_stores(self) -> None:
        self._collection_store = QdrantCollectionStore(
            client_getter=lambda: self.client,
            retry_caller=self.execute_with_retry,
            checked_cache_getter=lambda: self.__dict__.get(
                "_checked_collections", QdrantClientManager._checked_collections
            ),
        )
        self._query_store = VectorQueryStore(
            client_getter=lambda: self.client,
            retry_caller=self.execute_with_retry,
            collection_store=self._collection_store,
        )

    def _get_collection_store(self) -> QdrantCollectionStore:
        store = getattr(self, "_collection_store", None)
        if store is None:
            self._init_stores()
            store = self._collection_store
        return store

    def _get_query_store(self) -> VectorQueryStore:
        store = getattr(self, "_query_store", None)
        if store is None:
            self._init_stores()
            store = self._query_store
        return store

    @classmethod
    def get_qdrant_client(
        cls, force_reconnect: bool = False, prefer_grpc: bool | None = None
    ) -> QdrantClient:
        """Returns or reinitializes the singleton QdrantClient instance."""
        return QdrantClientManager.get_qdrant_client(
            force_reconnect=force_reconnect, prefer_grpc=prefer_grpc
        )

    def execute_with_retry(
        self,
        operation_name: str,
        func: Callable[..., Any],
        *args: Any,
        max_retries: int = 5,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        **kwargs: Any,
    ) -> Any:
        """Executes a Qdrant call with transient retries, client rebinding, and REST fallback."""
        return QdrantClientManager.execute_with_retry(
            self.client,
            operation_name,
            func,
            *args,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            on_reconnect=lambda new_client: setattr(self, "client", new_client),
            **kwargs,
        )

    @staticmethod
    def resolve_catalog_collection_name(domain: str) -> str:
        """Returns the domain's catalog collection name."""
        return resolve_catalog_collection_name(domain)

    @staticmethod
    def resolve_tags_collection_name(domain: str) -> str:
        """Returns the domain's tags collection name."""
        return resolve_tags_collection_name(domain)

    @staticmethod
    def inspect_schema_mismatch(info: Any) -> str | None:
        """Checks Qdrant collection info against active embedding/sparse config."""
        return inspect_schema_mismatch(info)

    @staticmethod
    def cleanup_local_collection_dir(collection_name: str) -> None:
        """Removes orphaned local Qdrant collection storage directories if present."""
        cleanup_local_collection_dir(collection_name)

    def get_collection_schema_mismatch(self, collection_name: str) -> str | None:
        """Inspects an existing Qdrant collection for schema incompatibility."""
        return self._get_collection_store().get_collection_schema_mismatch(collection_name)

    def ensure_collection_ready(self, collection_name: str) -> None:
        """Creates the collection if missing and verifies payload indexes and HNSW config."""
        self._get_collection_store().ensure_collection_ready(collection_name)

    def create_payload_indexes(self, collection_name: str) -> None:
        """Creates keyword payload indexes for the specified collection."""
        self._get_collection_store().create_payload_indexes(collection_name)

    def create_versioned_collection(self, alias_name: str, version: int = 1) -> str:
        """Creates a versioned physical collection (`{alias_name}_v{version}`)."""
        return self._get_collection_store().create_versioned_collection(alias_name, version=version)

    def swap_collection_alias(self, alias_name: str, new_physical_collection: str) -> None:
        """Atomically swaps a collection alias to a new physical collection."""
        self._get_collection_store().swap_collection_alias(alias_name, new_physical_collection)

    def get_collection_schema_info(self, collection_name: str) -> dict[str, Any]:
        """Returns collection status, counts, and config metadata."""
        return self._get_collection_store().get_collection_schema_info(collection_name)

    def delete_named_collection(self, collection_name: str) -> None:
        """Deletes a named Qdrant collection and cleans up local bind-mount artifacts."""
        self._get_collection_store().delete_named_collection(collection_name)

    def list_collection_details(self) -> dict[str, Any]:
        """Lists all Qdrant collections and fetches serialized details."""
        return QdrantClientManager.list_collection_details(
            client_getter=lambda: self.client,
            retry_caller=self.execute_with_retry,
        )

    def fetch_collection_details(self, collection_name: str) -> dict[str, Any] | None:
        """Fetches serialized collection metadata for one collection."""
        return QdrantClientManager.fetch_collection_details(
            client_getter=lambda: self.client,
            retry_caller=self.execute_with_retry,
            collection_name=collection_name,
        )

    @staticmethod
    def parse_sparse_dict(sparse_dict: dict[str, float] | None) -> SparseVector:
        """Converts lexical sparse weight dict to a Qdrant SparseVector."""
        return parse_sparse_dict(sparse_dict)

    @staticmethod
    def sanitize_catalog_payload(raw_payload: dict[str, Any], domain: str) -> dict[str, Any]:
        """Cleans raw catalog row dicts for Qdrant payload storage."""
        return sanitize_catalog_payload(raw_payload, domain)

    @staticmethod
    def build_point_vectors(
        dense_vec: Any,
        bm25_dict: dict | None,
        sparse_dict: dict | None,
        include_learned_sparse: bool,
    ) -> dict[str, Any]:
        """Constructs point vector dict for Qdrant upsert."""
        return build_point_vectors(dense_vec, bm25_dict, sparse_dict, include_learned_sparse)

    @staticmethod
    def build_hybrid_prefetch(
        dense_vec: np.ndarray,
        bm25_dict: dict | None,
        sparse_dict: dict | None,
        limit: int,
    ) -> list[Prefetch]:
        """Constructs Dense + BM25 + optional sparse Prefetch requests."""
        return build_hybrid_prefetch(dense_vec, bm25_dict, sparse_dict, limit)

    @staticmethod
    def convert_hit_to_dict(hit: Any) -> dict[str, Any]:
        """Converts a Qdrant ScoredPoint hit to a dictionary."""
        return convert_hit_to_dict(hit)

    @staticmethod
    def build_qdrant_filter(filters_json: str) -> Filter | None:
        """Parses a JSON filter specification into a Qdrant Filter."""
        return build_qdrant_filter(filters_json)

    def verify_connection(self) -> Any:
        """Verifies Qdrant connectivity by listing collections with retry."""
        return self.execute_with_retry("get_collections", self.client.get_collections)

    def check_collection_exists(self, collection_name: str) -> bool:
        """Checks whether a Qdrant collection exists."""
        return self._get_query_store().check_collection_exists(collection_name)

    def count_collection_points(self, collection_name: str) -> int:
        """Counts stored points in a Qdrant collection."""
        return self._get_query_store().count_collection_points(collection_name)

    def upsert_catalog_points(
        self,
        df: pd.DataFrame,
        dense_embeddings: np.ndarray,
        sparse_weights: list[dict] | None = None,
        bm25_weights: list[dict] | None = None,
        domain: str = config.DOMAIN_MARKET,
    ) -> None:
        """Upserts catalog SKU vectors and sanitized payloads."""
        self._get_query_store().upsert_catalog_points(
            df,
            dense_embeddings,
            sparse_weights=sparse_weights,
            bm25_weights=bm25_weights,
            domain=domain,
        )

    def fetch_existing_hashes(self, domain: str = config.DOMAIN_MARKET) -> dict[str, str]:
        """Returns `{db_uid: row_hash}` for all catalog points stored in Qdrant."""
        return self._get_query_store().fetch_existing_hashes(domain=domain)

    def search_catalog_hybrid(
        self,
        dense_vec: np.ndarray,
        sparse_vec_dict: dict[str, float] | None = None,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bt_filter: str | None = None,
        bm25_vec_dict: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Executes single-query hybrid RRF catalog search."""
        return self._get_query_store().search_catalog_hybrid(
            dense_vec,
            sparse_vec_dict=sparse_vec_dict,
            domain=domain,
            top_k=top_k,
            bt_filter=bt_filter,
            bm25_vec_dict=bm25_vec_dict,
        )

    def search_catalog_hybrid_batch(
        self,
        dense_vecs: list[np.ndarray],
        sparse_vec_dicts: list[dict[str, float]] | None = None,
        bt_filters: list[str | None] | None = None,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bm25_vec_dicts: list[dict[str, float]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Executes batch hybrid RRF catalog search."""
        return self._get_query_store().search_catalog_hybrid_batch(
            dense_vecs,
            sparse_vec_dicts=sparse_vec_dicts,
            bt_filters=bt_filters,
            domain=domain,
            top_k=top_k,
            bm25_vec_dicts=bm25_vec_dicts,
        )

    def search_catalog_neighbors(
        self,
        dense_vec: np.ndarray,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        """Retrieves nearest catalog SKUs by dense similarity."""
        return self._get_query_store().search_catalog_neighbors(
            dense_vec, domain=domain, top_k=top_k
        )

    def search_batch_catalog_neighbors(
        self,
        dense_vecs: list[np.ndarray],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = 15,
    ) -> list[list[dict[str, Any]]]:
        """Batch-retrieves nearest catalog SKUs by dense similarity."""
        return self._get_query_store().search_batch_catalog_neighbors(
            dense_vecs, domain=domain, top_k=top_k
        )

    def check_tags_synced(self, domain: str, dict_type: str, expected_count: int) -> bool:
        """Checks whether Qdrant already stores `expected_count` tags for `dict_type`."""
        return self._get_query_store().check_tags_synced(domain, dict_type, expected_count)

    def upsert_tag_vectors(
        self,
        tags: list[str],
        dense_embs: np.ndarray,
        sparse_embs: list[dict] | None = None,
        dict_type: str = "",
        domain: str = config.DOMAIN_MARKET,
        metadata_list: list[dict] | None = None,
        force: bool = False,
        bm25_embs: list[dict] | None = None,
    ) -> None:
        """Upserts dictionary tags and their multi-vector embeddings."""
        self._get_query_store().upsert_tag_vectors(
            tags,
            dense_embs,
            sparse_embs=sparse_embs,
            dict_type=dict_type,
            domain=domain,
            metadata_list=metadata_list,
            force=force,
            bm25_embs=bm25_embs,
        )

    def search_tags_hybrid(
        self,
        dense_query: np.ndarray,
        sparse_query: dict[str, float] | None = None,
        limit: int = 50,
        filter_dict_type: str | None = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags: list[str] | None = None,
        bm25_query: dict[str, float] | None = None,
    ) -> tuple[list[ScoredPoint], list[ScoredPoint], list[ScoredPoint]]:
        """Executes single-query hybrid tag retrieval."""
        return self._get_query_store().search_tags_hybrid(
            dense_query,
            sparse_query=sparse_query,
            limit=limit,
            filter_dict_type=filter_dict_type,
            domain=domain,
            allowed_tags=allowed_tags,
            bm25_query=bm25_query,
        )

    def search_tags_hybrid_batch(
        self,
        dense_queries: list[np.ndarray],
        sparse_queries: list[dict[str, float]] | None = None,
        filter_dict_type: str | None = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags_list: list[list[str] | None] | None = None,
        limit: int = config.TAG_SEARCH_LIMIT,
        bm25_queries: list[dict[str, float]] | None = None,
    ) -> list[tuple[list[ScoredPoint], list[ScoredPoint], list[ScoredPoint]]]:
        """Executes batch hybrid tag retrieval."""
        return self._get_query_store().search_tags_hybrid_batch(
            dense_queries,
            sparse_queries=sparse_queries,
            filter_dict_type=filter_dict_type,
            domain=domain,
            allowed_tags_list=allowed_tags_list,
            limit=limit,
            bm25_queries=bm25_queries,
        )

    def search_collection_points(
        self,
        collection_name: str,
        dense_vec: Any,
        bm25_vec_dict: dict | None = None,
        sparse_vec_dict: dict | None = None,
        mode: str = "dense",
        normalized_query: str = "",
        filters_json: str = "",
        score_threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Executes admin search across any Qdrant collection."""
        return self._get_query_store().search_collection_points(
            collection_name=collection_name,
            dense_vec=dense_vec,
            bm25_vec_dict=bm25_vec_dict,
            sparse_vec_dict=sparse_vec_dict,
            mode=mode,
            normalized_query=normalized_query,
            filters_json=filters_json,
            score_threshold=score_threshold,
            limit=limit,
        )
