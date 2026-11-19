import hashlib
import logging
import math
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import grpc
import httpx
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    Modifier,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PayloadSelectorExclude,
    PointStruct,
    Prefetch,
    QueryRequest,
    ScoredPoint,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from engine import config

logger = logging.getLogger("matchops.vector_store")

TRANSIENT_NETWORK_ERRORS = (
    ResponseHandlingException,
    httpx.ConnectError,
    httpx.NetworkError,
    httpx.TimeoutException,
    socket.gaierror,
    ConnectionError,
    TimeoutError,
    OSError,
)

_NON_RETRYABLE_GRPC_CODES = frozenset(
    {
        grpc.StatusCode.NOT_FOUND,
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.ALREADY_EXISTS,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.UNIMPLEMENTED,
        grpc.StatusCode.FAILED_PRECONDITION,
    }
)


class VectorStore:
    """Manages persistent vector storage and hybrid search using Qdrant."""

    _client: QdrantClient | None = None
    _prefer_grpc_override: bool | None = None
    _checked_collections: set = set()
    _lock = threading.RLock()
    _SEARCH_PAYLOAD = PayloadSelectorExclude(exclude=["row_hash"])

    @classmethod
    def get_client(
        cls, force_reconnect: bool = False, prefer_grpc: bool | None = None
    ) -> QdrantClient:
        """Returns or reinitializes the singleton QdrantClient instance."""
        with cls._lock:
            if prefer_grpc is not None:
                cls._prefer_grpc_override = prefer_grpc
            if cls._client is None or force_reconnect:
                old_client = cls._client
                use_grpc = (
                    cls._prefer_grpc_override
                    if cls._prefer_grpc_override is not None
                    else getattr(config, "QDRANT_PREFER_GRPC", True)
                )
                cls._client = QdrantClient(
                    url=config.QDRANT_URL,
                    grpc_port=getattr(config, "QDRANT_GRPC_PORT", 6334),
                    prefer_grpc=use_grpc,
                    timeout=config.QDRANT_TIMEOUT,
                    check_compatibility=False,
                )
                if old_client is not None:
                    try:
                        old_client.close()
                    except Exception:
                        pass
            return cls._client

    def __init__(self):
        os.makedirs(config.QDRANT_DATA_DIR, exist_ok=True)
        self.client = VectorStore.get_client()

    def _call_with_retry(
        self,
        operation_name: str,
        func: Callable[..., Any],
        *args: Any,
        max_retries: int = 5,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        **kwargs: Any,
    ) -> Any:
        """Executes a Qdrant client call with automatic retries, client rebinding, and gRPC->REST fallback."""
        delay = initial_delay
        last_error = None
        bound_method_name = (
            getattr(func, "__name__", None)
            if getattr(func, "__self__", None) is self.client
            else next(
                (
                    attr
                    for attr in (
                        "get_collections",
                        "get_collection",
                        "collection_exists",
                        "create_collection",
                        "update_collection",
                        "create_payload_index",
                        "upsert",
                        "query_points",
                        "query_batch_points",
                        "scroll",
                        "count",
                    )
                    if getattr(self.client, attr, None) is func
                ),
                None,
            )
        )

        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if getattr(e, "status_code", None) in (400, 401, 403, 404, 409, 422):
                    raise
                if isinstance(e, grpc.RpcError) and e.code() in _NON_RETRYABLE_GRPC_CODES:
                    raise

                err_msg = str(e).lower()
                is_grpc_unavailable = (
                    isinstance(e, grpc.RpcError) and e.code() == grpc.StatusCode.UNAVAILABLE
                )
                is_transient = (
                    is_grpc_unavailable
                    or (
                        isinstance(e, grpc.RpcError)
                        and e.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                    )
                    or isinstance(e, TRANSIENT_NETWORK_ERRORS)
                    or "name resolution" in err_msg
                    or "temporary failure" in err_msg
                    or "connection" in err_msg
                    or "timeout" in err_msg
                    or "errno -3" in err_msg
                    or "eai_again" in err_msg
                )

                if not is_transient or attempt == max_retries:
                    logger.error(
                        f"[QDRANT] Operation '{operation_name}' failed permanently on attempt {attempt}/{max_retries}: {e}"
                    )
                    raise

                # If gRPC port (6334) is unreachable in a local/legacy container, fall back seamlessly to REST
                if (
                    is_grpc_unavailable
                    and self.client is VectorStore._client
                    and VectorStore._prefer_grpc_override is not False
                    and getattr(config, "QDRANT_PREFER_GRPC", True)
                ):
                    logger.warning(
                        f"[QDRANT] gRPC port unavailable during '{operation_name}'; falling back to HTTP REST client."
                    )
                    self.client = VectorStore.get_client(force_reconnect=True, prefer_grpc=False)
                    if bound_method_name:
                        func = getattr(self.client, bound_method_name)
                    continue

                logger.warning(
                    f"[QDRANT] Transient error during '{operation_name}' (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor

                # Refresh singleton client instance on repeated connection glitches and rebind method
                if attempt >= 2 and self.client is VectorStore._client:
                    self.client = VectorStore.get_client(force_reconnect=True)
                    if bound_method_name:
                        func = getattr(self.client, bound_method_name)

        raise last_error

    def _get_collection_name(self, domain: str) -> str:
        """Returns the appropriate collection name for a domain."""
        return (
            config.QDRANT_COLLECTION_FOOD
            if domain == config.DOMAIN_FOOD
            else config.QDRANT_COLLECTION_MARKET
        )

    @staticmethod
    def _inspect_schema_mismatch(info) -> str | None:
        """Returns a schema mismatch description if collection params conflict with active config."""
        params = getattr(getattr(info, "config", None), "params", None)
        if params is None:
            return None
        vectors = getattr(params, "vectors", None)
        vectors = getattr(vectors, "root", vectors)
        dense_cfg = (
            vectors.get("dense")
            if isinstance(vectors, dict)
            else getattr(vectors, "dense", vectors)
        )
        actual_dim = getattr(dense_cfg, "size", None)
        if isinstance(actual_dim, int) and not isinstance(actual_dim, bool):
            if actual_dim != config.EMBEDDING_DIM:
                return f"dense dim={actual_dim}, expected {config.EMBEDDING_DIM}"
            sparse_vectors = getattr(params, "sparse_vectors", None)
            actual_sparse = (
                set(sparse_vectors.keys()) if isinstance(sparse_vectors, dict) else set()
            )
            if "bm25" not in actual_sparse:
                return f"missing 'bm25' sparse vector (found {sorted(actual_sparse)})"
            has_sparse = "sparse" in actual_sparse
            if has_sparse != config.has_learned_sparse():
                return (
                    f"learned sparse mismatch (collection has_sparse={has_sparse}, "
                    f"expected={config.has_learned_sparse()})"
                )
        return None

    def get_collection_schema_mismatch(self, collection_name: str) -> str | None:
        """Inspects an existing Qdrant collection and returns any mismatch against active config."""
        try:
            if not self._call_with_retry(
                "collection_exists", self.client.collection_exists, collection_name
            ):
                return None
            info = self._call_with_retry(
                "get_collection", self.client.get_collection, collection_name=collection_name
            )
        except Exception as e:
            logger.debug(f"[QDRANT] Could not inspect config on '{collection_name}': {e}")
            return None
        return self._inspect_schema_mismatch(info)

    def _ensure_collection(self, collection_name: str):
        """Creates the collection if it doesn't already exist and ensures indexes + HNSW settings."""
        with self._lock:
            if collection_name in self._checked_collections:
                return
            hnsw_on_disk = getattr(config, "QDRANT_HNSW_ON_DISK", False)
            exists = self._call_with_retry(
                "collection_exists", self.client.collection_exists, collection_name
            )
            if not exists:
                sparse_cfg: dict[str, SparseVectorParams] = {
                    "bm25": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=True  # BM25 inverted index stays on disk
                        ),
                        modifier=Modifier.IDF,
                    )
                }
                if config.has_learned_sparse():
                    sparse_cfg["sparse"] = SparseVectorParams(index=SparseIndexParams(on_disk=True))

                self._call_with_retry(
                    "create_collection",
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=config.EMBEDDING_DIM,
                            distance=Distance.COSINE,
                            on_disk=True,  # Raw float32 vectors stay memmapped on disk
                        )
                    },
                    sparse_vectors_config=sparse_cfg,
                    hnsw_config=HnswConfigDiff(
                        on_disk=hnsw_on_disk,
                        m=getattr(config, "QDRANT_HNSW_M", 16),
                        ef_construct=getattr(config, "QDRANT_HNSW_EF_CONSTRUCT", 100),
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=getattr(config, "QDRANT_INDEXING_THRESHOLD", 10000),
                        memmap_threshold=getattr(config, "QDRANT_MEMMAP_THRESHOLD", 5000),
                    ),
                )
                self._create_payload_indexes(collection_name)
            else:
                self._create_payload_indexes(collection_name)
                info = None
                try:
                    info = self._call_with_retry(
                        "get_collection",
                        self.client.get_collection,
                        collection_name=collection_name,
                    )
                except Exception as e:
                    logger.debug(f"[QDRANT] Could not inspect config on '{collection_name}': {e}")
                if info is not None:
                    mismatch = self._inspect_schema_mismatch(info)
                    if mismatch:
                        raise RuntimeError(
                            f"Qdrant collection '{collection_name}' schema mismatch ({mismatch}). "
                            f"Run `python -m scripts.sync --qdrant rebuild` to recreate collections for the active model."
                        )
                    if not hnsw_on_disk and hasattr(self.client, "update_collection"):
                        try:
                            cur_hnsw = getattr(getattr(info, "config", None), "hnsw_config", None)
                            if getattr(cur_hnsw, "on_disk", None) is True:
                                self._call_with_retry(
                                    "update_collection",
                                    self.client.update_collection,
                                    collection_name=collection_name,
                                    hnsw_config=HnswConfigDiff(on_disk=False),
                                )
                        except Exception as e:
                            logger.debug(
                                f"[QDRANT] Could not inspect/update HNSW config on '{collection_name}': {e}"
                            )
            self._checked_collections.add(collection_name)

    def _create_payload_indexes(self, collection_name: str):
        """Creates payload indexes for fields used in filtering."""
        index_fields = ("dict_type", "tag") if collection_name.endswith("_tags") else ("basictype",)
        for field_name in index_fields:
            try:
                self._call_with_retry(
                    "create_payload_index",
                    self.client.create_payload_index,
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug(f"[QDRANT] Could not create payload index on '{field_name}': {e}")

    def create_versioned_collection(self, alias_name: str, version: int = 1) -> str:
        """Creates a versioned physical collection (e.g. {alias_name}_v{version}) with complete on-disk optimizations."""
        physical_name = f"{alias_name}_v{version}"
        self._ensure_collection(physical_name)
        return physical_name

    def swap_collection_alias(self, alias_name: str, new_physical_collection: str) -> None:
        """Atomically switches an alias to point to the new physical collection for zero-downtime schema evolution."""
        from qdrant_client.http.models import (
            CreateAlias,
            CreateAliasOperation,
            DeleteAlias,
            DeleteAliasOperation,
        )

        operations = []
        try:
            aliases_res = self._call_with_retry("get_aliases", self.client.get_aliases)
            for item in getattr(aliases_res, "aliases", []):
                if getattr(item, "alias_name", None) == alias_name:
                    operations.append(
                        DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name))
                    )
        except Exception as e:
            logger.debug(f"[QDRANT] Could not inspect existing aliases for {alias_name}: {e}")

        operations.append(
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=new_physical_collection,
                    alias_name=alias_name,
                )
            )
        )
        self._call_with_retry(
            "update_collection_aliases",
            self.client.update_collection_aliases,
            change_aliases_operations=operations,
        )
        logger.info(
            f"[QDRANT] Successfully pointed alias '{alias_name}' -> '{new_physical_collection}'."
        )

    def get_collection_schema_info(self, collection_name: str) -> dict[str, Any]:
        """Returns collection configuration and vector parameters for verification and schema introspection."""
        try:
            info = self._call_with_retry(
                "get_collection", self.client.get_collection, collection_name=collection_name
            )
            return {
                "status": getattr(info, "status", None),
                "vectors_count": getattr(info, "vectors_count", None),
                "points_count": getattr(info, "points_count", None),
                "config": getattr(info, "config", None),
            }
        except Exception as e:
            logger.warning(f"[QDRANT] Could not get schema info for '{collection_name}': {e}")
            return {}

    def _parse_sparse_dict(self, sparse_dict: dict[str, float]) -> SparseVector:
        """Converts a dictionary of lexical weights to a Qdrant SparseVector."""
        indices, values = [], []
        if sparse_dict and isinstance(sparse_dict, dict):
            for k, v in sparse_dict.items():
                try:
                    indices.append(int(k))
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
        return SparseVector(indices=indices, values=values)

    @staticmethod
    def _sanitize_catalog_payload(raw_payload: dict, domain: str) -> dict:
        """Strips junk columns, standardizes price, and removes redundant structures from Qdrant payloads."""
        cleaned = {}
        junk_prefixes = ("_duplicated_", "Unnamed")
        for k, v in raw_payload.items():
            k_str = str(k).strip()
            if not k_str or any(k_str.startswith(p) for p in junk_prefixes):
                continue
            if v is pd.NA or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                v = None
            cleaned[k_str] = v

        # Clean price: retain numerical 'price' (float), remove redundant raw 'Price' string
        if "Price" in cleaned and "price" in cleaned:
            try:
                cleaned["price"] = float(cleaned.get("price") or 0.0)
            except Exception:
                pass
            del cleaned["Price"]
        elif "Price" in cleaned:
            from engine.data_pipeline.cache_manager import clean_price

            cleaned["price"] = clean_price(cleaned["Price"])
            del cleaned["Price"]

        # Keep internal row_hash on stored points so get_existing_hashes() can diff Qdrant state;
        # search queries exclude it via _SEARCH_PAYLOAD.

        # Food domain weight cleanup: if weight_val has no extracted values, drop it
        w_val = cleaned.get("weight_val")
        if domain == config.DOMAIN_FOOD or w_val in (
            [None, None, None],
            (None, None, None),
            "[null, null, null]",
            "(None, None, None)",
            None,
        ):
            cleaned.pop("weight_val", None)
            if domain == config.DOMAIN_FOOD and cleaned.get("clean_no_weights") == cleaned.get(
                "clean_text"
            ):
                cleaned.pop("clean_no_weights", None)

        # Clean entities: remove empty keys, convert sets to lists
        if "entities" in cleaned and isinstance(cleaned["entities"], dict):
            ents = {
                k: list(v) if isinstance(v, (set, list)) else v
                for k, v in cleaned["entities"].items()
                if v
            }
            if ents:
                cleaned["entities"] = ents
            else:
                cleaned.pop("entities", None)

        return cleaned

    def _build_point_vectors(
        self,
        dense_vec,
        bm25_dict: dict | None,
        sparse_dict: dict | None,
        include_learned_sparse: bool,
    ) -> dict[str, Any]:
        vecs: dict[str, Any] = {
            "dense": dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec),
            "bm25": self._parse_sparse_dict(bm25_dict or {}),
        }
        if include_learned_sparse:
            vecs["sparse"] = self._parse_sparse_dict(sparse_dict or {})
        return vecs

    def _build_hybrid_prefetch(
        self,
        dense_vec: np.ndarray,
        bm25_dict: dict | None,
        sparse_dict: dict | None,
        limit: int,
    ) -> list[Prefetch]:
        d_list = dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)
        prefetches = [Prefetch(query=d_list, using="dense", limit=limit)]
        bm25_sv = self._parse_sparse_dict(bm25_dict or {})
        if bm25_sv.indices:
            prefetches.append(Prefetch(query=bm25_sv, using="bm25", limit=limit))
        if config.has_learned_sparse() and sparse_dict:
            sparse_sv = self._parse_sparse_dict(sparse_dict)
            if sparse_sv.indices:
                prefetches.append(Prefetch(query=sparse_sv, using="sparse", limit=limit))
        return prefetches

    def sync(
        self,
        df: pd.DataFrame,
        dense_embeddings: np.ndarray,
        sparse_weights: list[dict] | None = None,
        bm25_weights: list[dict] | None = None,
        domain: str = config.DOMAIN_MARKET,
    ):
        """Upserts a batch of rows and their embeddings to Qdrant."""
        if df.empty:
            return

        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)
        include_learned_sparse = config.has_learned_sparse() and bool(sparse_weights)

        points = []
        records = df.to_dict(orient="records")
        indices = df.index.tolist()
        for list_idx, (orig_idx, payload) in enumerate(zip(indices, records)):
            db_uid = payload.get("db_uid")
            unique_str = db_uid if db_uid else f"{orig_idx}_{payload.get('Name', '')}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

            sanitized_payload = self._sanitize_catalog_payload(payload, domain=domain)
            bm25_d = bm25_weights[list_idx] if bm25_weights and list_idx < len(bm25_weights) else {}
            sparse_d = (
                sparse_weights[list_idx]
                if include_learned_sparse and sparse_weights and list_idx < len(sparse_weights)
                else None
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=self._build_point_vectors(
                        dense_embeddings[list_idx],
                        bm25_d,
                        sparse_d,
                        include_learned_sparse=include_learned_sparse,
                    ),
                    payload=sanitized_payload,
                )
            )

            if len(points) >= config.UPSERT_BATCH_SIZE:
                self._call_with_retry(
                    "upsert", self.client.upsert, collection_name=collection_name, points=points
                )
                points = []

        if points:
            self._call_with_retry(
                "upsert", self.client.upsert, collection_name=collection_name, points=points
            )

    def delete_named_collection(self, collection_name: str):
        """Deletes a Qdrant collection by exact name."""
        with self._lock:
            self._checked_collections.discard(collection_name)
        if self._call_with_retry(
            "collection_exists", self.client.collection_exists, collection_name
        ):
            self._call_with_retry(
                "delete_collection", self.client.delete_collection, collection_name=collection_name
            )

    def delete_collection(self, domain: str = config.DOMAIN_MARKET):
        """Deletes a domain's collection."""
        self.delete_named_collection(self._get_collection_name(domain))

    def get_existing_hashes(self, domain: str = config.DOMAIN_MARKET) -> dict[str, str]:
        """Bulk-fetches {db_uid: row_hash} for every point currently stored in the domain's
        catalog collection, straight from Qdrant's payloads.

        Used by the catalog sync script to make the *shared* Qdrant instance's actual
        contents the source of truth for "has this row already been embedded", instead of
        a local per-checkout hash file that has no way of knowing what a different checkout
        already pushed to the same Qdrant instance.
        """
        collection_name = self._get_collection_name(domain)
        exists = self._call_with_retry(
            "collection_exists", self.client.collection_exists, collection_name
        )
        if not exists:
            return {}

        result: dict[str, str] = {}
        next_offset = None
        while True:
            points, next_offset = self._call_with_retry(
                "scroll",
                self.client.scroll,
                collection_name=collection_name,
                limit=1000,
                offset=next_offset,
                with_payload=["db_uid", "row_hash"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                uid = payload.get("db_uid")
                row_hash = payload.get("row_hash")
                if uid and row_hash:
                    result[uid] = row_hash
            if next_offset is None:
                break
        return result

    @staticmethod
    def _hit_to_dict(hit) -> dict[str, Any]:
        p = hit.payload.copy() if hit.payload else {}
        p.pop("row_hash", None)
        p["_qdrant_score_"] = float(hit.score)
        return p

    def search_from_vectors(
        self,
        dense_vec: np.ndarray,
        sparse_vec_dict: dict[str, float] | None = None,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bt_filter: str | None = None,
        bm25_vec_dict: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Performs hybrid RRF search (Dense + BM25 + optional BGE-M3 learned sparse), optionally filtered by BT."""
        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)
        prefetch = self._build_hybrid_prefetch(dense_vec, bm25_vec_dict, sparse_vec_dict, top_k)

        must_conditions = []
        if bt_filter:
            must_conditions.append(
                FieldCondition(key="basictype", match=MatchValue(value=bt_filter))
            )
        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self._call_with_retry(
            "query_points",
            self.client.query_points,
            collection_name=collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=self._SEARCH_PAYLOAD,
        )
        return [self._hit_to_dict(hit) for hit in results.points]

    def search_batch_from_vectors(
        self,
        dense_vecs: list[np.ndarray],
        sparse_vec_dicts: list[dict[str, float]] | None = None,
        bt_filters: list[str | None] | None = None,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bm25_vec_dicts: list[dict[str, float]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Performs batch hybrid RRF search (Dense + BM25 + optional BGE-M3 learned sparse)."""
        if len(dense_vecs) == 0:
            return []

        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)
        n = len(dense_vecs)
        bt_filters = bt_filters or [None] * n

        requests = []
        for i in range(n):
            bm25_d = bm25_vec_dicts[i] if bm25_vec_dicts and i < len(bm25_vec_dicts) else None
            sparse_d = (
                sparse_vec_dicts[i] if sparse_vec_dicts and i < len(sparse_vec_dicts) else None
            )
            must_conditions = []
            if bt_filters[i]:
                must_conditions.append(
                    FieldCondition(key="basictype", match=MatchValue(value=bt_filters[i]))
                )
            query_filter = Filter(must=must_conditions) if must_conditions else None

            req = QueryRequest(
                prefetch=self._build_hybrid_prefetch(dense_vecs[i], bm25_d, sparse_d, top_k),
                query=FusionQuery(fusion=Fusion.RRF),
                filter=query_filter,
                limit=top_k,
                with_payload=self._SEARCH_PAYLOAD,
            )
            requests.append(req)

        results = []
        batch_size = config.QDRANT_BATCH_CHUNK_SIZE
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            batch_results = self._call_with_retry(
                "query_batch_points",
                self.client.query_batch_points,
                collection_name=collection_name,
                requests=chunk,
            )
            for res in batch_results:
                results.append([self._hit_to_dict(hit) for hit in res.points])

        return results

    def search_catalog_neighbors(
        self,
        dense_vec: np.ndarray,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Retrieves top_k nearest catalog SKUs by dense vector cosine similarity for Tier 2 Few-Shot matching.
        """
        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)
        d_vec = dense_vec.tolist() if hasattr(dense_vec, "tolist") else dense_vec
        try:
            results = self._call_with_retry(
                "query_points (catalog neighbors)",
                self.client.query_points,
                collection_name=collection_name,
                query=d_vec,
                using="dense",
                limit=top_k,
                with_payload=self._SEARCH_PAYLOAD,
            )
            return [self._hit_to_dict(hit) for hit in results.points]
        except Exception as e:
            logger.warning(f"[QDRANT] search_catalog_neighbors failed: {e}")
            return []

    def search_batch_catalog_neighbors(
        self,
        dense_vecs: list[np.ndarray],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = 15,
    ) -> list[list[dict[str, Any]]]:
        """
        Batch-retrieves top_k nearest catalog SKUs by dense vector cosine similarity for Tier 2 Few-Shot matching.
        """
        n = len(dense_vecs)
        if n == 0:
            return []

        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)

        requests = []
        for i in range(n):
            d_vec = dense_vecs[i].tolist() if hasattr(dense_vecs[i], "tolist") else dense_vecs[i]
            requests.append(
                QueryRequest(
                    query=d_vec,
                    using="dense",
                    limit=top_k,
                    with_payload=self._SEARCH_PAYLOAD,
                )
            )

        results = []
        batch_size = config.QDRANT_BATCH_CHUNK_SIZE
        try:
            for i in range(0, len(requests), batch_size):
                chunk = requests[i : i + batch_size]
                batch_res = self._call_with_retry(
                    "query_batch_points (catalog neighbors)",
                    self.client.query_batch_points,
                    collection_name=collection_name,
                    requests=chunk,
                )
                for res in batch_res:
                    results.append([self._hit_to_dict(hit) for hit in res.points])
        except Exception as e:
            logger.warning(f"[QDRANT] search_batch_catalog_neighbors failed: {e}")
            results.extend([[] for _ in range(n - len(results))])

        return results

    # --- Classifier Tag Methods ---

    def has_tags_synced(self, domain: str, dict_type: str, expected_count: int) -> bool:
        """Returns True when Qdrant already holds at least expected_count tags for dict_type under EMBEDDING_MODEL."""
        if expected_count <= 0:
            return True
        collection_name = f"{domain}_tags"
        try:
            self._ensure_collection(collection_name)
            type_filter = Filter(
                must=[FieldCondition(key="dict_type", match=MatchValue(value=dict_type))]
            )
            count_res = self._call_with_retry(
                "count",
                self.client.count,
                collection_name=collection_name,
                count_filter=type_filter,
            )
            if count_res.count >= expected_count:
                scroll_res = self._call_with_retry(
                    "scroll",
                    self.client.scroll,
                    collection_name=collection_name,
                    scroll_filter=type_filter,
                    limit=1,
                    with_payload=["embedding_model"],
                    with_vectors=False,
                )
                sample_pts = (
                    scroll_res[0] if isinstance(scroll_res, (tuple, list)) and scroll_res else []
                )
                existing_model = (
                    (sample_pts[0].payload or {}).get("embedding_model") if sample_pts else None
                )
                if existing_model == config.EMBEDDING_MODEL:
                    logger.info(
                        f"[QDRANT] Collection '{collection_name}' already contains {count_res.count} items for ({dict_type}). Skipping redundant upsert."
                    )
                    return True
                logger.info(
                    f"[QDRANT] Re-upserting '{collection_name}' ({dict_type}): "
                    f"model changed ({existing_model!r} -> {config.EMBEDDING_MODEL!r})."
                )
        except Exception as e:
            logger.warning(
                f"[QDRANT] Could not check existing count for {collection_name} ({dict_type}): {e}"
            )
        return False

    def upsert_tags(
        self,
        tags: list[str],
        dense_embs: np.ndarray,
        sparse_embs: list[dict] | None = None,
        dict_type: str = "",
        domain: str = config.DOMAIN_MARKET,
        metadata_list: list[dict] | None = None,
        force: bool = False,
        bm25_embs: list[dict] | None = None,
    ):
        """Upserts dictionary tags (Dense + BM25 + optional BGE-M3 learned sparse) for classification."""
        if not tags:
            return

        collection_name = f"{domain}_tags"
        self._ensure_collection(collection_name)

        if not force and self.has_tags_synced(domain, dict_type, len(set(tags))):
            return

        include_learned_sparse = config.has_learned_sparse() and bool(sparse_embs)
        points = []
        for i, (tag, dense) in enumerate(zip(tags, dense_embs)):
            point_id = hashlib.md5(f"{domain}_{dict_type}_{tag}".encode()).hexdigest()

            payload = {
                "tag": tag,
                "dict_type": dict_type,
                "domain": domain,
                "embedding_model": config.EMBEDDING_MODEL,
            }
            if metadata_list and i < len(metadata_list):
                payload.update(metadata_list[i])

            bm25_d = bm25_embs[i] if bm25_embs and i < len(bm25_embs) else {}
            sparse_d = (
                sparse_embs[i]
                if include_learned_sparse and sparse_embs and i < len(sparse_embs)
                else None
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=self._build_point_vectors(
                        dense,
                        bm25_d,
                        sparse_d,
                        include_learned_sparse=include_learned_sparse,
                    ),
                    payload=payload,
                )
            )

        batch_size = 500
        for i in range(0, len(points), batch_size):
            self._call_with_retry(
                "upsert",
                self.client.upsert,
                collection_name=collection_name,
                points=points[i : i + batch_size],
                wait=False,
            )
        logger.info(f"[QDRANT] Upserted {len(points)} items to {collection_name} ({dict_type}).")

    def search_hybrid_tags(
        self,
        dense_query: np.ndarray,
        sparse_query: dict[str, float] | None = None,
        limit: int = 50,
        filter_dict_type: str | None = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags: list[str] | None = None,
        bm25_query: dict[str, float] | None = None,
    ) -> tuple[list[ScoredPoint], list[ScoredPoint], list[ScoredPoint]]:
        """Returns (dense_hits, bm25_hits, sparse_hits) for tag classification fusion."""
        return self.search_batch_hybrid_tags(
            dense_queries=[dense_query],
            sparse_queries=[sparse_query] if sparse_query is not None else None,
            filter_dict_type=filter_dict_type,
            domain=domain,
            allowed_tags_list=[allowed_tags],
            limit=limit,
            bm25_queries=[bm25_query] if bm25_query is not None else None,
        )[0]

    def search_batch_hybrid_tags(
        self,
        dense_queries: list[np.ndarray],
        sparse_queries: list[dict[str, float]] | None = None,
        filter_dict_type: str | None = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags_list: list[list[str] | None] | None = None,
        limit: int = config.TAG_SEARCH_LIMIT,
        bm25_queries: list[dict[str, float]] | None = None,
    ) -> list[tuple[list[ScoredPoint], list[ScoredPoint], list[ScoredPoint]]]:
        """Performs batch dense, BM25, and (for BGE-M3) learned sparse tag searches via query_batch_points."""
        n = len(dense_queries)
        if n == 0:
            return []

        collection_name = f"{domain}_tags"
        if hasattr(self.client, "collection_exists"):
            self._ensure_collection(collection_name)
        if allowed_tags_list is None:
            allowed_tags_list = [None] * n

        use_learned_sparse = config.has_learned_sparse()
        requests = []
        has_bm25_flags = []
        has_sparse_flags = []

        for i in range(n):
            must_conditions = []
            if filter_dict_type:
                must_conditions.append(
                    FieldCondition(key="dict_type", match=MatchValue(value=filter_dict_type))
                )
            if allowed_tags_list[i]:
                must_conditions.append(
                    FieldCondition(key="tag", match=MatchAny(any=allowed_tags_list[i]))
                )
            query_filter = Filter(must=must_conditions) if must_conditions else None

            d_query = (
                dense_queries[i].tolist()
                if hasattr(dense_queries[i], "tolist")
                else dense_queries[i]
            )
            requests.append(
                QueryRequest(
                    query=d_query,
                    using="dense",
                    filter=query_filter,
                    limit=limit,
                    with_payload=["tag"],
                )
            )

            bm25_d = bm25_queries[i] if bm25_queries and i < len(bm25_queries) else None
            bm25_vector = self._parse_sparse_dict(bm25_d) if bm25_d else None
            if bm25_vector and bm25_vector.indices:
                requests.append(
                    QueryRequest(
                        query=bm25_vector,
                        using="bm25",
                        filter=query_filter,
                        limit=limit,
                        with_payload=["tag"],
                    )
                )
                has_bm25_flags.append(True)
            else:
                has_bm25_flags.append(False)

            sparse_d = sparse_queries[i] if sparse_queries and i < len(sparse_queries) else None
            sparse_vector = (
                self._parse_sparse_dict(sparse_d) if (use_learned_sparse and sparse_d) else None
            )
            if sparse_vector and sparse_vector.indices:
                requests.append(
                    QueryRequest(
                        query=sparse_vector,
                        using="sparse",
                        filter=query_filter,
                        limit=limit,
                        with_payload=["tag"],
                    )
                )
                has_sparse_flags.append(True)
            else:
                has_sparse_flags.append(False)

        batch_size = config.QDRANT_BATCH_CHUNK_SIZE
        raw_responses = []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            batch_res = self._call_with_retry(
                "query_batch_points (batch tags)",
                self.client.query_batch_points,
                collection_name=collection_name,
                requests=chunk,
            )
            raw_responses.extend(batch_res)

        results = []
        resp_idx = 0
        for i in range(n):
            dense_hits = raw_responses[resp_idx].points
            resp_idx += 1
            bm25_hits = []
            if has_bm25_flags[i]:
                bm25_hits = raw_responses[resp_idx].points
                resp_idx += 1
            sparse_hits = []
            if has_sparse_flags[i]:
                sparse_hits = raw_responses[resp_idx].points
                resp_idx += 1
            results.append((dense_hits, bm25_hits, sparse_hits))

        return results
