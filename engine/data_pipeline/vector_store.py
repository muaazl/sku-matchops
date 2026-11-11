import hashlib
import logging
import os
import socket
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
    OptimizersConfigDiff,
    PayloadSchemaType,
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

    @classmethod
    def get_client(
        cls, force_reconnect: bool = False, prefer_grpc: bool | None = None
    ) -> QdrantClient:
        """Returns or reinitializes the singleton QdrantClient instance."""
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

    def _ensure_collection(self, collection_name: str):
        """Creates the collection if it doesn't already exist and ensures indexes + HNSW settings."""
        if collection_name in self._checked_collections:
            return
        hnsw_on_disk = getattr(config, "QDRANT_HNSW_ON_DISK", False)
        exists = self._call_with_retry(
            "collection_exists", self.client.collection_exists, collection_name
        )
        if not exists:
            self._call_with_retry(
                "create_collection",
                self.client.create_collection,
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=config.BGE_M3_DENSE_DIM,
                        distance=Distance.COSINE,
                        on_disk=True,  # Raw float32 vectors stay memmapped on disk
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=True  # Sparse inverted index stays on disk
                        )
                    )
                },
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
            if not hnsw_on_disk and hasattr(self.client, "update_collection"):
                try:
                    info = self._call_with_retry(
                        "get_collection",
                        self.client.get_collection,
                        collection_name=collection_name,
                    )
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

        # Remove internal bookkeeping hash from search payload
        cleaned.pop("row_hash", None)

        # Food domain weight cleanup: if weight_val has no extracted values, drop it
        w_val = cleaned.get("weight_val")
        if domain == config.DOMAIN_FOOD or w_val in (
            [None, None, None],
            "[null, null, null]",
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

    def sync(
        self,
        df: pd.DataFrame,
        dense_embeddings: np.ndarray,
        sparse_weights: list[dict],
        domain: str = config.DOMAIN_MARKET,
    ):
        """Upserts a batch of rows and their embeddings to Qdrant."""
        if df.empty:
            return

        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)

        points = []
        records = df.to_dict(orient="records")
        indices = df.index.tolist()
        for list_idx, (orig_idx, payload) in enumerate(zip(indices, records)):
            # Deterministic UUID for idempotency (prefer stable db_uid if available)
            db_uid = payload.get("db_uid")
            unique_str = db_uid if db_uid else f"{orig_idx}_{payload.get('Name', '')}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

            # Sanitize payload: strip junk columns, duplicate fields, empty weights
            sanitized_payload = self._sanitize_catalog_payload(payload, domain=domain)

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embeddings[list_idx].tolist(),
                        "sparse": self._parse_sparse_dict(sparse_weights[list_idx]),
                    },
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

    def delete_collection(self, domain: str = config.DOMAIN_MARKET):
        """Deletes a domain's collection."""
        collection_name = self._get_collection_name(domain)
        self._checked_collections.discard(collection_name)
        if self._call_with_retry(
            "collection_exists", self.client.collection_exists, collection_name
        ):
            self._call_with_retry(
                "delete_collection", self.client.delete_collection, collection_name=collection_name
            )

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

    def search_from_vectors(
        self,
        dense_vec: np.ndarray,
        sparse_vec_dict: dict[str, float],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bt_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Performs hybrid RRF search using dense and sparse vectors, optionally filtered by BT."""
        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)
        sparse_vector = self._parse_sparse_dict(sparse_vec_dict)

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
            prefetch=[
                Prefetch(query=dense_vec.tolist(), using="dense", limit=top_k),
                Prefetch(query=sparse_vector, using="sparse", limit=top_k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        hits = []
        for hit in results.points:
            p = hit.payload.copy() if hit.payload else {}
            p["_qdrant_score_"] = hit.score
            hits.append(p)
        return hits

    def search_batch_from_vectors(
        self,
        dense_vecs: list[np.ndarray],
        sparse_vec_dicts: list[dict[str, float]],
        bt_filters: list[str | None],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> list[list[dict[str, Any]]]:
        """Performs batch hybrid RRF search using Qdrant's query_batch_points API."""
        if len(dense_vecs) == 0:
            return []

        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)

        requests = []
        for i in range(len(dense_vecs)):
            sparse_vector = self._parse_sparse_dict(sparse_vec_dicts[i])
            must_conditions = []
            if bt_filters[i]:
                must_conditions.append(
                    FieldCondition(key="basictype", match=MatchValue(value=bt_filters[i]))
                )
            query_filter = Filter(must=must_conditions) if must_conditions else None

            req = QueryRequest(
                prefetch=[
                    Prefetch(query=dense_vecs[i].tolist(), using="dense", limit=top_k),
                    Prefetch(query=sparse_vector, using="sparse", limit=top_k),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            requests.append(req)

        # Execute batch request with retry resilience
        # If the batch is very large, chunk it to avoid payload limits
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
                chunk_hits = []
                for hit in res.points:
                    p = hit.payload.copy() if hit.payload else {}
                    p["_qdrant_score_"] = hit.score
                    chunk_hits.append(p)
                results.append(chunk_hits)

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
                with_payload=True,
            )
            hits = []
            for hit in results.points:
                p = hit.payload.copy() if hit.payload else {}
                p["_qdrant_score_"] = float(hit.score)
                hits.append(p)
            return hits
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
                    with_payload=True,
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
                    chunk_hits = []
                    for hit in res.points:
                        p = hit.payload.copy() if hit.payload else {}
                        p["_qdrant_score_"] = float(hit.score)
                        chunk_hits.append(p)
                    results.append(chunk_hits)
        except Exception as e:
            logger.warning(f"[QDRANT] search_batch_catalog_neighbors failed: {e}")
            results.extend([[] for _ in range(n - len(results))])

        return results

    # --- Classifier Tag Methods ---

    def upsert_tags(
        self,
        tags: list[str],
        dense_embs: np.ndarray,
        sparse_embs: list[dict],
        dict_type: str,
        domain: str = config.DOMAIN_MARKET,
        metadata_list: list[dict] | None = None,
        force: bool = False,
    ):
        """Upserts dictionary tags for classification search."""
        if not tags:
            return

        collection_name = f"{domain}_tags"
        self._ensure_collection(collection_name)

        if not force:
            try:
                count_res = self._call_with_retry(
                    "count",
                    self.client.count,
                    collection_name=collection_name,
                    count_filter=Filter(
                        must=[FieldCondition(key="dict_type", match=MatchValue(value=dict_type))]
                    ),
                )
                # Point IDs are hashed from the tag, so duplicate tags collapse to one point
                if count_res.count >= len(set(tags)):
                    logger.info(
                        f"[QDRANT] Collection '{collection_name}' already contains {count_res.count} items for ({dict_type}). Skipping redundant upsert."
                    )
                    return
            except Exception as e:
                logger.warning(
                    f"[QDRANT] Could not check existing count for {collection_name} ({dict_type}): {e}"
                )

        points = []
        for i, (tag, dense, sparse) in enumerate(zip(tags, dense_embs, sparse_embs)):
            point_id = hashlib.md5(f"{domain}_{dict_type}_{tag}".encode()).hexdigest()

            payload = {"tag": tag, "dict_type": dict_type, "domain": domain}
            if metadata_list and i < len(metadata_list):
                payload.update(metadata_list[i])

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense.tolist() if hasattr(dense, "tolist") else dense,
                        "sparse": self._parse_sparse_dict(sparse),
                    },
                    payload=payload,
                )
            )

        # Batch upsert tags
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
        sparse_query: dict[str, float],
        limit: int = 50,
        filter_dict_type: str | None = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags: list[str] | None = None,
    ) -> tuple[list[ScoredPoint], list[ScoredPoint]]:
        """Returns separate dense and sparse search results for classification fusion.

        If allowed_tags is provided, Qdrant will only return results whose 'tag' payload
        is in that list (used to restrict food GK search to the bt_gk_map).
        """
        collection_name = f"{domain}_tags"
        if hasattr(self.client, "collection_exists"):
            self._ensure_collection(collection_name)

        must_conditions = []
        if filter_dict_type:
            must_conditions.append(
                FieldCondition(key="dict_type", match=MatchValue(value=filter_dict_type))
            )
        if allowed_tags:
            must_conditions.append(FieldCondition(key="tag", match=MatchAny(any=allowed_tags)))

        query_filter = Filter(must=must_conditions) if must_conditions else None
        d_query = dense_query.tolist() if hasattr(dense_query, "tolist") else dense_query
        sparse_vector = self._parse_sparse_dict(sparse_query)

        if sparse_vector.indices and hasattr(self.client, "query_batch_points"):
            batch_res = self._call_with_retry(
                "query_batch_points (hybrid tags)",
                self.client.query_batch_points,
                collection_name=collection_name,
                requests=[
                    QueryRequest(
                        query=d_query,
                        using="dense",
                        filter=query_filter,
                        limit=limit,
                        with_payload=["tag"],
                    ),
                    QueryRequest(
                        query=sparse_vector,
                        using="sparse",
                        filter=query_filter,
                        limit=limit,
                        with_payload=["tag"],
                    ),
                ],
            )
            return batch_res[0].points, batch_res[1].points

        dense_results = self._call_with_retry(
            "query_points (dense tags)",
            self.client.query_points,
            collection_name=collection_name,
            query=d_query,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=["tag"],
        ).points

        sparse_results = []
        if sparse_vector.indices:
            sparse_results = self._call_with_retry(
                "query_points (sparse tags)",
                self.client.query_points,
                collection_name=collection_name,
                query=sparse_vector,
                using="sparse",
                query_filter=query_filter,
                limit=limit,
                with_payload=["tag"],
            ).points

        return dense_results, sparse_results

    def search_batch_hybrid_tags(
        self,
        dense_queries: list[np.ndarray],
        sparse_queries: list[dict[str, float]],
        filter_dict_type: str | None = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags_list: list[list[str] | None] | None = None,
        limit: int = config.TAG_SEARCH_LIMIT,
    ) -> list[tuple[list[ScoredPoint], list[ScoredPoint]]]:
        """Performs batch dense and sparse tag searches using query_batch_points for high throughput."""
        n = len(dense_queries)
        if n == 0:
            return []

        collection_name = f"{domain}_tags"
        if hasattr(self.client, "collection_exists"):
            self._ensure_collection(collection_name)
        if allowed_tags_list is None:
            allowed_tags_list = [None] * n

        requests = []
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

            # Dense request
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

            # Sparse request
            sparse_vector = self._parse_sparse_dict(sparse_queries[i])
            if sparse_vector.indices:
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

        # Batch query points in chunks of QDRANT_BATCH_CHUNK_SIZE requests
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

        # Unpack back into [(dense_hits, sparse_hits), ...]
        results = []
        resp_idx = 0
        for i in range(n):
            dense_hits = raw_responses[resp_idx].points
            resp_idx += 1
            sparse_hits = []
            if has_sparse_flags[i]:
                sparse_hits = raw_responses[resp_idx].points
                resp_idx += 1
            results.append((dense_hits, sparse_hits))

        return results
