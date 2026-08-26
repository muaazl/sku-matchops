import hashlib
import logging
import os
import socket
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    QueryRequest,
    ScoredPoint,
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

class VectorStore:
    """Manages persistent vector storage and hybrid search using Qdrant."""

    _client: Optional[QdrantClient] = None
    _checked_collections: set = set()

    @classmethod
    def get_client(cls, force_reconnect: bool = False) -> QdrantClient:
        """Returns or reinitializes the singleton QdrantClient instance."""
        if cls._client is None or force_reconnect:
            cls._client = QdrantClient(url=config.QDRANT_URL, timeout=60.0)
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
        """Executes a Qdrant client call with automatic retries on transient network/DNS errors."""
        delay = initial_delay
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                is_transient = (
                    isinstance(e, TRANSIENT_NETWORK_ERRORS)
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

                logger.warning(
                    f"[QDRANT] Transient error during '{operation_name}' (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor

                # Refresh client instance on repeated connection glitches
                if attempt >= 2:
                    self.client = VectorStore.get_client(force_reconnect=True)

        raise last_error

    def _get_collection_name(self, domain: str) -> str:
        """Returns the appropriate collection name for a domain."""
        return config.QDRANT_COLLECTION_FOOD if domain == config.DOMAIN_FOOD else config.QDRANT_COLLECTION_MARKET

    def _ensure_collection(self, collection_name: str):
        """Creates the collection if it doesn't already exist with on-disk optimizations."""
        if collection_name in self._checked_collections:
            return
        exists = self._call_with_retry("collection_exists", self.client.collection_exists, collection_name)
        if not exists:
            from qdrant_client.http.models import HnswConfigDiff, OptimizersConfigDiff, SparseIndexParams
            self._call_with_retry(
                "create_collection",
                self.client.create_collection,
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=config.BGE_M3_DENSE_DIM,
                        distance=Distance.COSINE,
                        on_disk=True  # Vectors stay on disk
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
                    on_disk=True,  # HNSW graph stays on disk
                    m=getattr(config, "QDRANT_HNSW_M", 16),
                    ef_construct=getattr(config, "QDRANT_HNSW_EF_CONSTRUCT", 100),
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=getattr(config, "QDRANT_INDEXING_THRESHOLD", 10000),
                    memmap_threshold=getattr(config, "QDRANT_MEMMAP_THRESHOLD", 5000),
                ),
            )
            # Create payload indexes for fast filtering
            self._create_payload_indexes(collection_name)
        self._checked_collections.add(collection_name)

    def _create_payload_indexes(self, collection_name: str):
        """Creates payload indexes for fields used in filtering."""
        try:
            self._call_with_retry(
                "create_payload_index",
                self.client.create_payload_index,
                collection_name=collection_name,
                field_name="basictype",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.debug(f"[QDRANT] Could not create payload index on 'basictype': {e}")

    def _parse_sparse_dict(self, sparse_dict: Dict[str, float]) -> SparseVector:
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

    def sync(self, df: pd.DataFrame, dense_embeddings: np.ndarray, sparse_weights: List[dict], domain: str = config.DOMAIN_MARKET):
        """Upserts a batch of rows and their embeddings to Qdrant."""
        if df.empty:
            return

        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)

        points = []
        for list_idx, (orig_idx, row) in enumerate(df.iterrows()):
            # Deterministic UUID for idempotency (prefer stable db_uid if available)
            db_uid = row.get("db_uid")
            unique_str = db_uid if db_uid else f"{orig_idx}_{row.get('Name', '')}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

            payload = row.to_dict()
            # Clean up sets in payload for JSON serialization
            if "entities" in payload and isinstance(payload["entities"], dict):
                payload["entities"] = {k: list(v) if isinstance(v, set) else v for k, v in payload["entities"].items()}

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embeddings[list_idx].tolist(),
                        "sparse": self._parse_sparse_dict(sparse_weights[list_idx]),
                    },
                    payload=payload,
                )
            )

            if len(points) >= config.UPSERT_BATCH_SIZE:
                self._call_with_retry("upsert", self.client.upsert, collection_name=collection_name, points=points)
                points = []

        if points:
            self._call_with_retry("upsert", self.client.upsert, collection_name=collection_name, points=points)

    def delete_collection(self, domain: str = config.DOMAIN_MARKET):
        """Deletes a domain's collection."""
        collection_name = self._get_collection_name(domain)
        self._checked_collections.discard(collection_name)
        if self._call_with_retry("collection_exists", self.client.collection_exists, collection_name):
            self._call_with_retry("delete_collection", self.client.delete_collection, collection_name=collection_name)

    def search_from_vectors(
        self,
        dense_vec: np.ndarray,
        sparse_vec_dict: Dict[str, float],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bt_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Performs hybrid RRF search using dense and sparse vectors, optionally filtered by BT."""
        collection_name = self._get_collection_name(domain)
        self._ensure_collection(collection_name)
        sparse_vector = self._parse_sparse_dict(sparse_vec_dict)

        must_conditions = []
        if bt_filter:
            must_conditions.append(FieldCondition(key="basictype", match=MatchValue(value=bt_filter)))
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
        dense_vecs: List[np.ndarray],
        sparse_vec_dicts: List[Dict[str, float]],
        bt_filters: List[Optional[str]],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> List[List[Dict[str, Any]]]:
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
                must_conditions.append(FieldCondition(key="basictype", match=MatchValue(value=bt_filters[i])))
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
        batch_size = 100
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            batch_results = self._call_with_retry(
                "query_batch_points",
                self.client.query_batch_points,
                collection_name=collection_name,
                requests=chunk
            )
            for res in batch_results:
                chunk_hits = []
                for hit in res.points:
                    p = hit.payload.copy() if hit.payload else {}
                    p["_qdrant_score_"] = hit.score
                    chunk_hits.append(p)
                results.append(chunk_hits)
                
        return results

    # --- Classifier Tag Methods ---

    def upsert_tags(
        self,
        tags: List[str],
        dense_embs: np.ndarray,
        sparse_embs: List[dict],
        dict_type: str,
        domain: str = config.DOMAIN_MARKET,
        metadata_list: Optional[List[dict]] = None,
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
                    count_filter=Filter(must=[FieldCondition(key="dict_type", match=MatchValue(value=dict_type))])
                )
                if count_res.count >= len(tags):
                    logger.info(f"[QDRANT] Collection '{collection_name}' already contains {count_res.count} items for ({dict_type}). Skipping redundant upsert.")
                    return
            except Exception as e:
                logger.warning(f"[QDRANT] Could not check existing count for {collection_name} ({dict_type}): {e}")

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
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._call_with_retry("upsert", self.client.upsert, collection_name=collection_name, points=points[i : i + batch_size])
        logger.info(f"[QDRANT] Upserted {len(points)} items to {collection_name} ({dict_type}).")

    def search_hybrid_tags(
        self,
        dense_query: np.ndarray,
        sparse_query: Dict[str, float],
        limit: int = 50,
        filter_dict_type: Optional[str] = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags: Optional[List[str]] = None,
    ) -> Tuple[List[ScoredPoint], List[ScoredPoint]]:
        """Returns separate dense and sparse search results for classification fusion.
        
        If allowed_tags is provided, Qdrant will only return results whose 'tag' payload
        is in that list (used to restrict food GK search to the bt_gk_map).
        """
        collection_name = f"{domain}_tags"

        must_conditions = []
        if filter_dict_type:
            must_conditions.append(FieldCondition(key="dict_type", match=MatchValue(value=filter_dict_type)))
        if allowed_tags:
            must_conditions.append(FieldCondition(key="tag", match=MatchAny(any=allowed_tags)))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        dense_results = self._call_with_retry(
            "query_points (dense tags)",
            self.client.query_points,
            collection_name=collection_name,
            query=dense_query,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        ).points

        sparse_vector = self._parse_sparse_dict(sparse_query)
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
                with_payload=True
            ).points

        return dense_results, sparse_results

    def search_batch_hybrid_tags(
        self,
        dense_queries: List[np.ndarray],
        sparse_queries: List[Dict[str, float]],
        filter_dict_type: Optional[str] = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags_list: Optional[List[Optional[List[str]]]] = None,
        limit: int = 50,
    ) -> List[Tuple[List[ScoredPoint], List[ScoredPoint]]]:
        """Performs batch dense and sparse tag searches using query_batch_points for high throughput."""
        n = len(dense_queries)
        if n == 0:
            return []

        collection_name = f"{domain}_tags"
        if allowed_tags_list is None:
            allowed_tags_list = [None] * n

        requests = []
        has_sparse_flags = []

        for i in range(n):
            must_conditions = []
            if filter_dict_type:
                must_conditions.append(FieldCondition(key="dict_type", match=MatchValue(value=filter_dict_type)))
            if allowed_tags_list[i]:
                must_conditions.append(FieldCondition(key="tag", match=MatchAny(any=allowed_tags_list[i])))
            query_filter = Filter(must=must_conditions) if must_conditions else None

            # Dense request
            d_query = dense_queries[i].tolist() if hasattr(dense_queries[i], "tolist") else dense_queries[i]
            requests.append(QueryRequest(
                query=d_query,
                using="dense",
                filter=query_filter,
                limit=limit,
                with_payload=True
            ))

            # Sparse request
            sparse_vector = self._parse_sparse_dict(sparse_queries[i])
            if sparse_vector.indices:
                requests.append(QueryRequest(
                    query=sparse_vector,
                    using="sparse",
                    filter=query_filter,
                    limit=limit,
                    with_payload=True
                ))
                has_sparse_flags.append(True)
            else:
                has_sparse_flags.append(False)

        # Batch query points in chunks of 100 requests
        batch_size = 100
        raw_responses = []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            batch_res = self._call_with_retry(
                "query_batch_points (batch tags)",
                self.client.query_batch_points,
                collection_name=collection_name,
                requests=chunk
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
