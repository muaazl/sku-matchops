"""Qdrant upsert, hash diffing, and hybrid dense/BM25/sparse search queries."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PayloadSelectorExclude,
    PointStruct,
    Prefetch,
    QueryRequest,
    ScoredPoint,
    SparseVector,
)

from engine.core import config
from engine.preprocessing.weight_preprocessor import clean_price_value
from engine.vector_store.collection_store import (
    QdrantCollectionStore,
    resolve_catalog_collection_name,
    resolve_tags_collection_name,
)

logger = logging.getLogger("matchops.vector_store.query")

SEARCH_PAYLOAD = PayloadSelectorExclude(exclude=["row_hash"])


def parse_sparse_dict(sparse_dict: dict[str, float] | None) -> SparseVector:
    """Converts a lexical weight dictionary into a Qdrant SparseVector."""
    indices: list[int] = []
    values: list[float] = []
    if sparse_dict and isinstance(sparse_dict, dict):
        for k, v in sparse_dict.items():
            try:
                indices.append(int(k))
                values.append(float(v))
            except (ValueError, TypeError):
                pass
    return SparseVector(indices=indices, values=values)


def sanitize_catalog_payload(raw_payload: dict[str, Any], domain: str) -> dict[str, Any]:
    """Strips junk columns, normalizes numeric prices, and cleans entity payloads."""
    cleaned: dict[str, Any] = {}
    junk_prefixes = ("_duplicated_", "Unnamed")
    for k, v in raw_payload.items():
        k_str = str(k).strip()
        if not k_str or any(k_str.startswith(prefix) for prefix in junk_prefixes):
            continue
        if v is pd.NA or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = None
        cleaned[k_str] = v

    if "Price" in cleaned and "price" in cleaned:
        try:
            cleaned["price"] = float(cleaned.get("price") or 0.0)
        except Exception:
            pass
        del cleaned["Price"]
    elif "Price" in cleaned:
        cleaned["price"] = clean_price_value(cleaned["Price"])
        del cleaned["Price"]

    # Food domain weight cleanup: drop unpopulated weight_val triples.
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


def build_point_vectors(
    dense_vec: Any,
    bm25_dict: dict | None,
    sparse_dict: dict | None,
    include_learned_sparse: bool,
) -> dict[str, Any]:
    """Builds named vector dictionaries (`dense`, `bm25`, optional `sparse`) for Qdrant upsert."""
    vecs: dict[str, Any] = {
        "dense": dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec),
        "bm25": parse_sparse_dict(bm25_dict or {}),
    }
    if include_learned_sparse:
        vecs["sparse"] = parse_sparse_dict(sparse_dict or {})
    return vecs


def build_hybrid_prefetch(
    dense_vec: np.ndarray,
    bm25_dict: dict | None,
    sparse_dict: dict | None,
    limit: int,
) -> list[Prefetch]:
    """Constructs Dense + BM25 + optional learned sparse Prefetch requests for RRF."""
    d_list = dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)
    prefetches = [Prefetch(query=d_list, using="dense", limit=limit)]
    bm25_sv = parse_sparse_dict(bm25_dict or {})
    if bm25_sv.indices:
        prefetches.append(Prefetch(query=bm25_sv, using="bm25", limit=limit))
    if config.check_learned_sparse_enabled() and sparse_dict:
        sparse_sv = parse_sparse_dict(sparse_dict)
        if sparse_sv.indices:
            prefetches.append(Prefetch(query=sparse_sv, using="sparse", limit=limit))
    return prefetches


def convert_hit_to_dict(hit: Any) -> dict[str, Any]:
    """Converts a Qdrant ScoredPoint into a payload dictionary with `_qdrant_score_`."""
    payload = hit.payload.copy() if hit.payload else {}
    payload.pop("row_hash", None)
    payload["_qdrant_score_"] = float(hit.score)
    return payload


def build_qdrant_filter(filters_json: str) -> Filter | None:
    """Parses JSON filter specifications into a Qdrant Filter object."""
    if not filters_json:
        return None
    raw = json.loads(filters_json)
    if not isinstance(raw, dict):
        raise ValueError("filters must be a JSON object.")
    if not raw:
        return None
    if any(k in raw for k in ("must", "should", "must_not")):
        return Filter.model_validate(raw)
    must = [
        FieldCondition(
            key=str(k),
            match=MatchAny(any=v) if isinstance(v, list) else MatchValue(value=v),
        )
        for k, v in raw.items()
        if v is not None
    ]
    return Filter(must=must) if must else None


class VectorQueryStore:
    """Executes catalog and tag upserts, hash scrolling, and hybrid vector queries."""

    def __init__(
        self,
        client_getter: Callable[[], QdrantClient],
        retry_caller: Callable[..., Any],
        collection_store: QdrantCollectionStore,
    ) -> None:
        self._get_client = client_getter
        self._retry = retry_caller
        self._collections = collection_store

    def check_collection_exists(self, collection_name: str) -> bool:
        """Checks whether a Qdrant collection exists with transient retry handling."""
        return bool(
            self._retry("collection_exists", self._get_client().collection_exists, collection_name)
        )

    def count_collection_points(self, collection_name: str) -> int:
        """Returns the total point count inside a Qdrant collection."""
        count_res = self._retry("count", self._get_client().count, collection_name=collection_name)
        return int(getattr(count_res, "count", 0) or 0)

    def upsert_catalog_points(
        self,
        df: pd.DataFrame,
        dense_embeddings: np.ndarray,
        sparse_weights: list[dict] | None = None,
        bm25_weights: list[dict] | None = None,
        domain: str = config.DOMAIN_MARKET,
    ) -> None:
        """Upserts a DataFrame of catalog items and their multi-vector embeddings to Qdrant."""
        if df.empty:
            return

        collection_name = resolve_catalog_collection_name(domain)
        self._collections.ensure_collection_ready(collection_name)
        include_learned_sparse = config.check_learned_sparse_enabled() and bool(sparse_weights)

        points: list[PointStruct] = []
        records = df.to_dict(orient="records")
        indices = df.index.tolist()
        for list_idx, (orig_idx, payload) in enumerate(zip(indices, records)):
            db_uid = payload.get("db_uid")
            unique_str = db_uid if db_uid else f"{orig_idx}_{payload.get('Name', '')}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

            sanitized_payload = sanitize_catalog_payload(payload, domain=domain)
            bm25_d = bm25_weights[list_idx] if bm25_weights and list_idx < len(bm25_weights) else {}
            sparse_d = (
                sparse_weights[list_idx]
                if include_learned_sparse and sparse_weights and list_idx < len(sparse_weights)
                else None
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=build_point_vectors(
                        dense_embeddings[list_idx],
                        bm25_d,
                        sparse_d,
                        include_learned_sparse=include_learned_sparse,
                    ),
                    payload=sanitized_payload,
                )
            )

            if len(points) >= config.UPSERT_BATCH_SIZE:
                self._retry(
                    "upsert",
                    self._get_client().upsert,
                    collection_name=collection_name,
                    points=points,
                )
                points = []

        if points:
            self._retry(
                "upsert",
                self._get_client().upsert,
                collection_name=collection_name,
                points=points,
            )

    def fetch_existing_hashes(self, domain: str = config.DOMAIN_MARKET) -> dict[str, str]:
        """Scrolls Qdrant payloads to collect `{db_uid: row_hash}` for incremental sync."""
        collection_name = resolve_catalog_collection_name(domain)
        if not self.check_collection_exists(collection_name):
            return {}

        result: dict[str, str] = {}
        next_offset = None
        while True:
            points, next_offset = self._retry(
                "scroll",
                self._get_client().scroll,
                collection_name=collection_name,
                limit=1000,
                offset=next_offset,
                with_payload=["db_uid", "row_hash"],
                with_vectors=False,
            )
            for pt in points:
                payload = pt.payload or {}
                uid = payload.get("db_uid")
                row_hash = payload.get("row_hash")
                if uid and row_hash:
                    result[uid] = row_hash
            if next_offset is None:
                break
        return result

    def search_catalog_hybrid(
        self,
        dense_vec: np.ndarray,
        sparse_vec_dict: dict[str, float] | None = None,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bt_filter: str | None = None,
        bm25_vec_dict: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Executes single-query RRF hybrid search across dense, BM25, and sparse vectors."""
        collection_name = resolve_catalog_collection_name(domain)
        self._collections.ensure_collection_ready(collection_name)
        prefetch = build_hybrid_prefetch(dense_vec, bm25_vec_dict, sparse_vec_dict, top_k)

        must_conditions = []
        if bt_filter:
            must_conditions.append(
                FieldCondition(key="basictype", match=MatchValue(value=bt_filter))
            )
        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self._retry(
            "query_points",
            self._get_client().query_points,
            collection_name=collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=SEARCH_PAYLOAD,
        )
        return [convert_hit_to_dict(hit) for hit in results.points]

    def search_catalog_hybrid_batch(
        self,
        dense_vecs: list[np.ndarray],
        sparse_vec_dicts: list[dict[str, float]] | None = None,
        bt_filters: list[str | None] | None = None,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bm25_vec_dicts: list[dict[str, float]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Executes chunked batch RRF hybrid catalog search via `query_batch_points`."""
        if len(dense_vecs) == 0:
            return []

        collection_name = resolve_catalog_collection_name(domain)
        self._collections.ensure_collection_ready(collection_name)
        n = len(dense_vecs)
        bt_filters = bt_filters or [None] * n

        requests: list[QueryRequest] = []
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

            requests.append(
                QueryRequest(
                    prefetch=build_hybrid_prefetch(dense_vecs[i], bm25_d, sparse_d, top_k),
                    query=FusionQuery(fusion=Fusion.RRF),
                    filter=query_filter,
                    limit=top_k,
                    with_payload=SEARCH_PAYLOAD,
                )
            )

        results: list[list[dict[str, Any]]] = []
        batch_size = config.QDRANT_BATCH_CHUNK_SIZE
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            batch_results = self._retry(
                "query_batch_points",
                self._get_client().query_batch_points,
                collection_name=collection_name,
                requests=chunk,
            )
            for res in batch_results:
                results.append([convert_hit_to_dict(hit) for hit in res.points])

        return results

    def search_catalog_neighbors(
        self,
        dense_vec: np.ndarray,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        """Retrieves nearest catalog SKUs by dense cosine similarity for Tier 2 few-shot."""
        collection_name = resolve_catalog_collection_name(domain)
        self._collections.ensure_collection_ready(collection_name)
        d_vec = dense_vec.tolist() if hasattr(dense_vec, "tolist") else dense_vec
        try:
            results = self._retry(
                "query_points (catalog neighbors)",
                self._get_client().query_points,
                collection_name=collection_name,
                query=d_vec,
                using="dense",
                limit=top_k,
                with_payload=SEARCH_PAYLOAD,
            )
            return [convert_hit_to_dict(hit) for hit in results.points]
        except Exception as exc:
            logger.warning("[QDRANT] search_catalog_neighbors failed: %s", exc)
            return []

    def search_batch_catalog_neighbors(
        self,
        dense_vecs: list[np.ndarray],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = 15,
    ) -> list[list[dict[str, Any]]]:
        """Batch-retrieves nearest catalog SKUs by dense cosine similarity."""
        n = len(dense_vecs)
        if n == 0:
            return []

        collection_name = resolve_catalog_collection_name(domain)
        self._collections.ensure_collection_ready(collection_name)

        requests: list[QueryRequest] = []
        for vec in dense_vecs:
            d_vec = vec.tolist() if hasattr(vec, "tolist") else vec
            requests.append(
                QueryRequest(
                    query=d_vec,
                    using="dense",
                    limit=top_k,
                    with_payload=SEARCH_PAYLOAD,
                )
            )

        results: list[list[dict[str, Any]]] = []
        batch_size = config.QDRANT_BATCH_CHUNK_SIZE
        try:
            for i in range(0, len(requests), batch_size):
                chunk = requests[i : i + batch_size]
                batch_res = self._retry(
                    "query_batch_points (catalog neighbors)",
                    self._get_client().query_batch_points,
                    collection_name=collection_name,
                    requests=chunk,
                )
                for res in batch_res:
                    results.append([convert_hit_to_dict(hit) for hit in res.points])
        except Exception as exc:
            logger.warning("[QDRANT] search_batch_catalog_neighbors failed: %s", exc)
            results.extend([[] for _ in range(n - len(results))])

        return results

    def check_tags_synced(self, domain: str, dict_type: str, expected_count: int) -> bool:
        """Checks whether Qdrant already stores `expected_count` tags under the active model."""
        if expected_count <= 0:
            return True
        collection_name = resolve_tags_collection_name(domain)
        try:
            self._collections.ensure_collection_ready(collection_name)
            type_filter = Filter(
                must=[FieldCondition(key="dict_type", match=MatchValue(value=dict_type))]
            )
            count_res = self._retry(
                "count",
                self._get_client().count,
                collection_name=collection_name,
                count_filter=type_filter,
            )
            if count_res.count >= expected_count:
                scroll_res = self._retry(
                    "scroll",
                    self._get_client().scroll,
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
                        "[QDRANT] Collection '%s' already contains %d items for (%s). Skipping redundant upsert.",
                        collection_name,
                        count_res.count,
                        dict_type,
                    )
                    return True
                logger.info(
                    "[QDRANT] Re-upserting '%s' (%s): model changed (%r -> %r).",
                    collection_name,
                    dict_type,
                    existing_model,
                    config.EMBEDDING_MODEL,
                )
        except Exception as exc:
            logger.warning(
                "[QDRANT] Could not check existing count for %s (%s): %s",
                collection_name,
                dict_type,
                exc,
            )
        return False

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
        """Upserts dictionary tags and their embeddings for hybrid tag retrieval."""
        if not tags:
            return

        collection_name = resolve_tags_collection_name(domain)
        self._collections.ensure_collection_ready(collection_name)

        if not force and self.check_tags_synced(domain, dict_type, len(set(tags))):
            return

        include_learned_sparse = config.check_learned_sparse_enabled() and bool(sparse_embs)
        points: list[PointStruct] = []
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
                    vector=build_point_vectors(
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
            self._retry(
                "upsert",
                self._get_client().upsert,
                collection_name=collection_name,
                points=points[i : i + batch_size],
                wait=False,
            )
        logger.info(
            "[QDRANT] Upserted %d items to %s (%s).", len(points), collection_name, dict_type
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
        """Returns `(dense_hits, bm25_hits, sparse_hits)` for single-query tag fusion."""
        return self.search_tags_hybrid_batch(
            dense_queries=[dense_query],
            sparse_queries=[sparse_query] if sparse_query is not None else None,
            filter_dict_type=filter_dict_type,
            domain=domain,
            allowed_tags_list=[allowed_tags],
            limit=limit,
            bm25_queries=[bm25_query] if bm25_query is not None else None,
        )[0]

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
        """Executes batch dense, BM25, and optional sparse tag searches in one batch RPC."""
        n = len(dense_queries)
        if n == 0:
            return []

        collection_name = resolve_tags_collection_name(domain)
        client = self._get_client()
        if hasattr(client, "collection_exists"):
            self._collections.ensure_collection_ready(collection_name)
        if allowed_tags_list is None:
            allowed_tags_list = [None] * n

        use_learned_sparse = config.check_learned_sparse_enabled()
        requests: list[QueryRequest] = []
        has_bm25_flags: list[bool] = []
        has_sparse_flags: list[bool] = []

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
            bm25_vector = parse_sparse_dict(bm25_d) if bm25_d else None
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
                parse_sparse_dict(sparse_d) if (use_learned_sparse and sparse_d) else None
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
            batch_res = self._retry(
                "query_batch_points (batch tags)",
                self._get_client().query_batch_points,
                collection_name=collection_name,
                requests=chunk,
            )
            raw_responses.extend(batch_res)

        results: list[tuple[list[ScoredPoint], list[ScoredPoint], list[ScoredPoint]]] = []
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
        """Executes raw admin vector search (dense, sparse BM25 length-norm, or hybrid RRF)."""
        query_filter = build_qdrant_filter(filters_json)
        dense_list = dense_vec.tolist() if hasattr(dense_vec, "tolist") else dense_vec
        final_limit = limit or 10

        if mode == "sparse":
            bm25_vector = parse_sparse_dict(bm25_vec_dict or {})
            q_len = len(normalized_query.split())
            fetch_limit = max(final_limit * 3, 30)
            raw_points = self._retry(
                "query_points (bm25 collection search)",
                self._get_client().query_points,
                collection_name=collection_name,
                query=bm25_vector,
                using="bm25",
                limit=fetch_limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ).points

            scored_points: list[tuple[float, bool, Any]] = []
            for hit in raw_points:
                p_load = hit.payload or {}
                clean_name = str(p_load.get("clean_text") or "").strip().lower()
                doc_len = p_load.get("token_count") or (
                    len(clean_name.split()) if clean_name else q_len
                )
                len_ratio = min(1.0, q_len / max(1, doc_len)) if q_len > 0 else 1.0
                is_exact = bool(clean_name and clean_name == normalized_query)
                boost = 1.25 if is_exact else 1.0
                adjusted_score = hit.score * len_ratio * boost
                hit.score = float(adjusted_score)
                scored_points.append((adjusted_score, is_exact, hit))

            scored_points.sort(key=lambda x: (x[0], 1 if x[1] else 0), reverse=True)
            points = [sp[2] for sp in scored_points[:final_limit]]
        elif mode == "hybrid":
            bm25_vector = parse_sparse_dict(bm25_vec_dict or {})
            sparse_vector = parse_sparse_dict(sparse_vec_dict or {})
            prefetches = [Prefetch(query=dense_list, using="dense", limit=final_limit)]
            if bm25_vector.indices:
                prefetches.append(Prefetch(query=bm25_vector, using="bm25", limit=final_limit))
            if config.check_learned_sparse_enabled() and sparse_vector.indices:
                prefetches.append(Prefetch(query=sparse_vector, using="sparse", limit=final_limit))
            points = self._retry(
                "query_points (hybrid collection search)",
                self._get_client().query_points,
                collection_name=collection_name,
                prefetch=prefetches,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=final_limit,
                query_filter=query_filter,
                with_payload=True,
            ).points
        else:
            points = self._retry(
                "query_points (dense collection search)",
                self._get_client().query_points,
                collection_name=collection_name,
                query=dense_list,
                using="dense",
                limit=final_limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ).points

        return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in points]
