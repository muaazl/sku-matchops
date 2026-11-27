"""Qdrant collection lifecycle, schema validation, payload indexes, and alias management."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    HnswConfigDiff,
    Modifier,
    OptimizersConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from engine.core import config
from engine.vector_store.qdrant_client_store import QdrantClientManager

logger = logging.getLogger("matchops.vector_store.collection")


def resolve_catalog_collection_name(domain: str) -> str:
    """Returns the catalog Qdrant collection name for the requested domain."""
    return (
        config.QDRANT_COLLECTION_FOOD
        if domain == config.DOMAIN_FOOD
        else config.QDRANT_COLLECTION_MARKET
    )


def resolve_tags_collection_name(domain: str) -> str:
    """Returns the tags Qdrant collection name for the requested domain."""
    prefix = str(getattr(config, "QDRANT_TAGS_PREFIX", "") or "")
    return f"{prefix}{domain}_tags"


def inspect_schema_mismatch(info: Any) -> str | None:
    """Returns a mismatch reason if Qdrant collection parameters conflict with active config."""
    params = getattr(getattr(info, "config", None), "params", None)
    if params is None:
        return None
    vectors = getattr(params, "vectors", None)
    vectors = getattr(vectors, "root", vectors)
    dense_cfg = (
        vectors.get("dense") if isinstance(vectors, dict) else getattr(vectors, "dense", vectors)
    )
    actual_dim = getattr(dense_cfg, "size", None)
    if isinstance(actual_dim, int) and not isinstance(actual_dim, bool):
        if actual_dim != config.EMBEDDING_DIM:
            return f"dense dim={actual_dim}, expected {config.EMBEDDING_DIM}"
        sparse_vectors = getattr(params, "sparse_vectors", None)
        actual_sparse = set(sparse_vectors.keys()) if isinstance(sparse_vectors, dict) else set()
        if "bm25" not in actual_sparse:
            return f"missing 'bm25' sparse vector (found {sorted(actual_sparse)})"
        has_sparse = "sparse" in actual_sparse
        if has_sparse != config.check_learned_sparse_enabled():
            return (
                f"learned sparse mismatch (collection has_sparse={has_sparse}, "
                f"expected={config.check_learned_sparse_enabled()})"
            )
    return None


def cleanup_local_collection_dir(collection_name: str) -> None:
    """Removes orphaned local collection storage folders when QDRANT_DATA_DIR exists."""
    qdir_raw = str(getattr(config, "QDRANT_DATA_DIR", "") or "").strip()
    if not qdir_raw:
        return
    qdir = Path(qdir_raw)
    if not qdir.exists():
        return
    try:
        (qdir / ".deleted").mkdir(parents=True, exist_ok=True)
        col_dir = qdir / "collections" / collection_name
        if col_dir.exists():
            shutil.rmtree(col_dir, ignore_errors=True)
    except Exception:
        pass


class QdrantCollectionStore:
    """Manages Qdrant collection schemas, payload indexes, aliases, and deletion."""

    def __init__(
        self,
        client_getter: Callable[[], QdrantClient],
        retry_caller: Callable[..., Any],
        checked_cache_getter: Callable[[], set[str]] | None = None,
    ) -> None:
        self._get_client = client_getter
        self._retry = retry_caller
        self._get_checked_cache = checked_cache_getter or (
            lambda: QdrantClientManager._checked_collections
        )

    def get_collection_schema_mismatch(self, collection_name: str) -> str | None:
        """Inspects an existing collection and reports any mismatch against active config."""
        try:
            client = self._get_client()
            if not self._retry("collection_exists", client.collection_exists, collection_name):
                return None
            info = self._retry(
                "get_collection",
                self._get_client().get_collection,
                collection_name=collection_name,
            )
        except Exception as exc:
            logger.debug("[QDRANT] Could not inspect config on '%s': %s", collection_name, exc)
            return None
        return inspect_schema_mismatch(info)

    def ensure_collection_ready(self, collection_name: str) -> None:
        """Creates a missing collection or validates schema + HNSW/index setup on existing ones."""
        with QdrantClientManager._lock:
            checked_collections = self._get_checked_cache()
            if collection_name in checked_collections:
                return

            client = self._get_client()
            hnsw_on_disk = getattr(config, "QDRANT_HNSW_ON_DISK", False)
            exists = self._retry("collection_exists", client.collection_exists, collection_name)
            if not exists:
                cleanup_local_collection_dir(collection_name)
                sparse_cfg: dict[str, SparseVectorParams] = {
                    "bm25": SparseVectorParams(
                        index=SparseIndexParams(on_disk=True),
                        modifier=Modifier.IDF,
                    )
                }
                if config.check_learned_sparse_enabled():
                    sparse_cfg["sparse"] = SparseVectorParams(index=SparseIndexParams(on_disk=True))

                create_kwargs = {
                    "collection_name": collection_name,
                    "vectors_config": {
                        "dense": VectorParams(
                            size=config.EMBEDDING_DIM,
                            distance=Distance.COSINE,
                            on_disk=True,
                        )
                    },
                    "sparse_vectors_config": sparse_cfg,
                    "hnsw_config": HnswConfigDiff(
                        on_disk=hnsw_on_disk,
                        m=getattr(config, "QDRANT_HNSW_M", 16),
                        ef_construct=getattr(config, "QDRANT_HNSW_EF_CONSTRUCT", 100),
                    ),
                    "optimizers_config": OptimizersConfigDiff(
                        indexing_threshold=getattr(config, "QDRANT_INDEXING_THRESHOLD", 10000),
                        memmap_threshold=getattr(config, "QDRANT_MEMMAP_THRESHOLD", 5000),
                    ),
                }
                try:
                    self._retry(
                        "create_collection",
                        self._get_client().create_collection,
                        **create_kwargs,
                    )
                except Exception as create_err:
                    err_text = str(create_err).lower()
                    if "file exists" in err_text or "os error 17" in err_text:
                        logger.warning(
                            "[QDRANT] Orphaned storage directory detected for '%s'; "
                            "cleaning up and retrying create_collection...",
                            collection_name,
                        )
                        cleanup_local_collection_dir(collection_name)
                        try:
                            self._get_client().delete_collection(collection_name=collection_name)
                        except Exception:
                            pass
                        cleanup_local_collection_dir(collection_name)
                        self._retry(
                            "create_collection",
                            self._get_client().create_collection,
                            **create_kwargs,
                        )
                    else:
                        raise
                self.create_payload_indexes(collection_name)
            else:
                self.create_payload_indexes(collection_name)
                info = None
                try:
                    info = self._retry(
                        "get_collection",
                        self._get_client().get_collection,
                        collection_name=collection_name,
                    )
                except Exception as exc:
                    logger.debug(
                        "[QDRANT] Could not inspect config on '%s': %s", collection_name, exc
                    )

                if info is not None:
                    mismatch = inspect_schema_mismatch(info)
                    if mismatch:
                        raise RuntimeError(
                            f"Qdrant collection '{collection_name}' schema mismatch ({mismatch}). "
                            "Run `python -m scripts.sync --qdrant rebuild` to recreate collections for the active model."
                        )
                    client_after = self._get_client()
                    if not hnsw_on_disk and hasattr(client_after, "update_collection"):
                        try:
                            cur_hnsw = getattr(getattr(info, "config", None), "hnsw_config", None)
                            if getattr(cur_hnsw, "on_disk", None) is True:
                                self._retry(
                                    "update_collection",
                                    client_after.update_collection,
                                    collection_name=collection_name,
                                    hnsw_config=HnswConfigDiff(on_disk=False),
                                )
                        except Exception as exc:
                            logger.debug(
                                "[QDRANT] Could not inspect/update HNSW config on '%s': %s",
                                collection_name,
                                exc,
                            )
            self._get_checked_cache().add(collection_name)

    def create_payload_indexes(self, collection_name: str) -> None:
        """Creates keyword payload indexes for fields used in filtered vector queries."""
        index_fields = ("dict_type", "tag") if collection_name.endswith("_tags") else ("basictype",)
        for field_name in index_fields:
            try:
                self._retry(
                    "create_payload_index",
                    self._get_client().create_payload_index,
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.debug("[QDRANT] Could not create payload index on '%s': %s", field_name, exc)

    def create_versioned_collection(self, alias_name: str, version: int = 1) -> str:
        """Creates a versioned physical collection (e.g. `{alias_name}_v{version}`)."""
        physical_name = f"{alias_name}_v{version}"
        self.ensure_collection_ready(physical_name)
        return physical_name

    def swap_collection_alias(self, alias_name: str, new_physical_collection: str) -> None:
        """Atomically points an alias to a new physical collection for zero-downtime swaps."""
        operations: list[DeleteAliasOperation | CreateAliasOperation] = []
        try:
            aliases_res = self._retry("get_aliases", self._get_client().get_aliases)
            for item in getattr(aliases_res, "aliases", []):
                if getattr(item, "alias_name", None) == alias_name:
                    operations.append(
                        DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name))
                    )
        except Exception as exc:
            logger.debug("[QDRANT] Could not inspect existing aliases for %s: %s", alias_name, exc)

        operations.append(
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=new_physical_collection,
                    alias_name=alias_name,
                )
            )
        )
        self._retry(
            "update_collection_aliases",
            self._get_client().update_collection_aliases,
            change_aliases_operations=operations,
        )
        logger.info(
            "[QDRANT] Successfully pointed alias '%s' -> '%s'.",
            alias_name,
            new_physical_collection,
        )

    def get_collection_schema_info(self, collection_name: str) -> dict[str, Any]:
        """Returns collection status, counts, and config metadata for schema inspection."""
        try:
            info = self._retry(
                "get_collection",
                self._get_client().get_collection,
                collection_name=collection_name,
            )
            return {
                "status": getattr(info, "status", None),
                "vectors_count": getattr(info, "vectors_count", None),
                "points_count": getattr(info, "points_count", None),
                "config": getattr(info, "config", None),
            }
        except Exception as exc:
            logger.warning("[QDRANT] Could not get schema info for '%s': %s", collection_name, exc)
            return {}

    def delete_named_collection(self, collection_name: str) -> None:
        """Deletes a Qdrant collection by exact name and cleans up local storage folders."""
        with QdrantClientManager._lock:
            self._get_checked_cache().discard(collection_name)
            QdrantClientManager._checked_collections.discard(collection_name)

        qdir_raw = str(getattr(config, "QDRANT_DATA_DIR", "") or "").strip()
        if qdir_raw:
            qdir = Path(qdir_raw)
            if qdir.exists():
                try:
                    (qdir / ".deleted").mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

        client = self._get_client()
        if self._retry("collection_exists", client.collection_exists, collection_name):
            try:
                self._retry(
                    "delete_collection",
                    self._get_client().delete_collection,
                    collection_name=collection_name,
                )
            finally:
                cleanup_local_collection_dir(collection_name)
        else:
            cleanup_local_collection_dir(collection_name)
