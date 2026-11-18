from unittest.mock import MagicMock

from qdrant_client.http.models import Modifier

from engine import config
from engine.data_pipeline.vector_store import VectorStore


def test_vector_store_lifecycle_and_alias_swapping(monkeypatch):
    store = VectorStore.__new__(VectorStore)
    store.client = MagicMock()
    store._checked_collections = set()

    # Mock collection_exists and get_aliases
    store.client.collection_exists.return_value = True
    aliases_mock = MagicMock()
    alias_item = MagicMock()
    alias_item.alias_name = "food_catalog"
    aliases_mock.aliases = [alias_item]
    store.client.get_aliases.return_value = aliases_mock

    # Test create_versioned_collection
    phys_col = store.create_versioned_collection("food_catalog", version=2)
    assert phys_col == "food_catalog_v2"
    assert "food_catalog_v2" in store._checked_collections

    # Test swap_collection_alias
    store.swap_collection_alias("food_catalog", "food_catalog_v2")
    assert store.client.update_collection_aliases.called
    kwargs = store.client.update_collection_aliases.call_args[1]
    ops = kwargs.get("change_aliases_operations", [])
    assert len(ops) == 2  # Delete old alias + Create new alias

    # Test get_collection_schema_info
    coll_info_mock = MagicMock()
    coll_info_mock.status = "green"
    coll_info_mock.vectors_count = 1200
    coll_info_mock.points_count = 1200
    store.client.get_collection.return_value = coll_info_mock
    info = store.get_collection_schema_info("food_catalog_v2")
    assert info["status"] == "green"
    assert info["points_count"] == 1200


def test_ensure_collection_uses_embedding_dim_and_bm25_idf(monkeypatch):
    store = VectorStore.__new__(VectorStore)
    store.client = MagicMock()
    store._checked_collections = set()
    store.client.collection_exists.return_value = False

    monkeypatch.setattr(config, "has_learned_sparse", lambda: False)
    store._ensure_collection("market_catalog")

    kwargs = store.client.create_collection.call_args[1]
    assert kwargs["vectors_config"]["dense"].size == config.EMBEDDING_DIM
    assert kwargs["sparse_vectors_config"]["bm25"].modifier == Modifier.IDF
    assert "sparse" not in kwargs["sparse_vectors_config"]

    store._checked_collections.clear()
    store.client.create_collection.reset_mock()
    monkeypatch.setattr(config, "has_learned_sparse", lambda: True)
    store._ensure_collection("food_catalog")

    kwargs_bge = store.client.create_collection.call_args[1]
    assert kwargs_bge["sparse_vectors_config"]["bm25"].modifier == Modifier.IDF
    assert "sparse" in kwargs_bge["sparse_vectors_config"]
    assert kwargs_bge["sparse_vectors_config"]["sparse"].modifier is None


def test_search_from_vectors_uses_2way_or_3way_rrf(monkeypatch):
    import numpy as np

    store = VectorStore.__new__(VectorStore)
    store.client = MagicMock()
    store._checked_collections = {config.QDRANT_COLLECTION_MARKET}
    store.client.query_points.return_value = MagicMock(points=[])

    # Non-BGE-M3: Dense + BM25 only (2-way RRF), even if sparse dict passed
    monkeypatch.setattr(config, "has_learned_sparse", lambda: False)
    store.search_from_vectors(
        dense_vec=np.ones(4, dtype=np.float32),
        sparse_vec_dict={10: 0.9},
        bm25_vec_dict={20: 0.5},
        domain="market",
    )
    pref_non_bge = store.client.query_points.call_args.kwargs["prefetch"]
    assert [p.using for p in pref_non_bge] == ["dense", "bm25"]

    # BGE-M3: Dense + BM25 + Sparse (3-way RRF)
    store.client.query_points.reset_mock()
    monkeypatch.setattr(config, "has_learned_sparse", lambda: True)
    store.search_from_vectors(
        dense_vec=np.ones(4, dtype=np.float32),
        sparse_vec_dict={10: 0.9},
        bm25_vec_dict={20: 0.5},
        domain="market",
    )
    pref_bge = store.client.query_points.call_args.kwargs["prefetch"]
    assert [p.using for p in pref_bge] == ["dense", "bm25", "sparse"]


def test_ensure_collection_raises_on_schema_mismatch_and_get_existing_hashes(monkeypatch):
    import pytest
    from qdrant_client.http.models import (
        CollectionParams,
        Distance,
        PayloadSelectorExclude,
        SparseVectorParams,
        VectorParams,
    )

    store = VectorStore.__new__(VectorStore)
    store.client = MagicMock()
    store._checked_collections = set()

    # 1. Schema mismatch detection in _ensure_collection
    monkeypatch.setattr(config, "EMBEDDING_DIM", 1024)
    monkeypatch.setattr(config, "has_learned_sparse", lambda: False)
    store.client.collection_exists.return_value = True
    bad_info = MagicMock()
    bad_info.config.params = CollectionParams(
        vectors={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )
    store.client.get_collection.return_value = bad_info

    with pytest.raises(RuntimeError, match="--qdrant rebuild"):
        store._ensure_collection("market_catalog")

    # 2. Preserving row_hash in payload while excluding from searches
    sanitized = VectorStore._sanitize_catalog_payload(
        {"Name": "Test Item", "row_hash": "abc123", "Price": "15.0"},
        domain="market",
    )
    assert sanitized["row_hash"] == "abc123"

    pt = MagicMock(payload={"db_uid": "test item#occ_1", "row_hash": "abc123"})
    store.client.scroll.return_value = ([pt], None)
    assert store.get_existing_hashes("market") == {"test item#occ_1": "abc123"}

    store._checked_collections.add(config.QDRANT_COLLECTION_MARKET)
    hit = MagicMock(payload={"Name": "Test Item", "row_hash": "abc123"}, score=0.91)
    store.client.query_points.return_value = MagicMock(points=[hit])
    hits = store.search_from_vectors(dense_vec=[0.1, 0.2], domain="market")
    assert hits == [{"Name": "Test Item", "_qdrant_score_": 0.91}]
    assert isinstance(
        store.client.query_points.call_args.kwargs["with_payload"], PayloadSelectorExclude
    )
