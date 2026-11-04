from unittest.mock import MagicMock

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
