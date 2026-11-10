from unittest.mock import MagicMock, patch

from backend.app.api.endpoints.catalog import search_catalog


def test_catalog_search_dataset_catalog_no_duplicate_keys():
    """Verify search_catalog for dataset='catalog' returns only clean snake_case keys with NO case duplicates."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = 1
    # Mock row tuple matching:
    # id, name, brand, flavor, basictype, generic_keywords, price, category, region, description
    mock_row = (
        88434,
        "10 Great Wall Staples",
        "",
        None,
        "Staple Pins",
        "Stationery, Staple Pins",
        50.0,
        "Stationery",
        None,
        None,
    )
    mock_conn.execute.return_value.fetchall.return_value = [mock_row]

    with patch("backend.app.api.endpoints.catalog.get_connection", return_value=mock_conn):
        res = search_catalog(dataset="catalog", domain="market", page=1, page_size=25)

    assert "results" in res
    assert len(res["results"]) == 1
    item = res["results"][0]

    # Expected clean keys
    expected_keys = {
        "id",
        "name",
        "brand",
        "flavor",
        "basictype",
        "generic_keywords",
        "price",
        "category",
        "region",
        "description",
        "count",
    }
    assert set(item.keys()) == expected_keys

    # Strictly assert NO capitalized duplicate keys exist
    forbidden_duplicates = [
        "Name",
        "Brand",
        "Flavor",
        "Generic keywords",
        "Price",
        "Category",
        "Region",
        "Description",
    ]
    for dup in forbidden_duplicates:
        assert dup not in item, f"Found duplicate key '{dup}' in catalog item: {item}"

    assert item["id"] == "88434"
    assert item["name"] == "10 Great Wall Staples"
    assert item["basictype"] == "Staple Pins"
    assert item["generic_keywords"] == "Stationery, Staple Pins"
    assert item["price"] == 50.0
    assert item["category"] == "Stationery"


def test_catalog_search_dataset_brands_no_redundant_aliases():
    """Verify search_catalog for dataset='brands' does not duplicate name into brand_name/flavor_name."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = 1
    # Mock row tuple matching:
    # id, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, catalog_count
    mock_row = (1, "Anchor", "Anchor Butter", False, False, False, False, 42)
    mock_conn.execute.return_value.fetchall.return_value = [mock_row]

    with patch("backend.app.api.endpoints.catalog.get_connection", return_value=mock_conn):
        res = search_catalog(dataset="brands", domain="market", page=1, page_size=25)

    assert len(res["results"]) == 1
    item = res["results"][0]
    assert "name" in item
    assert "brand_name" not in item
    assert "flavor_name" not in item
    assert item["name"] == "Anchor"
    assert item["count"] == 42
    assert item["is_weak"] is False


def test_catalog_search_dataset_bt_gk_map_no_redundant_name():
    """Verify search_catalog for dataset='bt_gk_map' returns bt without duplicate name key."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = 1
    # Mock row tuple matching:
    # id, basictype, generic_keywords, gk_count, catalog_count
    mock_row = (1, "Staple Pins", "Stationery, Staple Pins", 2, 10)
    mock_conn.execute.return_value.fetchall.return_value = [mock_row]

    with patch("backend.app.api.endpoints.catalog.get_connection", return_value=mock_conn):
        res = search_catalog(dataset="bt_gk_map", domain="market", page=1, page_size=25)

    assert len(res["results"]) == 1
    item = res["results"][0]
    assert "bt" in item
    assert "name" not in item
    assert item["bt"] == "Staple Pins"
    assert item["gks"] == "Stationery, Staple Pins"
    assert item["gk_count"] == 2
    assert item["count"] == 10


def test_catalog_snapshot_feather_endpoints():
    """Verify internal snapshot endpoints return valid Feather streams from database DataFrames."""
    import io

    import pandas as pd
    import pyarrow.feather as feather

    from backend.app.api.endpoints.catalog import (
        get_brands_feather_snapshot,
        get_catalog_feather_snapshot,
        get_dictionaries_snapshot,
    )

    mock_cat = pd.DataFrame([{"Name": "Item A", "Price": 100.0, "clean_text": "item a"}])
    mock_brands = pd.DataFrame([{"Brand Name": "Brand X", "Aliases": "bx"}])
    mock_dicts = {"gk": ["dairy"], "bt": ["milk"], "category": ["groceries"]}

    with patch(
        "backend.app.api.endpoints.catalog.get_catalog_and_brands",
        return_value=(mock_cat, mock_brands),
    ), patch(
        "backend.app.api.endpoints.catalog.get_classifier_dicts",
        return_value=mock_dicts,
    ):
        # 1. Catalog Feather snapshot
        resp_cat = get_catalog_feather_snapshot("market")
        assert resp_cat.media_type == "application/octet-stream"
        df_cat = feather.read_feather(io.BytesIO(resp_cat.body))
        assert len(df_cat) == 1
        assert df_cat.iloc[0]["Name"] == "Item A"

        # 2. Brands Feather snapshot
        resp_brands = get_brands_feather_snapshot("market")
        assert resp_brands.media_type == "application/octet-stream"
        df_brands = feather.read_feather(io.BytesIO(resp_brands.body))
        assert len(df_brands) == 1
        assert df_brands.iloc[0]["Brand Name"] == "Brand X"

        # 3. Dictionaries JSON snapshot
        dicts = get_dictionaries_snapshot("market")
        assert dicts == mock_dicts


def test_engine_fetches_snapshot_from_backend(tmp_path):
    """Verify engine _fetch_snapshot_from_backend downloads snapshots into its local CACHE_DIR."""
    import io
    import os

    import pandas as pd
    import pyarrow.feather as feather

    from engine import config
    from engine.core.resource_loader import _fetch_snapshot_from_backend

    orig_cache = config.CACHE_DIR
    try:
        config.CACHE_DIR = str(tmp_path)
        cat_df = pd.DataFrame([{"Name": "Food Item 1", "Price": 250.0}])
        brands_df = pd.DataFrame([{"Flavor Name": "Spicy"}])

        buf_cat = io.BytesIO()
        feather.write_feather(cat_df, buf_cat)

        buf_brands = io.BytesIO()
        feather.write_feather(brands_df, buf_brands)

        mock_cat_resp = MagicMock(status_code=200, content=buf_cat.getvalue())
        mock_brands_resp = MagicMock(status_code=200, content=buf_brands.getvalue())
        mock_dicts_resp = MagicMock(
            status_code=200, text='{"gk": ["spice"], "bt": ["curry"], "region": ["south"]}'
        )

        def mock_get(url, **kwargs):
            if "catalog.feather" in url:
                return mock_cat_resp
            if "brands.feather" in url:
                return mock_brands_resp
            if "dictionaries.json" in url:
                return mock_dicts_resp
            return MagicMock(status_code=404)

        with patch("requests.get", side_effect=mock_get):
            success = _fetch_snapshot_from_backend("food")
            assert success is True

            cat_out = os.path.join(str(tmp_path), "food_catalog_mmap.feather")
            brands_out = os.path.join(str(tmp_path), "food_brands_mmap.feather")
            dicts_out = os.path.join(str(tmp_path), "food_classifier_dicts.json")

            assert os.path.exists(cat_out)
            assert os.path.exists(brands_out)
            assert os.path.exists(dicts_out)

            read_cat = feather.read_feather(cat_out)
            assert read_cat.iloc[0]["Name"] == "Food Item 1"
    finally:
        config.CACHE_DIR = orig_cache

