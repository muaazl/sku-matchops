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
