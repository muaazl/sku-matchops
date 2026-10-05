import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from backend.app.services.catalog_service import (
    get_bt_gk_cache,
    get_classifier_dicts,
)
from engine import config
from engine.core.db import ensure_db_initialized
from engine.data_pipeline.ingestion import DataIngestion
from engine.rules_engine.evaluator import _load_flavor_data, clear_flavor_cache


class TestCacheFallbackIsolation(unittest.TestCase):
    """
    Verifies that runtime catalog, dictionary, and rules queries strictly use local
    in-memory, Feather, and SQLite caches without attempting any network HTTP calls
    to Google Sheets.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_cache_dir = config.CACHE_DIR
        self.orig_staging_dir = config.STAGING_DIR
        self.orig_db_path = config.DB_PATH

        config.CACHE_DIR = self.test_dir
        config.STAGING_DIR = os.path.join(self.test_dir, "staged")
        config.DB_PATH = os.path.join(self.test_dir, "test_matchops.db")

        # Initialize SQLite DB
        conn = ensure_db_initialized(config.DB_PATH)

        # Seed minimal SQLite catalog and brand records for 'market' domain
        conn.execute(
            """
            INSERT INTO catalog_items (domain, name, brand, category, basictype, generic_keywords, price, clean_text, row_hash)
            VALUES ('market', 'Anchor Butter 200g', 'Anchor', 'Dairy', 'Butter', 'Dairy, Butter', 850.0, 'anchor butter 200g', 'hash_market_1')
            """
        )
        conn.execute(
            """
            INSERT INTO brand_flavors (domain, name, aliases, is_weak, row_hash)
            VALUES ('market', 'Anchor', 'anchor,anchr', 0, 'hash_brand_1')
            """
        )
        # Seed minimal SQLite records for 'food' domain
        conn.execute(
            """
            INSERT INTO catalog_items (domain, name, flavor, region, basictype, generic_keywords, price, clean_text, row_hash)
            VALUES ('food', 'Chicken Fried Rice', 'Chicken', 'Sri Lankan', 'Fried Rice', 'Rice, Fried Rice', 1200.0, 'chicken fried rice', 'hash_food_1')
            """
        )
        conn.execute(
            """
            INSERT INTO brand_flavors (domain, name, aliases, is_meat, is_vegetable, is_seafood, row_hash)
            VALUES ('food', 'Chicken', 'chk,chick', 1, 0, 0, 'hash_flavor_1')
            """
        )
        conn.execute(
            """
            INSERT INTO classifier_dictionaries (domain, tag_type, tag, catalog_count)
            VALUES ('market', 'bt', 'Butter', 1), ('market', 'gk', 'Dairy', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO bt_gk_map (domain, basictype, generic_keywords, catalog_count, gk_count)
            VALUES ('market', 'Butter', 'Dairy, Butter', 1, 2)
            """
        )
        conn.commit()
        conn.close()

        # Clear in-memory caches
        DataIngestion.clear_mem_cache()
        clear_flavor_cache()

    def tearDown(self):
        DataIngestion.clear_mem_cache()
        clear_flavor_cache()
        config.CACHE_DIR = self.orig_cache_dir
        config.STAGING_DIR = self.orig_staging_dir
        import gc

        gc.collect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("requests.get")
    def test_load_catalog_uses_sqlite_without_network(self, mock_get):
        """Verify load_catalog loads from SQLite without touching Google Sheets over network."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        cat_df, brands_df = DataIngestion.load_catalog(
            "fake_sheet_id", domain=config.DOMAIN_MARKET, force_fetch=False
        )

        self.assertFalse(
            mock_get.called, "requests.get should NEVER be called when SQLite cache is hot!"
        )
        self.assertEqual(len(cat_df), 1)
        self.assertEqual(len(brands_df), 1)
        self.assertEqual(cat_df.iloc[0]["Name"], "Anchor Butter 200g")
        self.assertEqual(brands_df.iloc[0]["Brand Name"], "Anchor")

    @patch("requests.get")
    def test_load_catalog_uses_feather_without_network(self, mock_get):
        """Verify load_catalog loads from Feather files when present without touching network."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        # Create Feather cache files
        cat_feather = os.path.join(self.test_dir, "market_catalog_mmap.feather")
        brands_feather = os.path.join(self.test_dir, "market_brands_mmap.feather")

        cat_sample = pd.DataFrame([{"Name": "Sample Item", "clean_text": "sample item"}])
        brands_sample = pd.DataFrame([{"Brand Name": "Sample Brand"}])
        cat_sample.to_feather(cat_feather)
        brands_sample.to_feather(brands_feather)

        DataIngestion.clear_mem_cache("market")

        cat_df, brands_df = DataIngestion.load_catalog(
            "fake_sheet_id", domain=config.DOMAIN_MARKET, force_fetch=False
        )

        self.assertFalse(mock_get.called)
        self.assertEqual(cat_df.iloc[0]["Name"], "Sample Item")
        self.assertEqual(brands_df.iloc[0]["Brand Name"], "Sample Brand")

    @patch("requests.get")
    def test_empty_cache_raises_error_without_network_when_not_force_fetch(self, mock_get):
        """Verify that an empty cache raises RuntimeError immediately without calling sheets when force_fetch=False."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        # Query an empty domain
        with self.assertRaises(RuntimeError) as ctx:
            DataIngestion.load_catalog(
                "fake_sheet_id", domain="non_existent_domain", force_fetch=False
            )

        self.assertIn(
            "No catalog data found in memory, Feather cache, or SQLite", str(ctx.exception)
        )
        self.assertFalse(
            mock_get.called, "requests.get must NOT be called when force_fetch=False on cache miss!"
        )

    @patch("requests.get")
    def test_force_fetch_calls_network(self, mock_get):
        """Verify that force_fetch=True does attempt to fetch from Google Sheets."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Name,Price\nItem 1,10.0\n"
        mock_get.return_value = mock_resp

        # force_fetch=True should trigger download
        try:
            DataIngestion.load_catalog(
                "fake_sheet_id", domain=config.DOMAIN_MARKET, force_fetch=True
            )
        except Exception:
            # We only care that requests.get was invoked
            pass

        self.assertTrue(mock_get.called, "requests.get SHOULD be called when force_fetch=True!")

    @patch("requests.get")
    def test_get_classifier_dicts_uses_sqlite_without_network(self, mock_get):
        """Verify get_classifier_dicts reads from SQLite without calling Google Sheets."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        dicts = get_classifier_dicts(config.DOMAIN_MARKET)

        self.assertFalse(mock_get.called)
        self.assertIn("bt", dicts)
        self.assertIn("Butter", dicts["bt"])
        self.assertIn("Dairy", dicts["gk"])

    @patch("requests.get")
    def test_flavor_rules_cache_evaluates_without_network(self, mock_get):
        """Verify _get_flavor_cache compiles flavor rules from SQLite without calling Google Sheets."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        flavor_cache = _load_flavor_data()

        self.assertFalse(mock_get.called)
        self.assertIn("flavors_dict", flavor_cache)
        self.assertIn("chicken", flavor_cache["meat_flavors"])

    def test_get_bt_gk_cache_reconstructs_from_sqlite(self):
        """Verify get_bt_gk_cache reconstructs bt_gk_map from SQLite when .pkl is absent."""
        bt_gk_data = get_bt_gk_cache(config.DOMAIN_MARKET)
        self.assertIn("bt_gk_map", bt_gk_data)
        self.assertIn("Butter", bt_gk_data["bt_gk_map"])
        self.assertIn("Dairy", bt_gk_data["bt_gk_map"]["Butter"])


if __name__ == "__main__":
    unittest.main()
