import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import text

from backend.app.db.session import get_engine
from backend.app.services.catalog_service import (
    get_bt_gk_cache,
    get_catalog_and_brands,
    get_classifier_dicts,
)
from engine import config
from engine.data_pipeline.ingestion import DataIngestion
from engine.rules_engine.evaluator import _load_flavor_data, clear_flavor_cache

_CATALOG_TABLES = "catalog_items, brand_flavors, classifier_dictionaries, bt_gk_map"


@pytest.mark.usefixtures("pg")
class TestCacheFallbackIsolation(unittest.TestCase):
    """
    Verifies the strict Engine vs Backend storage boundary:
      - Engine (`DataIngestion`, `evaluator`) reads strictly from in-memory and Feather caches,
        never querying PostgreSQL or Google Sheets when `force_fetch=False`.
      - Backend (`catalog_service`) falls back to PostgreSQL when local cache files are absent,
        never touching Google Sheets over the network.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_cache_dir = config.CACHE_DIR
        self.orig_staging_dir = config.STAGING_DIR

        config.CACHE_DIR = self.test_dir
        config.STAGING_DIR = os.path.join(self.test_dir, "staged")

        with get_engine().begin() as conn:
            conn.execute(text(f"TRUNCATE {_CATALOG_TABLES} RESTART IDENTITY"))

            # Seed minimal catalog and brand records for 'market' domain in PostgreSQL
            conn.execute(
                text("""
                INSERT INTO catalog_items (domain, name, brand, category, basictype, generic_keywords, price, clean_text, row_hash)
                VALUES ('market', 'Anchor Butter 200g', 'Anchor', 'Dairy', 'Butter', 'Dairy, Butter', 850.0, 'anchor butter 200g', 'hash_market_1')
                """)
            )
            conn.execute(
                text("""
                INSERT INTO brand_flavors (domain, name, aliases, is_weak, row_hash)
                VALUES ('market', 'Anchor', 'anchor,anchr', 0, 'hash_brand_1')
                """)
            )
            # Seed minimal records for 'food' domain in PostgreSQL
            conn.execute(
                text("""
                INSERT INTO catalog_items (domain, name, flavor, region, basictype, generic_keywords, price, clean_text, row_hash)
                VALUES ('food', 'Chicken Fried Rice', 'Chicken', 'Sri Lankan', 'Fried Rice', 'Rice, Fried Rice', 1200.0, 'chicken fried rice', 'hash_food_1')
                """)
            )
            conn.execute(
                text("""
                INSERT INTO brand_flavors (domain, name, aliases, is_meat, is_vegetable, is_seafood, row_hash)
                VALUES ('food', 'Chicken', 'chk,chick', 1, 0, 0, 'hash_flavor_1')
                """)
            )
            conn.execute(
                text("""
                INSERT INTO classifier_dictionaries (domain, tag_type, tag, catalog_count)
                VALUES ('market', 'bt', 'Butter', 1), ('market', 'gk', 'Dairy', 1)
                """)
            )
            conn.execute(
                text("""
                INSERT INTO bt_gk_map (domain, basictype, generic_keywords, catalog_count, gk_count)
                VALUES ('market', 'Butter', 'Dairy, Butter', 1, 2)
                """)
            )

        DataIngestion.clear_mem_cache()
        clear_flavor_cache()

    def tearDown(self):
        DataIngestion.clear_mem_cache()
        clear_flavor_cache()
        with get_engine().begin() as conn:
            conn.execute(text(f"TRUNCATE {_CATALOG_TABLES} RESTART IDENTITY"))
        config.CACHE_DIR = self.orig_cache_dir
        config.STAGING_DIR = self.orig_staging_dir
        import gc

        gc.collect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("requests.get")
    def test_backend_get_catalog_and_brands_uses_database_without_network(self, mock_get):
        """Backend catalog_service falls back to PostgreSQL without calling Google Sheets."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        cat_df, brands_df = get_catalog_and_brands(config.DOMAIN_MARKET)

        self.assertFalse(
            mock_get.called, "requests.get should NEVER be called when database cache is hot!"
        )
        self.assertEqual(len(cat_df), 1)
        self.assertEqual(len(brands_df), 1)
        self.assertEqual(cat_df.iloc[0]["Name"], "Anchor Butter 200g")
        self.assertEqual(brands_df.iloc[0]["Brand Name"], "Anchor")

    @patch("requests.get")
    def test_engine_load_catalog_never_queries_database_when_feather_missing(self, mock_get):
        """Engine DataIngestion raises RuntimeError on Feather miss even when PostgreSQL has data."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        with self.assertRaises(RuntimeError) as ctx:
            DataIngestion.load_catalog(
                "fake_sheet_id", domain=config.DOMAIN_MARKET, force_fetch=False
            )

        self.assertIn("No catalog data found in memory or Feather cache", str(ctx.exception))
        self.assertFalse(
            mock_get.called, "requests.get must NOT be called when force_fetch=False on cache miss!"
        )

    @patch("requests.get")
    def test_load_catalog_uses_feather_without_network(self, mock_get):
        """Engine DataIngestion loads from Feather files when present without touching network."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

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
    def test_force_fetch_calls_network(self, mock_get):
        """Verify that force_fetch=True downloads fresh CSVs and rebuilds Feather caches."""
        cat_resp = MagicMock()
        cat_resp.status_code = 200
        cat_resp.text = (
            "Name,Price,Brand,Category,BasicType,GenericKeywords\n"
            "Anchor Butter 200g,850.0,Anchor,Dairy,Butter,Dairy\n"
        )
        cat_resp.content = cat_resp.text.encode("utf-8")

        brands_resp = MagicMock()
        brands_resp.status_code = 200
        brands_resp.text = "Brand Name,Aliases,Is_Weak\nAnchor,anchor,0\n"
        brands_resp.content = brands_resp.text.encode("utf-8")

        mock_get.side_effect = [cat_resp, brands_resp]

        cat_df, brands_df = DataIngestion.load_catalog(
            "fake_sheet_id", domain=config.DOMAIN_MARKET, force_fetch=True
        )

        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(cat_df.iloc[0]["Name"], "Anchor Butter 200g")
        self.assertEqual(brands_df.iloc[0]["Brand Name"], "Anchor")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "market_catalog_mmap.feather")))

    @patch("requests.get")
    def test_get_classifier_dicts_uses_database_without_network(self, mock_get):
        """Backend get_classifier_dicts reads from PostgreSQL without calling Google Sheets."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        dicts = get_classifier_dicts(config.DOMAIN_MARKET)

        self.assertFalse(mock_get.called)
        self.assertIn("bt", dicts)
        self.assertIn("Butter", dicts["bt"])
        self.assertIn("Dairy", dicts["gk"])
        self.assertEqual(os.listdir(self.test_dir), [], "Backend must not write to CACHE_DIR")

    @patch("requests.get")
    def test_flavor_rules_cache_evaluates_from_feather_without_network(self, mock_get):
        """Engine _load_flavor_data reads food_brands_mmap.feather without network or DB."""
        mock_get.side_effect = ConnectionError("Network is disconnected!")

        flavors_feather = os.path.join(self.test_dir, "food_brands_mmap.feather")
        flavors_df = pd.DataFrame(
            [
                {
                    "Flavor Name": "Chicken",
                    "Aliases": "chk,chick",
                    "Is_Meat": True,
                    "Is_Vegetable": False,
                    "Is_Seafood": False,
                }
            ]
        )
        flavors_df.to_feather(flavors_feather)

        flavor_cache = _load_flavor_data()

        self.assertFalse(mock_get.called)
        self.assertIn("flavors_dict", flavor_cache)
        self.assertIn("chicken", flavor_cache["meat_flavors"])

    def test_get_bt_gk_cache_reconstructs_from_database(self):
        """Backend get_bt_gk_cache reconstructs bt_gk_map from PostgreSQL when .pkl is absent."""
        bt_gk_data = get_bt_gk_cache(config.DOMAIN_MARKET)
        self.assertIn("bt_gk_map", bt_gk_data)
        self.assertIn("Butter", bt_gk_data["bt_gk_map"])
        self.assertIn("Dairy", bt_gk_data["bt_gk_map"]["Butter"])


if __name__ == "__main__":
    unittest.main()
