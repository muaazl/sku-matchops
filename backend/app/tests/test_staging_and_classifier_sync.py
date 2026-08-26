import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json
import shutil
import pandas as pd
import numpy as np
import joblib

from engine import config
from engine.data_pipeline.ingestion import DataIngestion
from engine.classification.classifier import ZeroShotClassifier


class TestStagingAndClassifierSync(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.orig_staging_dir = config.STAGING_DIR
        self.orig_cache_dir = config.CACHE_DIR
        config.STAGING_DIR = os.path.join(self.test_dir, "staged_sheets")
        config.CACHE_DIR = self.test_dir

    def tearDown(self):
        config.STAGING_DIR = self.orig_staging_dir
        config.CACHE_DIR = self.orig_cache_dir
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_required_sheets_for_domain(self):
        market_sheets = DataIngestion.get_required_sheets_for_domain(config.DOMAIN_MARKET)
        self.assertIn("catalog", market_sheets)
        self.assertIn("brands_or_flavors", market_sheets)
        self.assertEqual(market_sheets["brands_or_flavors"], "Market_Brands")
        self.assertEqual(market_sheets["catalog"], config.MARKET_CATALOG_SHEET)

        food_sheets = DataIngestion.get_required_sheets_for_domain(config.DOMAIN_FOOD)
        self.assertIn("catalog", food_sheets)
        self.assertIn("brands_or_flavors", food_sheets)
        self.assertEqual(food_sheets["brands_or_flavors"], "Food_Flavors")
        self.assertEqual(food_sheets["catalog"], config.FOOD_CATALOG_SHEET)

    @patch("engine.data_pipeline.ingestion.requests.get")
    def test_stage_all_sheets_and_cleanup(self, mock_get):
        # Mock requests response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Name,basictype,Price\nTest Item,Test BT,10.0\n"
        mock_get.return_value = mock_response

        # Stage sheets for market domain
        staged = DataIngestion.stage_all_sheets("test_sheet_id", domains=[config.DOMAIN_MARKET])
        self.assertTrue(len(staged) > 0)
        
        # Verify files exist in STAGING_DIR
        for sheet_name, path in staged.items():
            self.assertTrue(os.path.exists(path))
            self.assertTrue(DataIngestion.is_sheet_staged(sheet_name))

        manifest_path = os.path.join(config.STAGING_DIR, "staging_manifest.json")
        self.assertTrue(os.path.exists(manifest_path))

        # Test _fetch_sheet_as_csv uses staged file
        with patch.object(DataIngestion, "_download_sheet_with_retries") as mock_download:
            df = DataIngestion._fetch_sheet_as_csv("test_sheet_id", config.MARKET_CATALOG_SHEET)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["Name"], "Test Item")
            # Should NOT call download since it is staged
            mock_download.assert_not_called()

        # Test cleanup
        DataIngestion.cleanup_staged_sheets()
        for sheet_name, path in staged.items():
            self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(manifest_path))

    @patch("engine.data_pipeline.ingestion.requests.get")
    def test_download_sheet_with_retries_resilience(self, mock_get):
        # Fail twice with connection error, succeed on 3rd attempt
        fail_response = Exception("Wi-Fi disconnected")
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.text = "col1,col2\nval1,val2\n"

        mock_get.side_effect = [fail_response, fail_response, success_response]

        with patch("time.sleep") as mock_sleep:
            text = DataIngestion._download_sheet_with_retries(
                "test_sheet", "Test_Tab", max_retries=3, backoff_factor=1.0
            )
            self.assertEqual(text, "col1,col2\nval1,val2\n")
            self.assertEqual(mock_get.call_count, 3)

    def test_classifier_stored_hashes_no_name_error(self):
        # Create a sample cat_df with all required columns
        sample_data = {
            "Name": [f"Item {i}" for i in range(20)],
            "basictype": [f"BT_{i % 3}" for i in range(20)],
            "category": [f"Cat_{i % 2}" for i in range(20)],
            "Generic keywords": [f"kw_{i}, kw_common" for i in range(20)],
            "Price": [10.0 + i for i in range(20)],
            "Description": [f"Desc {i}" for i in range(20)],
        }
        df = pd.DataFrame(sample_data)

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense": np.random.randn(len(df), 1024).astype(np.float32)}
        mock_model.embed_weighted_sku.return_value = {"dense": np.random.randn(len(df), 1024).astype(np.float32)}

        # Initialize classifier with mock model
        clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
        clf.domain = config.DOMAIN_MARKET
        clf.cache_dir = self.test_dir
        clf.model = mock_model
        clf.cat_df = df
        clf.bt_descs = {"BT_0": "desc 0", "BT_1": "desc 1", "BT_2": "desc 2"}
        clf.third_tag_descs = {"Cat_0": "desc 0", "Cat_1": "desc 1"}
        clf.third_tag_overrides = {}
        clf.bt_to_gk_umbrella = {}
        clf._trained = False

        # Run _try_train
        clf._try_train(force_retrain=True)

        # Verify training succeeded and stored_hashes file was written without NameError
        hash_file = os.path.join(config.CACHE_DIR, "model_hashes.json")
        self.assertTrue(os.path.exists(hash_file))
        with open(hash_file, "r") as f:
            hashes = json.load(f)
        self.assertIn("market_training_state", hashes)
        self.assertTrue(len(hashes["market_training_state"]) > 0)
        self.assertTrue(clf._trained)

    @patch("scripts.sync_catalog.DataIngestion.load_catalog")
    @patch("scripts.sync_catalog.DataIngestion.load_classifier_dictionaries")
    def test_rebuild_disk_caches_safe_token_count(self, mock_load_dicts, mock_load_cat):
        from scripts.sync_catalog import rebuild_disk_caches
        
        # Test catalog with None/float/missing clean_text
        sample_df = pd.DataFrame({
            "Name": ["Coca Cola 250ml", "Pepsi Max 500ml", "Sprite 1l"],
            "Price": [10.0, 15.0, 20.0],
            "clean_text": [None, np.nan, "sprite 1l"]
        })
        brands_df = pd.DataFrame({"Brand Name": ["coca cola", "pepsi"], "Aliases": ["", ""]})
        
        mock_load_cat.return_value = (sample_df, brands_df)
        mock_load_dicts.return_value = {"gk": ["soda"], "bt": ["beverage"], "category": ["drinks"]}
        
        # Should not raise AttributeError: Can only use .str accessor with string values, not floating
        rebuild_disk_caches("market", "test_sheet_id")
        
        metadata_path = os.path.join(config.CACHE_DIR, "market_catalog_metadata.pkl")
        self.assertTrue(os.path.exists(metadata_path))
        meta_df = joblib.load(metadata_path)
        self.assertIn("token_count", meta_df.columns)
        self.assertEqual(meta_df.loc[0, "token_count"], 3)  # coca cola 250ml -> 3 tokens


if __name__ == "__main__":
    unittest.main()

