import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the functions to test
from engine.template_suggest import title_case_with_exceptions, suggest_tags_from_template
from engine.nlp.ner_engine import NEREngine
from engine import config

class TestTemplateSuggest(unittest.TestCase):

    def setUp(self):
        self.orig_allow_unreg = getattr(config, "ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS", True)
        config.ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = True

    def tearDown(self):
        config.ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = self.orig_allow_unreg

    def test_title_case_with_exceptions(self):
        cases = [
            ("ice cream tub with chocolate", "Ice Cream Tub with Chocolate"),
            ("chicken fried rice of beef", "Chicken Fried Rice of Beef"),
            ("non-dairy ice-cream and milk", "Non-Dairy Ice-Cream and Milk"),
            ("with milk and sugar", "With Milk and Sugar"),  # First word is capitalized even if minor
        ]
        for inp, expected in cases:
            self.assertEqual(title_case_with_exceptions(inp), expected)

    @patch('engine.resource_loader.get_pipeline')
    @patch('engine.template_suggest.get_catalog_and_brands')
    @patch('engine.template_suggest.get_classifier')
    @patch('engine.template_suggest.get_classifier_dicts')
    def test_suggest_tags_from_template_food(self, mock_get_dicts, mock_get_classifier, mock_get_catalog, mock_get_pipeline):
        mock_get_pipeline.side_effect = Exception("Mock pipeline disabled")
        # 1. Mock Catalog DataFrame
        catalog_data = [
            {
                "Name": "Alerics Chocolate Ice Cream 1l", 
                "basictype": "Ice Cream Tub", 
                "Generic keywords": "Dairy, Ice Cream, Ice Cream Tub, Alerics Ice Cream, Chocolate Ice Cream, Alerics Ice Cream Tub, Chocolate Ice Cream Tub, Alerics Chocolate Ice Cream, Alerics Chocolate Ice Cream Tub"
            },
            {
                "Name": "Chicken Fried Rice", 
                "basictype": "Fried Rice", 
                "Generic keywords": "Fried Rice, Chicken Fried Rice"
            },
            {
                "Name": "Black Pepper Chicken", 
                "basictype": "Black Pepper Chicken", 
                "Generic keywords": "Chicken, Black Pepper, Black Pepper Chicken"
            },
            {
                "Name": "Cuttlefish Fried Rice",
                "basictype": "Fried Rice",
                "Generic keywords": "Fried Rice"
            },
            {
                "Name": "Lemonade",
                "basictype": "Lemonade",
                "Flavor": "lemonade",
                "Generic keywords": "Beverage, Soft Drinks"
            }
        ]
        mock_cat_df = pd.DataFrame(catalog_data)
        
        # Mock Brands/Flavors DataFrame
        brands_data = [
            {"Flavor Name": "chocolate", "Aliases": "choco"},
            {"Flavor Name": "vanilla", "Aliases": "vanil"},
            {"Flavor Name": "chicken", "Aliases": "chix"},
            {"Flavor Name": "beef", "Aliases": "bov"},
            {"Flavor Name": "cuttlefish", "Aliases": ""},
            {"Flavor Name": "lemonade", "Aliases": ""},
        ]
        mock_brands_df = pd.DataFrame(brands_data)
        
        mock_get_catalog.return_value = (mock_cat_df, mock_brands_df)
        
        # 2. Mock NEREngine and Classifier
        real_ner = NEREngine(mock_brands_df, domain="food")
        real_ner.model = None
        
        mock_classifier = MagicMock()
        mock_classifier.ner_engine = real_ner
        mock_get_classifier.return_value = mock_classifier
        
        # 3. Mock Dictionary
        mock_dicts = {
            "gk": [
                "Dairy", "Ice Cream", "Ice Cream Tub", "Alerics Ice Cream", 
                "Vanilla Ice Cream", "Alerics Ice Cream Tub", "Vanilla Ice Cream Tub", 
                "Fried Rice", "Chicken Fried Rice"
            ],
            "bt": ["Ice Cream Tub", "Fried Rice"]
        }
        mock_get_dicts.return_value = mock_dicts
        
        # --- TEST CASE 1: Alerics Vanilla Ice Cream 1l ---
        res = suggest_tags_from_template("Alerics Vanilla Ice Cream 1l", "food")
        self.assertTrue(res["matched"])
        self.assertEqual(res["base_sku"], "Alerics Chocolate Ice Cream 1l")
        self.assertEqual(res["base_entity"], "chocolate")
        self.assertEqual(res["new_entity"], "vanilla")
        self.assertEqual(res["suggested_bt"], "Ice Cream Tub")
        # Check snapped vs unregistered tags
        gks = res["suggested_gk"]
        self.assertIn("Vanilla Ice Cream", gks)  # Exact match in dict
        self.assertIn("Vanilla Ice Cream Tub", gks)  # Exact match in dict
        self.assertIn("Alerics Vanilla Ice Cream", gks)  # New / unregistered (should be formatted Title Case)
        
        # Check gk_info details for snapping
        gk_info_map = {item["original"].lower(): item for item in res["gk_info"]}
        self.assertEqual(gk_info_map["alerics vanilla ice cream"]["status"], "new_unregistered")
        self.assertEqual(gk_info_map["alerics vanilla ice cream"]["suggested"], "Alerics Vanilla Ice Cream")
        self.assertEqual(gk_info_map["vanilla ice cream"]["status"], "exact_dictionary_match")
        
        # --- TEST CASE 1b: Exact match skipped as self-candidate ---
        res_exact = suggest_tags_from_template("Alerics Chocolate Ice Cream 1l", "food")
        self.assertFalse(res_exact["matched"])
        self.assertIn("No template match found in catalog", res_exact["reason"])
        
        # --- TEST CASE 2: Beef Fried Rice ---
        res_rice = suggest_tags_from_template("Beef Fried Rice", "food")
        self.assertTrue(res_rice["matched"])
        self.assertEqual(res_rice["base_sku"], "Chicken Fried Rice")
        self.assertEqual(res_rice["suggested_bt"], "Fried Rice")
        self.assertIn("Beef Fried Rice", res_rice["suggested_gk"])
        
        # --- TEST CASE 3: Black Pepper Beef ---
        res_pepper = suggest_tags_from_template("Black Pepper Beef", "food")
        self.assertTrue(res_pepper["matched"])
        self.assertEqual(res_pepper["base_sku"], "Black Pepper Chicken")
        # BT is swapped to Black Pepper Beef (not in mock BT dict)
        self.assertEqual(res_pepper["suggested_bt"], "Black Pepper Beef")
        self.assertEqual(res_pepper["bt_info"]["status"], "new_unregistered")
        self.assertIn("Black Pepper Beef", res_pepper["suggested_gk"])
        
        # --- TEST CASE 4: Cuttlefish Fried Rice ---
        res_cuttlefish = suggest_tags_from_template("Cuttlefish Fried Rice", "food")
        self.assertTrue(res_cuttlefish["matched"])
        self.assertEqual(res_cuttlefish["base_sku"], "Chicken Fried Rice")
        self.assertEqual(res_cuttlefish["suggested_bt"], "Fried Rice")
        self.assertIn("Cuttlefish Fried Rice", res_cuttlefish["suggested_gk"])

        # --- TEST CASE 5 (REGRESSION): SEENISAMBOL & CHEESE SANDWICH ---
        # Should NOT match single-word "Lemonade" or any unrelated beverage item!
        res_sandwich = suggest_tags_from_template("SEENISAMBOL & CHEESE SANDWICH", "food", current_bt="Sandwiches")
        self.assertFalse(res_sandwich["matched"])

    @patch('engine.resource_loader.get_pipeline')
    @patch('engine.template_suggest.get_catalog_and_brands')
    @patch('engine.template_suggest.get_classifier')
    @patch('engine.template_suggest.get_classifier_dicts')
    def test_suggest_tags_from_template_market(self, mock_get_dicts, mock_get_classifier, mock_get_catalog, mock_get_pipeline):
        mock_get_pipeline.side_effect = Exception("Mock pipeline disabled")
        # 1. Mock Catalog DataFrame for market
        catalog_data = [
            {
                "Name": "Lux Chocolate Soap", 
                "basictype": "Soap", 
                "Generic keywords": "Soap, Lux Soap, Chocolate Soap"
            },
            {
                "Name": "Alerics Chocolate Ice Cream 1l",
                "basictype": "Ice Cream Tub", 
                "Generic keywords": "Dairy, Ice Cream, Ice Cream Tub, Alerics Ice Cream, Chocolate Ice Cream, Alerics Ice Cream Tub, Chocolate Ice Cream Tub, Alerics Chocolate Ice Cream, Alerics Chocolate Ice Cream Tub"
            }
        ]
        mock_cat_df = pd.DataFrame(catalog_data)
        
        # Brands: Lux is in market brand list
        brands_data = [
            {"Brand Name": "lux", "Aliases": ""},
            {"Brand Name": "alerics", "Aliases": ""},
        ]
        mock_brands_df = pd.DataFrame(brands_data)
        
        # Flavor brands list: chocolate and lemon are in food flavors list
        food_flavors_data = [
            {"Flavor Name": "chocolate", "Aliases": ""},
            {"Flavor Name": "lemon", "Aliases": ""},
            {"Flavor Name": "blueberry", "Aliases": ""},
        ]
        mock_food_flavors_df = pd.DataFrame(food_flavors_data)
        
        def side_effect(domain):
            if domain == "market":
                return mock_cat_df, mock_brands_df
            else:
                return pd.DataFrame(), mock_food_flavors_df
        mock_get_catalog.side_effect = side_effect
        
        # 2. Mock NEREngine and Classifier
        market_ner = NEREngine(mock_brands_df, domain="market")
        market_ner.model = None
        
        food_ner = NEREngine(mock_food_flavors_df, domain="food")
        food_ner.model = None
        
        def mock_get_clf(domain):
            mock_clf = MagicMock()
            if domain == "market":
                mock_clf.ner_engine = market_ner
            else:
                mock_clf.ner_engine = food_ner
            return mock_clf
        mock_get_classifier.side_effect = mock_get_clf
        
        # 3. Mock Dictionary
        mock_dicts = {
            "gk": ["Soap", "Lux Soap", "Lemon Soap", "Dairy", "Ice Cream", "Ice Cream Tub"],
            "bt": ["Soap", "Ice Cream Tub"]
        }
        mock_get_dicts.return_value = mock_dicts
        
        # Run test: suggest tags for Lux Lemon Soap (lemon flavor should be detected and swapped in market domain)
        res = suggest_tags_from_template("Lux Lemon Soap", "market")
        self.assertTrue(res["matched"])
        self.assertEqual(res["base_sku"], "Lux Chocolate Soap")
        self.assertEqual(res["base_entity"], "chocolate")
        self.assertEqual(res["new_entity"], "lemon")
        self.assertEqual(res["entity_type"], "flavor")
        self.assertEqual(res["suggested_bt"], "Soap")
        self.assertIn("Lemon Soap", res["suggested_gk"])
        
        # Run test: suggest tags for Alerics Blueberry Ice Cream 1l (blueberry flavor fallback and Alerics brand in market domain)
        res_blueberry = suggest_tags_from_template("Alerics Blueberry Ice Cream 1l", "market")
        self.assertTrue(res_blueberry["matched"])
        self.assertEqual(res_blueberry["base_sku"], "Alerics Chocolate Ice Cream 1l")
        self.assertEqual(res_blueberry["base_entity"], "chocolate")
        self.assertEqual(res_blueberry["new_entity"], "blueberry")
        self.assertEqual(res_blueberry["entity_type"], "flavor")
        self.assertEqual(res_blueberry["suggested_bt"], "Ice Cream Tub")
        self.assertIn("Alerics Blueberry Ice Cream", res_blueberry["suggested_gk"])

if __name__ == '__main__':
    unittest.main()

