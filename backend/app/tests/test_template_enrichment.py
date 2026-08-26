import unittest
from unittest.mock import patch, MagicMock
from engine import config
from engine.template_suggest import suggest_tags_from_template

class TestTemplateEnrichmentToggles(unittest.TestCase):
    def setUp(self):
        # Save original config flags
        self.orig_enrichment = getattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", True)
        self.orig_allow_unreg = getattr(config, "ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS", True)

    def tearDown(self):
        # Restore config flags
        config.ENABLE_TEMPLATE_TAG_ENRICHMENT = self.orig_enrichment
        config.ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = self.orig_allow_unreg

    @patch('engine.template_suggest.get_catalog_and_brands')
    @patch('engine.template_suggest.get_classifier')
    @patch('engine.template_suggest.get_classifier_dicts')
    def test_allow_unregistered_toggle_true(self, mock_get_dicts, mock_get_classifier, mock_get_catalog):
        import pandas as pd
        mock_cat_df = pd.DataFrame([{
            "Name": "Alerics Chocolate Ice Cream 1l",
            "basictype": "Ice Cream Tub",
            "Generic keywords": "Dairy, Ice Cream, Ice Cream Tub, Alerics Ice Cream, Chocolate Ice Cream, Alerics Ice Cream Tub, Chocolate Ice Cream Tub",
            "Flavor": "chocolate",
            "Brand": "Alerics"
        }])
        mock_brands_df = pd.DataFrame()
        mock_get_catalog.return_value = (mock_cat_df, mock_brands_df)

        mock_ner = MagicMock()
        mock_ner.extract_entities.return_value = {"flavor": {"chicken"}}
        mock_ner._get_dict_entities.return_value = (set(), set())
        
        mock_clf = MagicMock()
        mock_clf.ner_engine = mock_ner
        mock_get_classifier.return_value = mock_clf
        
        mock_get_dicts.return_value = {
            "bt": ["Ice Cream Tub"],
            "gk": ["Dairy", "Ice Cream", "Ice Cream Tub", "Alerics Ice Cream"]
        }

        # Case 1: ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = True
        config.ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = True
        res_true = suggest_tags_from_template("Alerics Chicken Ice Cream 1l", "food")
        
        self.assertTrue(res_true.get("matched"))
        gks_true = res_true.get("suggested_gk", [])
        # "Chicken Ice Cream" should be present as new_unregistered tag
        self.assertIn("Chicken Ice Cream", gks_true)

    @patch('engine.template_suggest.get_catalog_and_brands')
    @patch('engine.template_suggest.get_classifier')
    @patch('engine.template_suggest.get_classifier_dicts')
    def test_allow_unregistered_toggle_false(self, mock_get_dicts, mock_get_classifier, mock_get_catalog):
        import pandas as pd
        mock_cat_df = pd.DataFrame([{
            "Name": "Alerics Chocolate Ice Cream 1l",
            "basictype": "Ice Cream Tub",
            "Generic keywords": "Dairy, Ice Cream, Ice Cream Tub, Alerics Ice Cream, Chocolate Ice Cream, Alerics Ice Cream Tub, Chocolate Ice Cream Tub",
            "Flavor": "chocolate",
            "Brand": "Alerics"
        }])
        mock_brands_df = pd.DataFrame()
        mock_get_catalog.return_value = (mock_cat_df, mock_brands_df)

        mock_ner = MagicMock()
        mock_ner.extract_entities.return_value = {"flavor": {"chicken"}}
        mock_ner._get_dict_entities.return_value = (set(), set())
        
        mock_clf = MagicMock()
        mock_clf.ner_engine = mock_ner
        mock_get_classifier.return_value = mock_clf
        
        mock_get_dicts.return_value = {
            "bt": ["Ice Cream Tub"],
            "gk": ["Dairy", "Ice Cream", "Ice Cream Tub", "Alerics Ice Cream"]
        }

        # Case 2: ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = False
        config.ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = False
        res_false = suggest_tags_from_template("Alerics Chicken Ice Cream 1l", "food")
        
        self.assertTrue(res_false.get("matched"))
        gks_false = res_false.get("suggested_gk", [])
        # "Chicken Ice Cream" should be FILTERED OUT because it is unregistered
        self.assertNotIn("Chicken Ice Cream", gks_false)
        self.assertIn("Dairy", gks_false)
        self.assertIn("Ice Cream Tub", gks_false)

if __name__ == "__main__":
    unittest.main()
