import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from engine.nlp.text_cleaner import UnitUtils
import engine.rules_engine.evaluator as eval_mod
from engine.rules_engine.evaluator import _extract_flavors_from_sku
from engine.classification.classifier import ZeroShotClassifier
from engine.classification.tagger import tag_all_skus, _resolve_flavors_from_text


class TestPerformanceOptimizations(unittest.TestCase):
    """Tests to verify correctness of performance optimization routines."""

    def test_fast_path_weight_extraction(self):
        # Test standard unit patterns handled by the fast-path regex
        test_cases = [
            ("Anchor Butter 250g", (250.0, "g", "solid")),
            ("Fresh Milk 1L", (1000.0, "ml", "liquid")),
            ("Olive Oil 500ml", (500.0, "ml", "liquid")),
            ("Basmati Rice 5kg", (5000.0, "g", "solid")),
            ("Sugar 1.5kg", (1500.0, "g", "solid")),
            ("Yogurt 100 g", (100.0, "g", "solid")),
            ("Vanilla Essence 50 ml", (50.0, "ml", "liquid")),
            ("Cream 25cl", (250.0, "ml", "liquid")),
            ("Salt 750mg", (0.75, "g", "solid")),
        ]
        for text, expected in test_cases:
            res = UnitUtils.get_normalized_weight(text)
            self.assertIsNotNone(res, f"Failed for {text}")
            self.assertEqual(res[0], expected[0], f"Value mismatch for {text}")
            self.assertEqual(res[1], expected[1], f"Unit mismatch for {text}")
            self.assertEqual(res[2], expected[2], f"Form mismatch for {text}")

    def test_precompiled_flavor_extraction(self):
        eval_mod._FLAVOR_CACHE = {
            "flavors_dict": {
                "chicken": "chicken",
                "chicken breast": "chicken",
                "black pepper": "black pepper",
                "pepper": "pepper",
                "spinach": "spinach",
                "prawn": "prawn",
                "shrimp": "prawn"
            },
            "meat_flavors": {"chicken"},
            "vegetable_flavors": {"spinach"},
            "seafood_flavors": {"prawn"}
        }
        eval_mod._extract_flavors_from_sku_cached.cache_clear()

        # 1. Single term
        flavors = _extract_flavors_from_sku("Crispy Chicken Bites")
        self.assertIn("chicken", flavors)

        # 2. Multi-word alias takes precedence over single word substring
        flavors = _extract_flavors_from_sku("Grilled Chicken Breast with Black Pepper")
        self.assertIn("chicken", flavors)
        self.assertIn("black pepper", flavors)

        # 3. Alias mapped to canonical
        flavors = _extract_flavors_from_sku("Fresh Shrimp Cocktail")
        self.assertIn("prawn", flavors)

        # 4. No flavor
        flavors = _extract_flavors_from_sku("Plain Plain White Plate")
        self.assertEqual(len(flavors), 0)

    def test_tagger_resolve_flavors_from_text(self):
        flavors_dict = {
            "chicken": "chicken",
            "beef": "beef",
            "chili": "chili"
        }
        res = _resolve_flavors_from_text("Spicy Chili Chicken Wings", flavors_dict)
        self.assertEqual(res, {"chili", "chicken"})

        res_empty = _resolve_flavors_from_text("", flavors_dict)
        self.assertEqual(res_empty, set())

    def test_batch_predict_third_tag_and_gk(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense": np.zeros((2, 1024), dtype=np.float32)}

        descriptions = {
            "bt_descriptions": {"Noodles": "wheat noodles", "Rice": "steamed rice"},
            "third_tag_descriptions": {"Western": "western food", "Asian": "asian cuisine"},
            "third_tag_overrides": {"Noodles": "Asian"},
            "bt_to_gk_umbrella": {"Noodles": ["Pasta", "Noodles"]},
            "bt_gk_map": {"Noodles": ["Wheat", "Instant Noodles"]}
        }

        clf = ZeroShotClassifier(mock_model, domain="food", descriptions=descriptions)
        clf._trained = False # Zero-shot mode

        vecs = np.random.randn(2, 1024).astype(np.float32)
        prices = [500.0, 1200.0]

        # Batch predict GK
        gk_res = clf.batch_predict_gk(vecs, prices)
        self.assertEqual(len(gk_res), 2)
        self.assertEqual(gk_res[0][2], "zero-shot")

        # Batch predict Third Tag (1 with BT override, 1 with zero-shot)
        third_res = clf.batch_predict_third_tag(
            vecs,
            names=["Mie Goreng", "Steamed Basmati"],
            predicted_bts=["Noodles", "Rice"],
            prices=prices
        )
        self.assertEqual(len(third_res), 2)
        self.assertEqual(third_res[0], ("Asian", 1.0, "override"))
        self.assertEqual(third_res[1][2], "zero-shot")


if __name__ == '__main__':
    unittest.main()
