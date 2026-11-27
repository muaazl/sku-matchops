import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import engine.pipelines.rules_evaluation_pipeline as eval_mod
from engine.pipelines.classification_pipeline import TaxonomyClassifier
from engine.pipelines.rules_evaluation_pipeline import _extract_flavors_from_sku
from engine.preprocessing.flavor_preprocessor import extract_canonical_flavors
from engine.preprocessing.text_normalizer import extract_normalized_weight


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
            res = extract_normalized_weight(text)
            self.assertIsNotNone(res, f"Failed for {text}")
            self.assertEqual(res[0], expected[0], f"Value mismatch for {text}")
            self.assertEqual(res[1], expected[1], f"Unit mismatch for {text}")
            self.assertEqual(res[2], expected[2], f"Form mismatch for {text}")

    def test_precompiled_flavor_extraction(self):
        orig_cache = eval_mod._FLAVOR_CACHE
        eval_mod._FLAVOR_CACHE = {
            "flavors_dict": {
                "chicken": "chicken",
                "chicken breast": "chicken",
                "black pepper": "black pepper",
                "pepper": "pepper",
                "spinach": "spinach",
                "prawn": "prawn",
                "shrimp": "prawn",
            },
            "meat_flavors": {"chicken"},
            "vegetable_flavors": {"spinach"},
            "seafood_flavors": {"prawn"},
        }
        eval_mod._extract_flavors_from_sku_cached.cache_clear()
        try:
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
        finally:
            eval_mod._FLAVOR_CACHE = orig_cache
            eval_mod._extract_flavors_from_sku_cached.cache_clear()

    def test_tagger_resolve_flavors_from_text(self):
        flavors_dict = {"chicken": "chicken", "beef": "beef", "chili": "chili"}
        res = extract_canonical_flavors("Spicy Chili Chicken Wings", flavors_dict)
        self.assertEqual(res, {"chili", "chicken"})

        res_empty = extract_canonical_flavors("", flavors_dict)
        self.assertEqual(res_empty, set())

    @patch.object(TaxonomyClassifier, "_load_arcface_model", return_value=None)
    def test_batch_predict_third_tag_and_gk(self, _mock_load):
        import tempfile

        # Return aligned orthogonal unit vectors for pure labels vs descriptions
        def fake_encode(texts, **_kwargs):
            n = len(texts)
            vecs = np.zeros((n, 1024), dtype=np.float32)
            for i in range(n):
                vecs[i, i % 1024] = 1.0
            return {"dense": vecs}

        mock_model = MagicMock()
        mock_model.encode.side_effect = fake_encode

        descriptions = {
            "bt_descriptions": {"Noodles": "wheat noodles", "Rice": "steamed rice"},
            "third_tag_descriptions": {"Western": "western food", "Asian": "asian cuisine"},
            "third_tag_overrides": {"Noodles": "Asian"},
            "bt_to_gk_umbrella": {"Noodles": ["Pasta", "Noodles"]},
            "bt_gk_map": {"Noodles": ["Wheat", "Instant Noodles"]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            clf = TaxonomyClassifier(
                mock_model, domain="food", descriptions=descriptions, cache_dir=tmpdir
            )
            clf._trained = False

            # Query vectors: align strictly with each label's encoded prototype
            asian_idx = clf.third_tag_labels.index("Asian")
            western_idx = clf.third_tag_labels.index("Western")
            vecs = np.stack(
                [clf.third_tag_embs_pure[western_idx], clf.third_tag_embs_pure[asian_idx]]
            ).astype(np.float32)
            prices = [500.0, 1200.0]

            # Zero-shot mode falls back cleanly for GK
            gk_res = clf.batch_predict_gk(vecs, prices)
            self.assertEqual(gk_res, [([], 0.0, "zero-shot"), ([], 0.0, "zero-shot")])

            # Batch predict Third Tag: row 0 uses BT override for Noodles; row 1 uses zero-shot cosine
            third_res = clf.batch_predict_third_tag(
                vecs,
                names=["Mie Goreng", "Steamed Basmati"],
                predicted_bts=["Noodles", "Rice"],
                prices=prices,
            )
            self.assertEqual(len(third_res), 2)
            self.assertEqual(third_res[0], ("Asian", 1.0, "override"))
            self.assertEqual(third_res[1][0], "Asian")
            self.assertAlmostEqual(third_res[1][1], 1.0, places=5)
            self.assertEqual(third_res[1][2], "zero-shot")

            # Flipping vec to Western prototype switches zero-shot prediction deterministically
            vecs_flipped = np.expand_dims(clf.third_tag_embs_pure[western_idx], axis=0).astype(
                np.float32
            )
            third_flipped = clf.batch_predict_third_tag(
                vecs_flipped,
                names=["Pasta Carbonara"],
                predicted_bts=["Rice"],
                prices=[1500.0],
            )
            self.assertEqual(third_flipped[0][0], "Western")
            self.assertEqual(third_flipped[0][2], "zero-shot")


if __name__ == "__main__":
    unittest.main()
