import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from engine import config
from engine.classification.cold_start_router import (
    ClassLifecycleTier,
    ColdStartRouter,
    TaxonomyLifecycleRegistry,
    Tier1ZeroShotClassifier,
    Tier2FewShotClassifier,
    Tier3CentroidClassifier,
)
from engine.classification.classifier import ZeroShotClassifier
from engine.nlp.embedding_engine import EmbeddingEngine


class TestColdStartRouter(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_cache_dir = config.CACHE_DIR
        config.CACHE_DIR = self.test_dir

    def tearDown(self):
        config.CACHE_DIR = self.orig_cache_dir
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_taxonomy_lifecycle_registry_partitioning(self):
        """Tests that TaxonomyLifecycleRegistry partitions classes accurately based on N_c."""
        registry = TaxonomyLifecycleRegistry(domain="market", cache_dir=self.test_dir, few_shot_threshold=15)

        # Create synthetic catalog counts:
        # 'Novel Tag A': 0 examples (Tier 1 Zero-Shot)
        # 'Emerging Tag B': 5 examples (Tier 2 Few-Shot)
        # 'Mature Tag C': 25 examples (Tier 3 Warm Centroid)
        cat_df = pd.DataFrame({
            "Name": ["item"] * 30,
            "basictype": (["Emerging Tag B"] * 5) + (["Mature Tag C"] * 25),
        })

        registered_bts = ["Novel Tag A", "Emerging Tag B", "Mature Tag C"]
        descriptions = {
            "Novel Tag A": "A brand new market tag without prior labeled data",
            "Emerging Tag B": "Few examples available",
            "Mature Tag C": "High volume historical category",
        }

        registry.sync_from_catalog(
            registered_bts=registered_bts,
            descriptions=descriptions,
            cat_df=cat_df,
            embed_engine=None,
        )

        self.assertEqual(registry.get_tier("Novel Tag A"), ClassLifecycleTier.TIER_1_ZERO_SHOT)
        self.assertEqual(registry.get_tier("Emerging Tag B"), ClassLifecycleTier.TIER_2_FEW_SHOT)
        self.assertEqual(registry.get_tier("Mature Tag C"), ClassLifecycleTier.TIER_3_WARM_CENTROID)

        self.assertEqual(registry.get_count("Novel Tag A"), 0)
        self.assertEqual(registry.get_count("Emerging Tag B"), 5)
        self.assertEqual(registry.get_count("Mature Tag C"), 25)

        self.assertIn("Novel Tag A", registry.zero_shot_classes)
        self.assertIn("Emerging Tag B", registry.few_shot_classes)
        self.assertIn("Mature Tag C", registry.warm_classes)

    def test_dynamic_tag_registration(self):
        """Tests runtime dynamic tag registration for newly introduced tags without historical data."""
        registry = TaxonomyLifecycleRegistry(domain="food", cache_dir=self.test_dir)
        registry.register_tag("Artisanal Kombucha", "Fermented effervescent sweetened tea drink", sample_count=0)

        self.assertEqual(registry.get_tier("Artisanal Kombucha"), ClassLifecycleTier.TIER_1_ZERO_SHOT)
        self.assertEqual(registry.get_count("Artisanal Kombucha"), 0)
        self.assertIn("Artisanal Kombucha", registry.zero_shot_classes)
        self.assertEqual(
            registry.class_descriptions["Artisanal Kombucha"],
            "Fermented effervescent sweetened tea drink",
        )

    def test_tier1_zero_shot_cross_encoder_scoring(self):
        """Tests Tier 1 Cross-Encoder evaluation and pair formatting."""
        registry = TaxonomyLifecycleRegistry(domain="market", cache_dir=self.test_dir)
        registry.register_tag(
            "Eco Dishwasher Tablet",
            "Biodegradable plant-based dish cleaning tablets",
            sample_count=0,
        )

        mock_embed = MagicMock(spec=EmbeddingEngine)
        # Mock cross-encoder returning positive logit (relevant match: logit 2.5 -> sigmoid ~0.92)
        mock_embed.score_cross_encoder.return_value = np.array([2.5], dtype=np.float32)

        tier1 = Tier1ZeroShotClassifier(
            embed_engine=mock_embed,
            registry=registry,
            confidence_threshold=0.50,
        )

        sku_name = "Eco Clean Zero Waste Dishwasher Tablets 30pk"
        tag, conf, source = tier1.predict(sku_name)

        self.assertEqual(tag, "Eco Dishwasher Tablet")
        self.assertGreater(conf, 0.90)
        self.assertEqual(source, "zero-shot")

        # Verify pair formatting: (sku_name, "Basic Type: {bt_name} - {bt_description}")
        mock_embed.score_cross_encoder.assert_called_once()
        call_pairs = mock_embed.score_cross_encoder.call_args[0][0]
        self.assertEqual(len(call_pairs), 1)
        self.assertEqual(call_pairs[0][0], sku_name)
        self.assertEqual(
            call_pairs[0][1],
            "Basic Type: Eco Dishwasher Tablet - Biodegradable plant-based dish cleaning tablets",
        )

    def test_tier2_few_shot_knn_and_mlknn_keywords(self):
        """Tests Tier 2 Cosine-Weighted k-NN probability and multi-label generic keyword propagation."""
        registry = TaxonomyLifecycleRegistry(domain="market", cache_dir=self.test_dir)
        registry.register_tag("Oat Milk Barista", sample_count=6)
        registry.register_tag("Almond Milk", sample_count=8)

        mock_vs = MagicMock()
        tier2 = Tier2FewShotClassifier(
            registry=registry,
            vector_store=mock_vs,
            top_k=5,
            confidence_threshold=0.50,
            gk_weight_threshold=0.35,
        )

        # Synthetic top-k retrieved catalog items with similarities, BTs, and GKs
        neighbor_hits = [
            {"basictype": "Oat Milk Barista", "Generic keywords": "oat milk, barista, plant milk", "_qdrant_score_": 0.90},
            {"basictype": "Oat Milk Barista", "Generic keywords": "oat milk, plant milk", "_qdrant_score_": 0.85},
            {"basictype": "Oat Milk Barista", "Generic keywords": "oat milk, barista", "_qdrant_score_": 0.80},
            {"basictype": "Almond Milk", "Generic keywords": "almond milk, nut milk", "_qdrant_score_": 0.60},
            {"basictype": "Almond Milk", "Generic keywords": "almond milk", "_qdrant_score_": 0.50},
        ]
        # Total similarity = 0.90 + 0.85 + 0.80 + 0.60 + 0.50 = 3.65
        # Oat Milk Barista sum = 2.55 -> Prob = 2.55 / 3.65 = 0.6986 (~0.70)
        # Almond Milk sum = 1.10 -> Prob = 1.10 / 3.65 = 0.3013 (~0.30)
        # Keyword weights:
        # 'oat milk': (0.90 + 0.85 + 0.80) / 3.65 = 0.6986 >= 0.35 (KEEP)
        # 'barista': (0.90 + 0.80) / 3.65 = 1.70 / 3.65 = 0.4657 >= 0.35 (KEEP)
        # 'plant milk': (0.90 + 0.85) / 3.65 = 1.75 / 3.65 = 0.4794 >= 0.35 (KEEP)
        # 'almond milk': (0.60 + 0.50) / 3.65 = 1.10 / 3.65 = 0.3013 < 0.35 (DROP)

        best_bt, prob, propagated_gks, src = tier2.evaluate_neighbors(neighbor_hits)

        self.assertEqual(best_bt, "Oat Milk Barista")
        self.assertAlmostEqual(prob, 2.55 / 3.65, places=3)
        self.assertEqual(src, "few-shot")

        # Verify propagated keywords order and thresholds
        self.assertIn("oat milk", propagated_gks)
        self.assertIn("barista", propagated_gks)
        self.assertIn("plant milk", propagated_gks)
        self.assertNotIn("almond milk", propagated_gks)

    def test_tier3_temperature_scaled_centroid_prediction(self):
        """Tests Tier 3 normalized centroid matrix multiplication with temperature scaling."""
        registry = TaxonomyLifecycleRegistry(domain="market", cache_dir=self.test_dir)
        registry.register_tag("Liquid Detergent", sample_count=50)
        registry.register_tag("Fabric Softener", sample_count=40)

        # Create two orthogonal 1024-dim normalized prototypes
        dim = 1024
        proto_detergent = np.zeros(dim, dtype=np.float32)
        proto_detergent[0] = 1.0

        proto_softener = np.zeros(dim, dtype=np.float32)
        proto_softener[1] = 1.0

        registry.warm_centroids = np.vstack([proto_detergent, proto_softener])
        registry.warm_index_to_class = ["Liquid Detergent", "Fabric Softener"]
        registry.class_to_warm_index = {"Liquid Detergent": 0, "Fabric Softener": 1}

        tier3 = Tier3CentroidClassifier(registry=registry, tau=0.05)

        # Query vector strongly aligned with Liquid Detergent
        query_vec = np.zeros(dim, dtype=np.float32)
        query_vec[0] = 0.95
        query_vec[1] = 0.05
        query_vec /= np.linalg.norm(query_vec)

        pred_class, conf, src = tier3.predict(query_vec)
        self.assertEqual(pred_class, "Liquid Detergent")
        self.assertGreater(conf, 0.99)
        self.assertEqual(src, "trained")

        # Test batch centroid prediction
        batch_vecs = np.vstack([query_vec, proto_softener])
        batch_results = tier3.batch_predict(batch_vecs)
        self.assertEqual(len(batch_results), 2)
        self.assertEqual(batch_results[0][0], "Liquid Detergent")
        self.assertEqual(batch_results[1][0], "Fabric Softener")

    def test_router_integration_as_classifier_fallback(self):
        """Tests that ColdStartRouter acts as a seamless fallback inside ZeroShotClassifier."""
        descriptions = {
            "bt_descriptions": {
                "Standard Coffee": "Traditional brewed filter coffee",
                "New Cold Brew Tag": "Steeped cold coffee concentrate drink",
            }
        }

        mock_embed = MagicMock(spec=EmbeddingEngine)
        mock_embed.encode.return_value = {
            "dense": np.random.randn(2, 1024).astype(np.float32),
            "sparse": [{}, {}],
        }

        # Initialize classifier with "logreg" model without pre-existing training data
        clf = ZeroShotClassifier(
            model=mock_embed,
            domain="market",
            descriptions=descriptions,
            cache_dir=self.test_dir,
        )

        self.assertIsNotNone(clf.cold_start_router)

        # Register a brand new tag dynamically
        clf.register_new_tag(
            "Organic Oat Milk",
            "Plant-based dairy alternative made from organic oats",
        )

        self.assertIn("Organic Oat Milk", clf.cold_start_router.registry.zero_shot_classes)
        self.assertEqual(
            clf.cold_start_router.registry.get_tier("Organic Oat Milk"),
            ClassLifecycleTier.TIER_1_ZERO_SHOT,
        )


if __name__ == "__main__":
    unittest.main()
