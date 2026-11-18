from unittest.mock import MagicMock

import pandas as pd
import pytest

from engine.matching.logic_gates import LogicGates


class TestTextWeightPriorityMatching:
    """Tests to verify that exact/fuzzy text matching takes priority over weight matching."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        return embedder

    @pytest.fixture
    def logic_gates(self, mock_embedder):
        return LogicGates(embed_engine=mock_embedder)

    def test_weight_boost_gated_by_text_similarity(self, logic_gates):
        """Test that weight match only adds +2.0 boost if text similarity is sufficient (>= 65)."""
        input_clean = "smak hot spicy mixture 150 g"
        input_no_weights = "smak hot spicy mixture"
        input_w_data = (150.0, "g", "solid")
        input_entities = {"brand": {"smak"}, "flavor": {"hot", "spicy"}}

        # Candidate 1: Unrelated product with matching 150g weight ("Smak Spicy Bite 150g")
        cand_spicy_bite = {
            "Name": "Smak Spicy Bite 150g",
            "clean_text": "smak spicy bite 150 g",
            "clean_no_weights": "smak spicy bite",
            "weight_val": (150.0, "g", "solid"),
            "entities": {"brand": {"smak"}, "flavor": {"spicy"}},
            "basictype": "Snacks",
            "Generic keywords": "",
        }

        score_sb, status_sb, reasons_sb = logic_gates.apply_logic_gates(
            input_clean=input_clean,
            input_entities=input_entities,
            match_row=cand_spicy_bite,
            raw_ai_score=0.60,  # cosine reranker score
            input_price=0.0,
            input_w_data=input_w_data,
            input_no_weights=input_no_weights,
            domain="market",
        )

        # "smak hot spicy mixture" vs "smak spicy bite" has token_sort_ratio < 65
        # It should NOT receive the +0.10 weight boost into high confidence
        assert "Low Text Sim" in reasons_sb
        # Since raw score was 0.60 and no boost was added, score remains 0.60 (< 0.75 High threshold)
        assert status_sb != "High Confidence"

        # Candidate 2: Similar product name with matching weight
        cand_spicy_mix = {
            "Name": "Smak Spicy Hot Mixture 150g",
            "clean_text": "smak spicy hot mixture 150 g",
            "clean_no_weights": "smak spicy hot mixture",
            "weight_val": (150.0, "g", "solid"),
            "entities": {"brand": {"smak"}, "flavor": {"spicy", "hot"}},
            "basictype": "Snacks",
            "Generic keywords": "",
        }

        score_sm, status_sm, reasons_sm = logic_gates.apply_logic_gates(
            input_clean=input_clean,
            input_entities=input_entities,
            match_row=cand_spicy_mix,
            raw_ai_score=0.60,
            input_price=0.0,
            input_w_data=input_w_data,
            input_no_weights=input_no_weights,
            domain="market",
        )

        # Token sort ratio is 100%, so it should receive the +0.10 boost
        assert "Weight Match (150)" in reasons_sm
        assert "Low Text Sim" not in reasons_sm

    def test_fuzzy_bypass_scores_highest_even_with_weight_mismatch(self, logic_gates):
        """Test that Whole-SKU Fuzzy Match (100%) achieves max score even if weight mismatches."""
        input_clean = "smak hot spicy mixture 150 g"
        input_no_weights = "smak hot spicy mixture"
        input_w_data = (150.0, "g", "solid")
        input_entities = {"brand": {"smak"}, "flavor": {"hot", "spicy"}}

        cand_200g = {
            "Name": "Smak Hot & Spicy Mixture 200g",
            "clean_text": "smak hot spicy mixture 200 g",
            "clean_no_weights": "smak hot spicy mixture",
            "weight_val": (200.0, "g", "solid"),
            "entities": {"brand": {"smak"}, "flavor": {"hot", "spicy"}},
            "basictype": "Snacks",
            "Generic keywords": "",
        }

        score_200, status_200, reasons_200 = logic_gates.apply_logic_gates(
            input_clean=input_clean,
            input_entities=input_entities,
            match_row=cand_200g,
            raw_ai_score=0.70,
            input_price=0.0,
            input_w_data=input_w_data,
            input_no_weights=input_no_weights,
            domain="market",
        )

        assert "Whole-SKU Fuzzy Match (100%)" in reasons_200
        assert "Weight Mismatch (150 vs 200)" in reasons_200
        assert status_200 == "High Confidence"
        assert score_200 == 1.0  # Max probability score 1.0 for whole-SKU 100% fuzzy match

    def test_token_sorted_map_multi_weight_selection(self):
        """Test that token_sorted_map selects the best pack size or provides mismatch note."""
        from engine.matching.matcher import SKUMatcher

        catalog_df = pd.DataFrame(
            [
                {
                    "Name": "Smak Hot & Spicy Mixture 200g",
                    "Brand": "Smak",
                    "BasicType": "Snacks",
                    "Generic keywords": "Mix",
                    "clean_text": "smak hot spicy mixture 200 g",
                    "clean_no_weights": "smak hot spicy mixture",
                    "weight_val": (200.0, "g", "solid"),
                },
                {
                    "Name": "Smak Hot & Spicy Mixture 150g",
                    "Brand": "Smak",
                    "BasicType": "Snacks",
                    "Generic keywords": "Mix",
                    "clean_text": "smak hot spicy mixture 150 g",
                    "clean_no_weights": "smak hot spicy mixture",
                    "weight_val": (150.0, "g", "solid"),
                },
            ]
        )

        mock_cache = MagicMock()
        mock_cache.manage_catalog_cache.return_value = (catalog_df, MagicMock())

        matcher = SKUMatcher(
            catalog_df=catalog_df,
            brands_df=pd.DataFrame(),
            ner_engine=MagicMock(),
            embed_engine=MagicMock(),
            cache_manager=mock_cache,
            logic_gates=MagicMock(),
            domain="market",
        )

        # Verify token_sorted_map holds both entries for the same product text
        sorted_key = "hot mixture smak spicy"
        assert sorted_key in matcher.token_sorted_map
        assert len(matcher.token_sorted_map[sorted_key]) == 2

        # 1. Exact 150g match should select the 150g catalog row
        res_150 = matcher.process_inputs(pd.DataFrame([{"Name": "Smak Hot & Spicy Mixture 150G"}]))
        assert res_150.iloc[0]["Matched Catalog Name"] == "Smak Hot & Spicy Mixture 150g"

        # 2. 100g query (not in catalog) should match the 100% text match product with mismatch note
        res_100 = matcher.process_inputs(pd.DataFrame([{"Name": "Smak Hot & Spicy Mixture 100G"}]))
        assert "Smak Hot & Spicy Mixture" in res_100.iloc[0]["Matched Catalog Name"]
        assert "Weight Mismatch" in res_100.iloc[0]["Logic Notes"]

        # 3. Query without weight should match with 100% fuzzy match
        res_noweight = matcher.process_inputs(pd.DataFrame([{"Name": "Smak Hot & Spicy Mixture"}]))
        assert "Smak Hot & Spicy Mixture" in res_noweight.iloc[0]["Matched Catalog Name"]
        assert "Fuzzy Match (100%)" in res_noweight.iloc[0]["Logic Notes"]

    def test_fractional_unit_standardization_and_reranker_nan_sanitization(self):
        """Ensure sub-unit scaling keeps decimals (500mg -> 0.5g) and NaNs never leak into rerank prompts."""
        from engine.nlp.text_cleaner import TextPipeline

        assert "0.5g" in TextPipeline.standardize_units("Vitamin C 500mg")
        assert "0.25g" in TextPipeline.standardize_units("Folic Acid 250mg")

        prompt = TextPipeline.format_matcher_rerank_text(
            name=float("nan"),
            seller_category=float("nan"),
            brand_or_flavor="nan",
            description=None,
            domain="market",
        )
        assert prompt == "Name: "

        cat_prompt = TextPipeline.format_catalog_row_for_reranker(
            {
                "Name": "Kraft Cheese 200g",
                "Brand": float("nan"),
                "entities": {"brand": {"kraft"}},
                "Description": "nan",
                "Categories": float("nan"),
            },
            domain="market",
        )
        assert "Brand: kraft" in cat_prompt
        assert "| Description:" not in cat_prompt
        assert "SellerCategory:" not in cat_prompt

    def test_food_unspecified_flavor_cap_after_bt_bump_and_lowercase_category(self, logic_gates):
        """Ensure food unspecified-flavor cap runs after BT bump and market category gate reads lowercase key."""
        food_gates = LogicGates(
            embed_engine=MagicMock(),
            brands_df=pd.DataFrame([{"Flavor Name": "chocolate", "Aliases": "", "Is_Weak": False}]),
        )
        cand_food = {
            "Name": "Chocolate Waffle",
            "clean_text": "chocolate waffle",
            "clean_no_weights": "chocolate waffle",
            "weight_val": (None, None, None),
            "entities": {"flavor": {"chocolate"}},
            "basictype": "Waffle",
        }
        score, status, _ = food_gates.apply_logic_gates(
            input_clean="plain waffle",
            input_entities={},
            match_row=cand_food,
            raw_ai_score=0.90,
            input_price=0.0,
            input_w_data=(None, None, None),
            input_no_weights="plain waffle",
            domain="food",
            predicted_bt="Waffle",
        )
        from engine import config

        assert score < config.CONFIDENCE_THRESHOLD_HIGH
        assert status != "High Confidence"

        cand_market = {
            "Name": "Sample Cola 500ml",
            "clean_text": "sample cola 500 ml",
            "clean_no_weights": "sample cola",
            "weight_val": (500.0, "ml", "liquid"),
            "entities": {"brand": {"sample"}},
            "basictype": "Soft Drink",
            "category": "Cold Beverages",
        }
        _, _, reasons = logic_gates.apply_logic_gates(
            input_clean="sample cola 500 ml",
            input_entities={"brand": {"sample"}},
            match_row=cand_market,
            raw_ai_score=0.70,
            input_price=0.0,
            input_w_data=(500.0, "ml", "liquid"),
            input_no_weights="sample cola",
            domain="market",
            input_category="beverages",
        )
        assert "Category Match Bonus" in reasons
