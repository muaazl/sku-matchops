import pytest
import pandas as pd
from unittest.mock import MagicMock
from engine.matching.logic_gates import LogicGates
from engine.nlp.text_cleaner import TextPipeline


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
            raw_ai_score=2.0,  # cross encoder score
            input_price=0.0,
            input_w_data=input_w_data,
            input_no_weights=input_no_weights,
            domain="market",
        )

        # "smak hot spicy mixture" vs "smak spicy bite" has token_sort_ratio < 65
        # It should NOT receive the +2.0 boost into high confidence
        assert "Low Text Sim" in reasons_sb
        # Since raw score was 2.0 and no boost was added, score should remain 2.0 (prob < 0.8)
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
            raw_ai_score=2.0,
            input_price=0.0,
            input_w_data=input_w_data,
            input_no_weights=input_no_weights,
            domain="market",
        )

        # Token sort ratio is 100%, so it should receive the +2.0 boost
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
            raw_ai_score=3.0,
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

        catalog_df = pd.DataFrame([
            {"Name": "Smak Hot & Spicy Mixture 200g", "Brand": "Smak", "BasicType": "Snacks", "Generic keywords": "Mix", "clean_text": "smak hot spicy mixture 200 g", "clean_no_weights": "smak hot spicy mixture", "weight_val": (200.0, "g", "solid")},
            {"Name": "Smak Hot & Spicy Mixture 150g", "Brand": "Smak", "BasicType": "Snacks", "Generic keywords": "Mix", "clean_text": "smak hot spicy mixture 150 g", "clean_no_weights": "smak hot spicy mixture", "weight_val": (150.0, "g", "solid")},
        ])

        mock_cache = MagicMock()
        mock_cache.manage_catalog_cache.return_value = (catalog_df, MagicMock())

        matcher = SKUMatcher(
            catalog_df=catalog_df,
            brands_df=pd.DataFrame(),
            ner_engine=MagicMock(),
            embed_engine=MagicMock(),
            cache_manager=mock_cache,
            logic_gates=MagicMock(),
            domain="market"
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
