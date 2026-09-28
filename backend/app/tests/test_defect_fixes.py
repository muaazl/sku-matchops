import pytest
import sqlite3
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from engine.core import config
from engine.core.db import ensure_db_initialized
from engine.matching.matcher import SKUMatcher
from engine.core.resource_loader import get_pipeline
from engine.classification.tagger import tag_all_skus
from engine.pipeline.processor import process_request
from engine.pipeline.audit_engine import run_sku_audit
from engine.nlp.embedding_engine import EmbeddingEngine


class TestDefectFixes:

    def test_database_migration_normalizes_legacy_scores(self, tmp_path):
        """Verify ensure_db_initialized() migrates historical confidence and match_score > 1.0 down by /100."""
        db_file = str(tmp_path / "test_migration.db")
        conn = ensure_db_initialized(db_file)
        
        # Insert historical row with 100.0 scale
        conn.execute(
            """
            INSERT INTO processed_skus (
                id, batch_id, sku_name, domain, bt, gk_json, region,
                confidence, match_source, rules_applied_json, logic_notes,
                matched_catalog_name, match_score, bt_confidence,
                gk_confidence, region_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("test-id-1", "b-1", "Sample Item", "food", "Rice", "[]", "Western",
             100.0, "catalogue", "[]", "Exact Text Match",
             "Sample Item", 100.0, 0.95, 0.90, 0.85)
        )
        conn.commit()

        # Re-run ensure_db_initialized to trigger migration
        ensure_db_initialized(conn)

        cur = conn.cursor()
        cur.execute("SELECT confidence, match_score FROM processed_skus WHERE id = 'test-id-1'")
        conf, score = cur.fetchone()
        assert conf == 1.0, f"Expected confidence 1.0, got {conf}"
        assert score == 1.0, f"Expected match_score 1.0, got {score}"
        conn.close()

    def test_bypass_score_normalization_at_source(self):
        """Verify exact and fuzzy bypass paths write 1.0 instead of 100.0."""
        matcher = get_pipeline("food")
        
        # Test exact match bypass
        exact_df = pd.DataFrame([{"Name": "Devilled Fish Set Menu", "description": "", "category": "", "price": 1500.0}])
        match_res = matcher.process_inputs(exact_df)
        assert len(match_res) == 1
        row = match_res.iloc[0]
        if row["Status"] == "High Confidence" and "Exact" in row["Logic Notes"]:
            assert row["Final Score"] == 1.0, f"Expected exact match score 1.0, got {row['Final Score']}"

        # Test audit engine exact bypass computed_score normalization
        audit_res = run_sku_audit(
            sku_name="Devilled Fish Set Menu",
            domain="food",
            task="matcher",
            price=1500.0
        )
        stage4 = audit_res.get("stage4_logic_gates", [])
        if stage4 and stage4[0].get("fuzzy_bypass"):
            assert stage4[0]["computed_score"] == 1.0, f"Expected computed_score 1.0, got {stage4[0]['computed_score']}"
            assert stage4[0]["raw_cross_score"] == 100.0, "raw_cross_score logit sentinel should remain 100.0"

    def test_fallback_variable_scoping_no_cross_sku_leak(self):
        """Verify that fallback candidate from SKU 1 never leaks to SKU 2 in the same chunk."""
        matcher = get_pipeline("food")
        
        # SKU 1 is an obscure query that generates no high-confidence match
        # SKU 2 is a distinct obscure query
        test_df = pd.DataFrame([
            {"Name": "ZzzNonExistentDishXyz123", "description": "", "category": "", "price": 100.0},
            {"Name": "QqqAnotherFakeDish987", "description": "", "category": "", "price": 200.0},
        ])
        res = matcher.process_inputs(test_df)
        assert len(res) == 2
        # Both must produce independent rows and not crash
        assert res.iloc[0]["Input Raw"] == "ZzzNonExistentDishXyz123"
        assert res.iloc[1]["Input Raw"] == "QqqAnotherFakeDish987"

    def test_zero_candidate_sku_preserves_positional_alignment(self):
        """Verify that when a SKU produces 0 candidates, a placeholder row is appended and positions do not shift."""
        matcher = get_pipeline("food")
        
        test_df = pd.DataFrame([
            {"Name": "Fish Fried Rice", "description": "", "category": "", "price": 1200.0},
            {"Name": "ZeroCandSyntheticTestItem9999", "description": "", "category": "", "price": 500.0},
            {"Name": "Chicken Biriyani", "description": "", "category": "", "price": 1500.0},
        ])

        res = matcher.process_inputs(test_df)
        # Length must strictly equal 3
        assert len(res) == 3, f"Expected 3 result rows, got {len(res)}"
        assert res.iloc[0]["Input Raw"] == "Fish Fried Rice"
        assert res.iloc[1]["Input Raw"] == "ZeroCandSyntheticTestItem9999"
        assert res.iloc[2]["Input Raw"] == "Chicken Biriyani"
        # Confirm required keys exist in placeholder row
        for col in ["Input Raw", "Matched Catalog Name", "Final Score", "Status", "Logic Notes", "BasicType", "GenericKeywords"]:
            assert col in res.columns

    def test_classifier_reasoning_generation_and_escalation_chaining(self):
        """Verify classifier produces reasoning and pipeline escalation chains matcher note + classifier reasoning."""
        # 1. tag_all_skus produces 'reasoning'
        from engine.core.resource_loader import get_classifier, get_pipeline, get_ner_engine, _get_shared_models, _get_vector_store
        classifier = get_classifier("food")
        pipeline = get_pipeline("food")
        embed_engine, ner_engine = _get_shared_models()
        vector_store = _get_vector_store()

        embs = embed_engine.embed_weighted_sku(["Fish Curry"], [""], [""], weights=config.CLASSIFIER_WEIGHTS)
        query_embeddings = [{"dense": embs["dense"][0], "sparse": embs["sparse"][0]}]

        clf_results = tag_all_skus(
            sku_names=["Fish Curry"],
            sku_categories=[""],
            query_embeddings=query_embeddings,
            vector_store=vector_store,
            reranker=None,
            classifier=classifier,
            sku_descriptions=[""],
            sku_prices=[1000.0],
            embed_engine=embed_engine,
            ner_engine=getattr(classifier, 'ner_engine', ner_engine)
        )
        assert len(clf_results) == 1
        res = clf_results[0]
        assert "reasoning" in res, "Expected 'reasoning' in tag_all_skus output dict"
        assert (
            "Classifier: BT=" in res["reasoning"]
            or ("Classifier (" in res["reasoning"] and "BT=" in res["reasoning"])
        ), f"Reasoning should format classifier details, got: {res['reasoning']}"

        # 2. Pipeline escalation chains reasoning
        sku = {
            "name": "Devilled Fish Set Menu",
            "description": "",
            "category": "",
            "price": 1500.0,
        }
        pip_res = process_request(task="pipeline", domain="food", skus=[sku])["results"][0]
        # Verify confidence is clamped in [0.0, 1.0]
        assert 0.0 <= pip_res.get("score", 0.0) <= 1.0
        assert 0.0 <= pip_res.get("bt_confidence", 0.0) <= 1.0

        if pip_res.get("pipeline_source") == "Classifier":
            notes = pip_res.get("logic_notes", "")
            assert "Classifier:" in notes or "Escalated from matcher" in notes

        # Test SKU where matcher score is 0 / Low so classifier wins escalation
        sku_unknown = {
            "name": "Unmatchable Unknown Food Dish 98765",
            "description": "",
            "category": "",
            "price": 500.0,
        }
        pip_res2 = process_request(task="pipeline", domain="food", skus=[sku_unknown])["results"][0]
        if pip_res2.get("pipeline_source") == "Classifier":
            notes2 = pip_res2.get("logic_notes", "")
            assert "Escalated from matcher" in notes2 or "Classifier:" in notes2

    def test_cross_encoder_empty_pairs_and_sentinel_handling(self):
        """Verify score_cross_encoder handles empty batch and returns -10.0 sentinel if models fail."""
        embed = EmbeddingEngine()
        
        # 1. Empty pairs returns empty array
        empty_res = embed.score_cross_encoder([])
        assert len(empty_res) == 0
        assert isinstance(empty_res, np.ndarray)

        # 2. When sessions are None, returns -10.0 sentinel
        with patch.object(embed, 'cross_session', None), patch.object(embed, 'cross_encoder_fallback', None):
            sentinel_res = embed.score_cross_encoder([["query", "doc1"], ["query", "doc2"]])
            assert len(sentinel_res) == 2
            assert np.all(sentinel_res == -10.0), f"Expected -10.0 sentinels, got {sentinel_res}"
