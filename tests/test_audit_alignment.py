import pytest

from engine.classification.loader import is_conflicting_dish_tag
from engine.core.resource_loader import get_classifier, get_pipeline
from engine.pipeline.audit_engine import run_sku_audit
from engine.pipeline.processor import process_request
from engine.rules_engine import loader

SEAFOOD_RULE_ID = "test_audit_seafood"

# Production "Rule 1": seafood dishes get the 'Seafood' GK unless they are short eats,
# already say "Seafood", or also contain land meat.
SEAFOOD_CONDITIONS = [
    (1, "flavor_is", "seafood", 0),
    (2, "gk_contains", "Short Eats", 1),
    (3, "sku_contains", "Seafood", 1),
    (4, "flavor_is", "meat", 1),
]


@pytest.fixture(scope="class")
def seafood_rule():
    """Seeds the food-domain Seafood rule directly into the Engine rule cache, then restores it."""
    prev = (loader._RULES_CACHE, loader._RULES_VERSION)
    loader.set_rules_from_dicts(
        [
            {
                "rule_id": SEAFOOD_RULE_ID,
                "domain": "food",
                "priority": 1,
                "description": "Seafood GK",
                "reasoning": "Seafood dishes get the Seafood GK",
                "condition_logic": "AND",
                "conditions": [
                    {
                        "condition_group": g,
                        "condition_type": t,
                        "value": v,
                        "negate": bool(n),
                    }
                    for g, t, v, n in SEAFOOD_CONDITIONS
                ],
                "actions": [{"action_type": "add_gk", "value": "Seafood"}],
            }
        ]
    )
    yield
    loader._RULES_CACHE, loader._RULES_VERSION = prev
    loader._cached_rules.cache_clear()


def test_cross_dish_contamination_filter():
    # Chop Suey Rice must be marked as conflicting dish under Fried Rice
    assert is_conflicting_dish_tag("Chop Suey Rice", "Fried Rice") is True
    assert is_conflicting_dish_tag("Vegetable Chop Suey Rice", "Fried Rice") is True
    assert is_conflicting_dish_tag("Chicken Biriyani", "Fried Rice") is True

    # Valid sub-dishes of Fried Rice must be allowed
    assert is_conflicting_dish_tag("Devilled Fish Fried Rice", "Fried Rice") is False
    assert is_conflicting_dish_tag("Fish Fried Rice", "Fried Rice") is False
    assert is_conflicting_dish_tag("Seafood Fried Rice", "Fried Rice") is False
    assert is_conflicting_dish_tag("Set Menu", "Fried Rice") is False
    assert is_conflicting_dish_tag("Seafood", "Fried Rice") is False


@pytest.mark.live_models
@pytest.mark.usefixtures("seafood_rule")
class TestAuditAlignment:
    @classmethod
    def setup_class(cls):
        import os

        import httpx

        from engine import config

        meta_path = os.path.join(config.CACHE_DIR, "food_catalog_metadata.pkl")
        if (
            not os.path.exists(config.EMBEDDING_ONNX)
            or not os.path.exists(config.get_bt_arcface_onnx_path("food"))
            or not os.path.exists(meta_path)
        ):
            pytest.skip("Local ONNX models or synced food catalog cache not found on disk")
        try:
            if httpx.get(f"{config.QDRANT_URL.rstrip('/')}/readyz", timeout=1.0).status_code >= 500:
                pytest.skip(f"Qdrant not ready at {config.QDRANT_URL}")
        except Exception:
            pytest.skip(f"Qdrant not reachable at {config.QDRANT_URL}")
        # Warm up models for food domain
        get_classifier("food")
        get_pipeline("food")

    def test_classifier_parity_audit_vs_processor(self):
        sku = {
            "name": "Devilled Fish Set Menu",
            "description": "",
            "category": "",
            "price": 1500.0,
        }

        proc_clf = process_request(task="classifier", domain="food", skus=[sku])["results"][0]
        audit_clf = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="classifier",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )["final_output"]

        assert proc_clf["suggested_bt"] == audit_clf["suggested_bt"]
        assert proc_clf["suggested_gk"] == audit_clf["suggested_gk"]
        assert proc_clf["suggested_region"] == audit_clf["suggested_region"]

        # Ensure Chop Suey is NOT in suggested GK
        assert "Chop Suey" not in proc_clf["suggested_gk"]
        assert "Chop Suey" not in audit_clf["suggested_gk"]

    def test_pipeline_parity_audit_vs_processor(self):
        sku = {
            "name": "Devilled Fish Set Menu",
            "description": "",
            "category": "",
            "price": 1500.0,
        }

        proc_pip = process_request(task="pipeline", domain="food", skus=[sku])["results"][0]
        audit_pip = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="pipeline",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )["final_output"]

        assert proc_pip["matched_catalog_name"] == audit_pip["matched_catalog_name"]
        assert proc_pip["score"] == audit_pip["score"]
        assert 0.0 <= proc_pip["score"] <= 1.0
        assert proc_pip["status"] == audit_pip["status"]
        assert proc_pip["suggested_bt"] == audit_pip["suggested_bt"]
        assert proc_pip["suggested_gk"] == audit_pip["suggested_gk"]
        assert proc_pip["suggested_region"] == audit_pip["suggested_region"]

    def test_gk_classifier_parity_audit_vs_processor(self):
        sku = {
            "name": "Devilled Chicken Burger",
            "description": "",
            "category": "",
            "price": 1000.0,
        }
        proc_clf = process_request(task="classifier", domain="food", skus=[sku])["results"][0]
        audit_full = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="classifier",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )
        audit_clf = audit_full["final_output"]

        assert proc_clf["suggested_bt"] == audit_clf["suggested_bt"]
        assert proc_clf["suggested_gk"] == audit_clf["suggested_gk"]
        assert proc_clf["suggested_region"] == audit_clf["suggested_region"]
        assert 0.0 <= proc_clf["bt_confidence"] <= 1.0
        assert 0.0 <= proc_clf["gk_confidence"] <= 1.0

    def test_matcher_parity_audit_vs_processor(self):
        sku = {
            "name": "Chicken Burger",
            "description": "",
            "category": "",
            "price": 1300.0,
        }
        proc_res = process_request(task="matcher", domain="food", skus=[sku])["results"][0]
        audit_res = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="matcher",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )["final_output"]

        assert proc_res["matched_catalog_name"] == audit_res["matched_catalog_name"]
        assert proc_res["score"] == audit_res["score"]
        assert proc_res["status"] == audit_res["status"]
        assert proc_res["suggested_bt"] == audit_res["suggested_bt"]
        assert proc_res["suggested_gk"] == audit_res["suggested_gk"]
        assert proc_res["suggested_region"] == audit_res["suggested_region"]

    def test_chicken_noodles_mix_description_resolution(self):
        sku = {
            "name": "Chicken Noodles (Normal)",
            "description": "Carrot, Leaks, Eggs, Boiled Chicken Mix With Noodles. Served With Chili Paste Chicken Gravy & Tomato Sauce",
            "category": "NOODLES",
            "price": 1280.0,
        }
        proc_res = process_request(skus=[sku], task="pipeline", domain="food")["results"][0]
        audit_res = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="pipeline",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )["final_output"]

        # Ensure neither processor nor audit appended 'mixed' or diverged across paths
        assert "mixed" not in proc_res.get("sku_name", "").lower()
        assert proc_res["matched_catalog_name"] == audit_res["matched_catalog_name"]
        assert proc_res["score"] == audit_res["score"]
        assert proc_res["status"] == audit_res["status"]
        assert proc_res["suggested_bt"] == audit_res["suggested_bt"]
        assert proc_res["suggested_gk"] == audit_res["suggested_gk"]


def test_tagger_reranker_status_and_audit_matcher_exact_bypass_without_qdrant(monkeypatch):
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    from engine import config
    from engine.classification.tagger import get_status
    from engine.matching.logic_gates import LogicGates

    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD_HIGH", 0.62)
    monkeypatch.setattr(config, "RERANKER_THRESHOLD", 0.30)
    assert get_status(0.65, True, "reranker") == "AUTO"
    assert get_status(0.40, True, "reranker") == "REVIEW"
    assert get_status(0.20, True, "reranker") == "LOW"

    fake_pipeline = MagicMock()
    fake_pipeline.raw_catalog = pd.DataFrame(
        [
            {
                "Name": "Plain Item",
                "clean_text": "plain item",
                "basictype": "General",
                "Generic keywords": "General",
                "category": "Misc",
                "weight_val": (None, None, None),
            }
        ]
    )
    fake_pipeline.exact_match_map = {"plain item": [0]}
    fake_pipeline.token_sorted_map = {}
    fake_pipeline.ner = MagicMock()
    fake_pipeline.ner.extract_entities.return_value = {}
    fake_pipeline.embedder = MagicMock()
    fake_pipeline.embedder.embed_weighted_sku.return_value = {
        "dense": np.ones((1, 4), dtype=np.float32),
        "bm25": [{}],
    }
    fake_pipeline.classifier = None
    fake_pipeline.search_candidates.return_value = pd.DataFrame()
    fake_pipeline.rules = LogicGates(embed_engine=fake_pipeline.embedder)

    audit = run_sku_audit(
        sku_name="Plain Item",
        domain="market",
        task="matcher",
        pipeline=fake_pipeline,
    )
    assert audit["final_output"]["matched_catalog_name"] == "Plain Item"
    assert audit["final_output"]["score"] == 1.0
    assert "stage7_template_enrichment" in audit
