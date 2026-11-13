"""Fast offline unit tests for TextPipeline, NEREngine dict/fallback paths, and rules_engine.loader."""

from unittest.mock import MagicMock, patch

import pandas as pd

from engine.nlp.ner_engine import NEREngine
from engine.nlp.text_cleaner import TextPipeline
from engine.rules_engine import loader


def test_text_pipeline_multipack_units_weights_and_unicode():
    # Multipack multiplication (forward & reverse) and unit normalization
    assert " ".join(TextPipeline.standardize_units("Sprite 4 x 250ml Can").split()) == (
        "sprite 1000ml can"
    )
    assert TextPipeline.standardize_units("Biscuit 100g x 3") == "biscuit 300g"
    assert TextPipeline.standardize_units("Milk 1.5L Bottle") == "milk 1500ml bottle"
    assert TextPipeline.standardize_units("Tea Pack of 6 (12 pieces)") == "tea 6pc (12pc)"
    # Range stripping
    assert TextPipeline.standardize_units("Potato 100g - 200g") == "potato"

    # normalize_final preserves Sinhala/Tamil script while removing numeric SKU noise
    normalized = TextPipeline.normalize_final("SKU 12345 චිකන් රයිස් 99887!")
    assert "චිකන් රයිස්" in normalized
    assert "99887" not in normalized

    # strip_weights & prep_for_ner
    assert TextPipeline.strip_weights("Anchor Butter 200g") == "anchor butter"
    assert TextPipeline.prep_for_ner("Anchor Butter 200g") == "Product: Anchor Butter"

    # build_ner_input skips null-like strings
    assert TextPipeline.build_ner_input("Rice", "nan", "<NA>") == "Rice"
    assert TextPipeline.build_ner_input("Rice", "Spicy", "Main") == "Rice Spicy Main"


def test_ner_engine_dict_priority_weak_brands_and_layer2_skip():
    brands_df = pd.DataFrame(
        [
            {"Brand Name": "Kotmale", "Aliases": "kotmale dairy", "Is_Weak": 0},
            {"Brand Name": "Fresh", "Aliases": "", "Is_Weak": 1},
            {"Brand Name": "Anchor", "Aliases": "anchr", "Is_Weak": 0},
        ]
    )
    mock_gliner = MagicMock(spec=["predict_entities"])
    ner = NEREngine(brands_df, domain="market", shared_model=mock_gliner)

    # Strong brand wins when both strong and weak appear; skips GLiNER Layer 2
    res_strong = ner.extract_entities("kotmale dairy fresh milk")
    assert res_strong["brand"] == {"kotmale"}
    mock_gliner.predict_entities.assert_not_called()

    # Weak brand falls back into brand only when no strong brand matched
    res_weak = ner.extract_entities("fresh farm tomatoes")
    assert res_weak["brand"] == {"fresh"}

    # Batch extract runs Layer 2 ONLY on SKUs where Layer 1 missed completely
    mock_gliner.predict_entities.return_value = [{"label": "brand", "text": "Maliban"}]
    batch_res = ner.batch_extract_entities(["Anchor Butter", "Unknown Cookie Brand"])
    assert batch_res[0]["brand"] == {"anchor"}
    assert batch_res[1]["brand"] == {"maliban"}
    mock_gliner.predict_entities.assert_called_once()


def test_ner_engine_food_domain_collects_all_flavors_and_fuzzy_fallback():
    flavors_df = pd.DataFrame(
        [
            {
                "Flavor Name": "Chicken",
                "Aliases": "chk",
                "Is_Meat": 1,
                "Is_Vegetable": 0,
                "Is_Seafood": 0,
            },
            {
                "Flavor Name": "Prawn",
                "Aliases": "shrimp",
                "Is_Meat": 0,
                "Is_Vegetable": 0,
                "Is_Seafood": 1,
            },
        ]
    )
    ner = NEREngine(flavors_df, domain="food", shared_model=MagicMock())
    res = ner.extract_entities("Chk and Shrimp Noodles")
    assert res["flavor"] == {"chicken", "prawn"}
    assert ner.meat_flavors == {"chicken"}
    assert ner.seafood_flavors == {"prawn"}


def test_rules_loader_priority_sorting_and_malformed_redis_resilience():
    prev_state = (loader._RULES_CACHE, loader._RULES_VERSION)
    try:
        loader.set_rules_from_dicts(
            [
                {
                    "rule_id": "dom_low",
                    "domain": "food",
                    "priority": 50,
                    "description": "d50",
                    "reasoning": "r50",
                },
                {
                    "rule_id": "shared_high",
                    "domain": "shared",
                    "priority": 10,
                    "description": "s10",
                    "reasoning": "rs10",
                },
                {
                    "rule_id": "dom_top",
                    "domain": "food",
                    "priority": 5,
                    "description": "d5",
                    "reasoning": "r5",
                },
            ]
        )
        rules = loader.get_rules("food")
        assert [r.rule_id for r in rules] == ["dom_top", "shared_high", "dom_low"]

        snap = loader.snapshot_rules()
        # Malformed JSON from Redis must preserve the existing valid in-memory cache
        fake_redis = MagicMock()
        fake_redis.get.return_value = "{not-valid-json"
        with patch.object(loader, "_redis", return_value=fake_redis):
            loader.load_rules_from_cache()

        assert [r.rule_id for r in loader.get_rules("food")] == [
            "dom_top",
            "shared_high",
            "dom_low",
        ]
        assert loader.snapshot_rules().version == snap.version

        # Uninitialized cache + malformed JSON initializes cleanly to empty
        loader._RULES_CACHE = None
        loader._RULES_VERSION = "none"
        with patch.object(loader, "_redis", return_value=fake_redis):
            loader.load_rules_from_cache()
        assert loader._RULES_CACHE == {}
    finally:
        loader._RULES_CACHE, loader._RULES_VERSION = prev_state
        loader._cached_rules.cache_clear()
