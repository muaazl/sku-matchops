"""Fast, deterministic unit tests for processor.process_request and audit_engine.run_sku_audit."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.core import config
from engine.pipelines.audit_pipeline import run_sku_audit
from engine.pipelines.request_pipeline import process_request


def _make_food_ner(flavor_sets: list[set[str]] | None = None):
    ner = MagicMock()
    ner.seafood_flavors = {"fish", "prawn"}
    ner.meat_flavors = {"chicken", "beef", "egg", "mixed"}
    ner.vegetable_flavors = {"carrot", "cabbage", "leek"}
    if flavor_sets is not None:
        ner.batch_extract_entities.return_value = [{"flavor": fs} for fs in flavor_sets]
    return ner


def _make_embed_engine(n: int = 1):
    embed = MagicMock()
    embed.cross_session = None
    embed.embed_weighted_sku.return_value = {
        "dense": np.ones((n, 4), dtype=np.float32),
        "sparse": [{} for _ in range(n)],
    }
    return embed


def test_process_request_rejects_unknown_task():
    with patch(
        "engine.pipelines.request_pipeline.check_models_loaded", return_value={"food": True}
    ):
        with pytest.raises(ValueError, match="Unknown task"):
            process_request(task="invalid_task", domain="market", skus=[{"name": "Milk"}])


def test_process_request_honors_cancellation_callback():
    fake_matcher = MagicMock()
    fake_matcher.process_inputs.side_effect = lambda _df, progress_callback=None: progress_callback(
        10.0, "step"
    )
    with (
        patch(
            "engine.pipelines.request_pipeline.check_models_loaded", return_value={"market": True}
        ),
        patch("engine.pipelines.request_pipeline.get_pipeline", return_value=fake_matcher),
        pytest.raises(InterruptedError, match="cancelled"),
    ):
        process_request(
            task="matcher",
            domain="market",
            skus=[{"name": "Anchor Butter 200g"}],
            is_cancelled=lambda: True,
        )


def test_process_request_food_protein_prioritization_and_short_df_alignment(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", False)
    skus = [
        {"name": "Devilled Fish Set Menu", "price": 1000.0},
        {"name": "Chicken Special Platter", "price": 1000.0},
        {"name": "Chicken Egg Rice", "price": 950.0},
        {"name": "Egg Veggie Noodles", "price": 900.0},
        {"name": "Mixed Grill Platter", "price": 1200.0},
        {"name": "Meat Grill", "price": 1200.0},
        {"name": "Surf Turf", "price": 1500.0},
        {"name": "Garden Bowl", "price": 700.0},
    ]
    ner = _make_food_ner(
        [
            {"fish"},  # seafood only
            {"chicken", "mixed", "carrot"},  # meta 'mixed' does not inflate meat_count
            {"chicken", "egg"},  # egg + real meat -> no mixed
            {"egg", "carrot", "leek"},  # egg primary protein with veggies -> no mixed
            {"chicken", "beef"},  # already has 'mix' in name -> no duplicate suffix
            {"chicken", "beef"},  # 2 land meats -> appends mixed
            {"chicken", "prawn"},  # meat + seafood -> appends mixed
            {"carrot", "cabbage"},  # veg only -> do not mutate input name
        ]
    )
    seen_inputs = []

    class _FakeMatcher:
        def process_inputs(self, df, progress_callback=None):
            seen_inputs.extend(df["SKU"].tolist())
            # Deliberately return fewer rows than inputs to verify positional padding
            return pd.DataFrame(
                [
                    {
                        "Matched Catalog Name": f"Catalog {i}",
                        "Final Score": 0.91,
                        "Status": "High Confidence",
                        "Logic Notes": "Exact",
                        "BasicType": "Rice",
                        "GenericKeywords": "Food",
                        "Region": "Asian",
                    }
                    for i in range(len(df) - 1)
                ]
            )

    with (
        patch("engine.pipelines.request_pipeline.check_models_loaded", return_value={"food": True}),
        patch(
            "engine.pipelines.request_pipeline._get_shared_models",
            return_value=(_make_embed_engine(), ner),
        ),
        patch("engine.pipelines.request_pipeline.get_ner_engine", return_value=ner),
        patch("engine.pipelines.request_pipeline.get_pipeline", return_value=_FakeMatcher()),
    ):
        res = process_request(task="matcher", domain="food", skus=skus)

    assert seen_inputs == [
        "Devilled Fish Set Menu",
        "Chicken Special Platter",
        "Chicken Egg Rice",
        "Egg Veggie Noodles",
        "Mixed Grill Platter",
        "Meat Grill mixed",
        "Surf Turf mixed",
        "Garden Bowl",
    ]
    assert res["total"] == len(skus)
    assert res["results"][0]["matched_catalog_name"] == "Catalog 0"
    assert res["results"][-1]["matched_catalog_name"] == ""
    assert res["results"][-1]["status"] == "Low / Rejected"
    assert res["results"][-1]["logic_notes"] == "No candidate found"


def test_pipeline_escalation_decision_and_rules_application(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", False)
    skus = [
        {"name": "Strong Match Dish", "price": 1000.0},
        {"name": "Weak Match Escalates To Classifier", "price": 1100.0},
        {"name": "Weak Match Kept By Matcher", "price": 1200.0},
        {"name": "Unmatched Missing Row", "price": 500.0},
    ]

    match_rows = pd.DataFrame(
        [
            {
                "Matched Catalog Name": "Exact Dish",
                "Final Score": 0.94,
                "Status": "High Confidence",
                "Logic Notes": "Exact match",
                "BasicType": "Rice",
                "GenericKeywords": "Rice",
                "Region": "Local",
            },
            {
                "Matched Catalog Name": "Poor Guess",
                "Final Score": 0.35,
                "Status": "Low Confidence",
                "Logic Notes": "Weak vector candidate",
                "BasicType": "Noodles",
                "GenericKeywords": "Noodles",
                "Region": "Asian",
            },
            {
                "Matched Catalog Name": "Decent Fuzzy",
                "Final Score": 0.58,
                "Status": "Low Confidence",
                "Logic Notes": "Fuzzy fallback",
                "BasicType": "Curry",
                "GenericKeywords": "Curry",
                "Region": "Sri Lankan",
            },
        ]
    )

    class _FakeMatcher:
        def process_inputs(self, df, progress_callback=None):
            if progress_callback:
                progress_callback(50.0, "Halfway")
            return match_rows

    def _fake_tag_all_skus(sku_names, **_kwargs):
        # Escalated indices are [1, 2, 3]
        return [
            {
                "suggested_bt": "Kottu",
                "bt_confidence": 0.90,
                "bt_status": "AUTO",
                "suggested_gk": "Kottu, Street Food",
                "gk_confidence": 0.85,
                "gk_status": "AUTO",
                "suggested_region": "Sri Lankan",
                "region_confidence": 0.80,
                "region_status": "AUTO",
                "model": "arcface",
                "bt_model": "arcface",
                "reasoning": "Classifier: BT=Kottu",
            },
            {
                "suggested_bt": "Soup",
                "bt_confidence": 0.40,
                "bt_status": "LOW",
                "suggested_gk": "Soup",
                "gk_confidence": 0.40,
                "gk_status": "LOW",
                "suggested_region": "Asian",
                "region_confidence": 0.40,
                "region_status": "LOW",
                "model": "logreg",
                "bt_model": "logreg",
                "reasoning": "Classifier: BT=Soup",
            },
            {
                "suggested_bt": "Snacks",
                "bt_confidence": 0.72,
                "bt_status": "REVIEW",
                "suggested_gk": "Snacks",
                "gk_confidence": 0.70,
                "gk_status": "REVIEW",
                "suggested_region": "Local",
                "region_confidence": 0.70,
                "region_status": "REVIEW",
                "model": "logreg",
                "bt_model": "logreg",
                "reasoning": "Classifier: BT=Snacks",
            },
        ]

    def _fake_rules(record):
        updated = dict(record)
        if record["bt"] == "Kottu":
            updated["gk"] = [*record["gk"], "Popular"]
            updated["rules_applied"] = [{"rule_id": "rule_kottu", "description": "Add Popular"}]
        else:
            updated["rules_applied"] = []
        return updated

    ner = _make_food_ner([set(), set(), set(), set()])
    fake_clf = MagicMock(ner_engine=ner)

    with (
        patch("engine.pipelines.request_pipeline.check_models_loaded", return_value={"food": True}),
        patch(
            "engine.pipelines.request_pipeline._get_shared_models",
            return_value=(_make_embed_engine(n=3), ner),
        ),
        patch("engine.pipelines.request_pipeline.get_ner_engine", return_value=ner),
        patch("engine.pipelines.request_pipeline.get_pipeline", return_value=_FakeMatcher()),
        patch("engine.pipelines.request_pipeline.get_classifier", return_value=fake_clf),
        patch("engine.pipelines.request_pipeline._get_vector_store", return_value=MagicMock()),
        patch("engine.pipelines.request_pipeline.tag_all_skus", side_effect=_fake_tag_all_skus),
        patch("engine.pipelines.request_pipeline.run_rules_engine", side_effect=_fake_rules),
    ):
        res = process_request(task="pipeline", domain="food", skus=skus)

    assert res["escalated_count"] == 3
    out = res["results"]

    # SKU 0: High confidence matcher row skips escalation
    assert out[0]["escalated"] is False
    assert out[0]["pipeline_source"] == "Matcher"
    assert out[0]["suggested_bt"] == "Rice"

    # SKU 1: Escalated, classifier (0.90) > matcher (0.35), rule applied
    assert out[1]["escalated"] is True
    assert out[1]["pipeline_source"] == "Classifier"
    assert out[1]["suggested_bt"] == "Kottu"
    assert (
        "Escalated from matcher [Weak vector candidate] -> Classifier: BT=Kottu"
        == out[1]["logic_notes"]
    )
    assert out[1]["suggested_gk"] == "Kottu, Street Food, Popular"
    assert json.loads(out[1]["rules_applied"])[0]["rule_id"] == "rule_kottu"

    # SKU 2: Escalated, classifier (0.40) <= matcher (0.58), matcher kept
    assert out[2]["escalated"] is True
    assert out[2]["pipeline_source"] == "Matcher"
    assert out[2]["suggested_bt"] == "Curry"
    assert out[2]["bt_confidence"] is None

    # SKU 3: Missing matcher row padded as 0.0 and promoted by classifier (0.72)
    assert out[3]["escalated"] is True
    assert out[3]["pipeline_source"] == "Classifier"
    assert out[3]["suggested_bt"] == "Snacks"


def test_run_sku_audit_food_protein_prioritization_uses_real_code():
    ner = _make_food_ner()
    fake_matcher = SimpleNamespace(
        ner=ner,
        embedder=_make_embed_engine(n=1),
        classifier=None,
        raw_catalog=pd.DataFrame(
            [{"Name": "Devilled Platter", "clean_text": "devilled platter", "Price": 1500.0}]
        ),
        exact_match_map={},
        token_sorted_map={},
        search_candidates=lambda *args, **kwargs: pd.DataFrame(),
    )

    with patch("engine.pipelines.audit_pipeline.run_rules_engine", side_effect=lambda rec: rec):
        ner.extract_entities.return_value = {"flavor": {"chicken", "prawn"}}
        res_mixed = run_sku_audit(
            sku_name="Surf Platter",
            domain="food",
            task="matcher",
            price=1500.0,
            pipeline=fake_matcher,
        )

        ner.extract_entities.return_value = {"flavor": {"chicken", "egg"}}
        res_egg_meat = run_sku_audit(
            sku_name="Chicken Egg Rice",
            domain="food",
            task="matcher",
            price=1000.0,
            pipeline=fake_matcher,
        )

    assert res_mixed["stage1_nlp"]["food_suffix_added"] == "mixed"
    assert res_mixed["stage1_nlp"]["food_domain_effective_sku"] == "Surf Platter mixed"
    assert res_egg_meat["stage1_nlp"]["food_suffix_added"] == ""
    assert res_egg_meat["stage1_nlp"]["food_domain_effective_sku"] == "Chicken Egg Rice"


def test_pipeline_reranking_stage_progress_and_eta():
    events: list[tuple[str, float, int | None]] = []

    class _SteppingMatcher:
        def process_inputs(self, df, progress_callback=None):
            if progress_callback:
                progress_callback(10.0, "Embedded")
                progress_callback(30.0, "Searching")
                progress_callback(50.0, "Reranking 0 of 2...")
                progress_callback(62.5, "Reranking 1 of 2...")
                progress_callback(75.0, "Reranking 2 of 2...")
                progress_callback(100.0, "Done")
            return pd.DataFrame(
                [
                    {
                        "Matched Catalog Name": "Exact Dish",
                        "Final Score": 0.94,
                        "Status": "High Confidence",
                        "Logic Notes": "Exact match",
                        "BasicType": "Rice",
                        "GenericKeywords": "Rice",
                        "Region": "Local",
                    }
                ]
            )

    ner = _make_food_ner([set()])
    with (
        patch("engine.pipelines.request_pipeline.check_models_loaded", return_value={"food": True}),
        patch(
            "engine.pipelines.request_pipeline._get_shared_models",
            return_value=(_make_embed_engine(n=1), ner),
        ),
        patch("engine.pipelines.request_pipeline.get_ner_engine", return_value=ner),
        patch("engine.pipelines.request_pipeline.get_pipeline", return_value=_SteppingMatcher()),
        patch("engine.pipelines.request_pipeline.run_rules_engine", side_effect=lambda rec: rec),
    ):
        process_request(
            task="pipeline",
            domain="food",
            skus=[{"name": "Dish", "price": 500.0}],
            progress_callback=lambda s, p, e: events.append((s, p, e)),
        )

    stages_by_pct = {pct: stage for stage, pct, _ in events}
    assert stages_by_pct[21.0] == "vector_search"
    assert stages_by_pct[35.0] == "reranking"
    assert stages_by_pct[43.75] == "reranking"
    assert stages_by_pct[52.5] == "classifying"
    assert stages_by_pct[100.0] == "done"


def test_resolve_bt_consensus_and_escalation_guards(monkeypatch):
    from engine.preprocessing.weight_preprocessor import resolve_bt_consensus_indices

    catalog = [
        {"Name": "Butter Chicken", "basictype": "Burger"},
        {"Name": "Butter Chicken", "basictype": "Masala Curry"},
        {"Name": "Butter Chicken", "basictype": "Masala Curry"},
        {"Name": "Butter Chicken", "basictype": "Masala Curry"},
    ]
    bts_lower = ["burger", "masala curry", "masala curry", "masala curry"]
    # Majority-vote BasicType wins over index-0 outlier even if conf_bt is None or matches the 1-row outlier
    assert resolve_bt_consensus_indices(
        catalog, [0, 1, 2, 3], conf_bt=None, cat_bts_lower=bts_lower
    ) == [
        1,
        2,
        3,
    ]
    assert resolve_bt_consensus_indices(
        catalog, [0, 1, 2, 3], conf_bt="Burger", cat_bts_lower=bts_lower
    ) == [1, 2, 3]

    # Escalation guard: empty classifier BT or confirmed Matcher BasicType Match must keep Matcher tags
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", False)

    class _TwoItemMatcher:
        def process_inputs(self, df, progress_callback=None):
            return pd.DataFrame(
                [
                    {
                        "Matched Catalog Name": "Chicken Kebab",
                        "Final Score": 0.294,
                        "Status": "Low / Rejected",
                        "Logic Notes": "No BT filter",
                        "BasicType": "Kebab",
                        "GenericKeywords": "Kebab, Chicken",
                        "Region": "Indian",
                    },
                    {
                        "Matched Catalog Name": "Special Noodles",
                        "Final Score": 0.57,
                        "Status": "Medium Confidence",
                        "Logic Notes": "BT-Filtered (Fried Noodles) | BasicType Match (Fried Noodles)",
                        "BasicType": "Fried Noodles",
                        "GenericKeywords": "Noodles",
                        "Region": "Chinese",
                    },
                ]
            )

    fake_clf_out = [
        {
            "suggested_bt": "",
            "bt_confidence": 0.453,
            "bt_status": "LOW",
            "suggested_gk": "",
            "gk_confidence": 0.0,
            "gk_status": "LOW",
            "suggested_region": "North Indian",
            "region_confidence": 0.294,
            "region_status": "LOW",
            "reasoning": "Classifier: BT=''",
        },
        {
            "suggested_bt": "Burger",
            "bt_confidence": 0.961,
            "bt_status": "AUTO",
            "suggested_gk": "Burger",
            "gk_confidence": 0.0,
            "gk_status": "LOW",
            "suggested_region": "Western",
            "region_confidence": 1.0,
            "region_status": "AUTO",
            "reasoning": "Classifier: BT='Burger'",
        },
    ]

    ner = _make_food_ner([set(), set()])
    with (
        patch("engine.pipelines.request_pipeline.check_models_loaded", return_value={"food": True}),
        patch(
            "engine.pipelines.request_pipeline._get_shared_models",
            return_value=(_make_embed_engine(n=2), ner),
        ),
        patch("engine.pipelines.request_pipeline.get_ner_engine", return_value=ner),
        patch("engine.pipelines.request_pipeline.get_pipeline", return_value=_TwoItemMatcher()),
        patch(
            "engine.pipelines.request_pipeline.get_classifier",
            return_value=SimpleNamespace(domain="food"),
        ),
        patch("engine.pipelines.request_pipeline._get_vector_store", return_value=MagicMock()),
        patch("engine.pipelines.request_pipeline.tag_all_skus", return_value=fake_clf_out),
        patch("engine.pipelines.request_pipeline.run_rules_engine", side_effect=lambda rec: rec),
    ):
        res = process_request(
            task="pipeline",
            domain="food",
            skus=[
                {"name": "Bangalore Styled Oil Kebab", "price": 1500.0},
                {"name": "My Burger Special Noodles", "price": 1200.0},
            ],
        )["results"]

    assert res[0]["pipeline_source"] == "Matcher"
    assert res[0]["suggested_bt"] == "Kebab"
    assert res[1]["pipeline_source"] == "Matcher"
    assert res[1]["suggested_bt"] == "Fried Noodles"


def test_pipeline_selective_gk_preservation_when_classifier_agrees_on_bt(monkeypatch):
    """When Classifier wins escalation and predicts the same BasicType as Matcher's candidate,
    preserve Matcher's curated catalog GKs while adopting Classifier's BT/Region confidence."""
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", False)
    skus = [
        {"name": "Seafood Fried Rice Special", "price": 1200.0},
        {"name": "Spicy Kottu Platter", "price": 1100.0},
    ]

    match_rows = pd.DataFrame(
        [
            {
                "Matched Catalog Name": "Seafood Fried Rice",
                "Final Score": 0.55,
                "Status": "Medium Confidence",
                "Logic Notes": "Candidate match",
                "BasicType": "Fried Rice",
                "GenericKeywords": "Fried Rice, Seafood",
                "Region": "Chinese",
            },
            {
                "Matched Catalog Name": "Spicy Noodles",
                "Final Score": 0.40,
                "Status": "Low / Rejected",
                "Logic Notes": "Weak match",
                "BasicType": "Noodles",
                "GenericKeywords": "Noodles",
                "Region": "Chinese",
            },
        ]
    )

    class _FakeMatcher:
        def process_inputs(self, df, progress_callback=None):
            return match_rows

    def _fake_tag_all_skus(sku_names, **_kwargs):
        return [
            {
                "suggested_bt": "Fried Rice",
                "bt_confidence": 0.92,
                "bt_status": "AUTO",
                "suggested_gk": "Fried Rice, Rice, Chinese Rice, Wok, Egg Fried Rice, Mixed Fried Rice",
                "gk_confidence": 0.80,
                "gk_status": "AUTO",
                "suggested_region": "Chinese",
                "region_confidence": 0.95,
                "region_status": "AUTO",
                "model": "arcface",
                "bt_model": "arcface",
                "reasoning": "Classifier (arcface): BT='Fried Rice'",
            },
            {
                "suggested_bt": "Kottu",
                "bt_confidence": 0.91,
                "bt_status": "AUTO",
                "suggested_gk": "Kottu, Street Food",
                "gk_confidence": 0.85,
                "gk_status": "AUTO",
                "suggested_region": "Sri Lankan",
                "region_confidence": 0.90,
                "region_status": "AUTO",
                "model": "arcface",
                "bt_model": "arcface",
                "reasoning": "Classifier (arcface): BT='Kottu'",
            },
        ]

    ner = _make_food_ner([set(), set()])
    with (
        patch("engine.pipelines.request_pipeline.check_models_loaded", return_value={"food": True}),
        patch(
            "engine.pipelines.request_pipeline._get_shared_models",
            return_value=(_make_embed_engine(n=2), ner),
        ),
        patch("engine.pipelines.request_pipeline.get_ner_engine", return_value=ner),
        patch("engine.pipelines.request_pipeline.get_pipeline", return_value=_FakeMatcher()),
        patch(
            "engine.pipelines.request_pipeline.get_classifier",
            return_value=SimpleNamespace(domain="food"),
        ),
        patch("engine.pipelines.request_pipeline._get_vector_store", return_value=MagicMock()),
        patch("engine.pipelines.request_pipeline.tag_all_skus", side_effect=_fake_tag_all_skus),
        patch("engine.pipelines.request_pipeline.run_rules_engine", side_effect=lambda rec: rec),
    ):
        out = process_request(task="pipeline", domain="food", skus=skus)["results"]

    # SKU 0: Classifier won escalation AND predicted same BT ("Fried Rice") -> keeps Matcher's curated GKs
    assert out[0]["pipeline_source"] == "Classifier"
    assert out[0]["suggested_bt"] == "Fried Rice"
    assert out[0]["suggested_gk"] == "Fried Rice, Seafood"

    # SKU 1: Classifier won escalation AND changed BT ("Noodles" -> "Kottu") -> uses Classifier's GKs
    assert out[1]["pipeline_source"] == "Classifier"
    assert out[1]["suggested_bt"] == "Kottu"
    assert out[1]["suggested_gk"] == "Kottu, Street Food"
