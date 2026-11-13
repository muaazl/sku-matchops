"""Unit tests for template-based tag enrichment behavior in engine.pipeline.processor."""

from unittest.mock import patch

from engine import config
from engine.pipeline.processor import _apply_template_tag_enrichment


def test_template_enrichment_skips_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", False)
    results = [{"suggested_bt": "Old BT", "suggested_gk": "Old GK", "bt_status": "REVIEW"}]

    with patch("engine.pipeline.processor.suggest_tags_from_template") as mock_suggest:
        _apply_template_tag_enrichment(
            [{"name": "Koththu Chicken"}], results, domain="food", mode="classifier"
        )

    mock_suggest.assert_not_called()
    assert results[0]["suggested_bt"] == "Old BT"


def test_template_enrichment_skips_high_confidence_and_auto(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", True)
    results = [
        {"status": "High Confidence", "suggested_bt": "Pizza"},
        {"bt_status": "AUTO", "suggested_bt": "Burger"},
    ]

    with patch("engine.pipeline.processor.suggest_tags_from_template") as mock_suggest:
        _apply_template_tag_enrichment(
            [{"name": "Chicken Pizza"}, {"name": "Beef Burger"}],
            results,
            domain="food",
            mode="classifier",
        )

    mock_suggest.assert_not_called()
    assert results[0]["suggested_bt"] == "Pizza"
    assert results[1]["suggested_bt"] == "Burger"


def test_template_enrichment_updates_classifier_and_merges_gks(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", True)
    results = [
        {
            "bt_status": "LOW",
            "suggested_bt": "Rice",
            "bt_confidence": 0.40,
            "suggested_gk": "Rice",
            "gk_confidence": 0.40,
        }
    ]

    with patch(
        "engine.pipeline.processor.suggest_tags_from_template",
        return_value={
            "matched": True,
            "suggested_bt": "Kotthu",
            "suggested_gk": ["Kotthu", "Rice", "Street Food"],
        },
    ):
        _apply_template_tag_enrichment(
            [{"name": "Chicken Kotthu"}], results, domain="food", mode="classifier"
        )

    assert results[0]["suggested_bt"] == "Kotthu"
    assert results[0]["bt_status"] == "AUTO"
    assert results[0]["bt_confidence"] == 0.95
    assert results[0]["suggested_gk"] == "Rice, Kotthu, Street Food"
    assert results[0]["gk_status"] == "AUTO"
    assert results[0]["gk_confidence"] == 0.95


def test_template_enrichment_sets_high_confidence_in_pipeline_mode(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", True)
    results = [{"status": "Low Confidence", "bt_status": "LOW", "bt_confidence": 0.98}]

    with patch(
        "engine.pipeline.processor.suggest_tags_from_template",
        return_value={"matched": True, "suggested_bt": "Noodles", "suggested_gk": ["Noodles"]},
    ):
        _apply_template_tag_enrichment(
            [{"name": "Spicy Noodles"}], results, domain="food", mode="pipeline"
        )

    assert results[0]["suggested_bt"] == "Noodles"
    assert results[0]["bt_status"] == "High Confidence"
    # Existing higher confidence is preserved
    assert results[0]["bt_confidence"] == 0.98
    assert results[0]["gk_status"] == "High Confidence"
