import pytest

from backend.app.services.persistence import build_processed_sku_rows
from engine.classification.tagger import compute_weighted_confidence


def test_compute_weighted_confidence_all_tags():
    # Kinder Bueno: BT 83%, GK 92%, Region 100%
    # 0.5 * 0.83 + 0.3 * 0.92 + 0.2 * 1.0 = 0.415 + 0.276 + 0.20 = 0.891
    conf = compute_weighted_confidence(0.83, 0.92, 1.00)
    assert conf == pytest.approx(0.891, abs=1e-3)


def test_compute_weighted_confidence_missing_region():
    # Only BT and GK present (weights 0.5 and 0.3, normalized to 0.5/0.8 = 0.625, 0.3/0.8 = 0.375)
    # 0.625 * 0.80 + 0.375 * 0.90 = 0.50 + 0.3375 = 0.8375
    conf = compute_weighted_confidence(0.80, 0.90, None, has_region=False)
    assert conf == pytest.approx(0.8375, abs=1e-3)


def test_compute_weighted_confidence_only_bt():
    conf = compute_weighted_confidence(0.85, None, None, has_gk=False, has_region=False)
    assert conf == pytest.approx(0.85, abs=1e-3)


def test_build_processed_sku_rows_matcher():
    # Matcher SKU: tag confidences must be None, confidence must equal score
    rows = build_processed_sku_rows(
        job_id="job-m",
        domain="food",
        task="pipeline",
        input_skus=[{"name": "Cojo Cojo Mango Juice Drink 320ml"}],
        results=[
            {
                "pipeline_source": "Matcher",
                "matched_catalog_name": "Mango Drink",
                "score": 0.94,
                "bt_confidence": 0.94,  # Even if upstream sent 0.94, persistence enforces None
                "gk_confidence": 0.94,
                "region_confidence": 0.94,
                "suggested_bt": "Fruit Juice",
            }
        ],
        rule_version="v1",
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["match_source"] == "Matcher"
    assert r["match_score"] == 0.94
    assert r["confidence"] == 0.94
    assert r["bt_confidence"] is None
    assert r["gk_confidence"] is None
    assert r["region_confidence"] is None


def test_build_processed_sku_rows_classifier_escalated():
    # Escalated Classifier SKU: preserves match_score from candidate, computes weighted average
    rows = build_processed_sku_rows(
        job_id="job-c",
        domain="food",
        task="pipeline",
        input_skus=[{"name": "Kinder Bueno Milk 43g"}],
        results=[
            {
                "pipeline_source": "Classifier",
                "matched_catalog_name": "Kinder Bueno",
                "score": 0.45,
                "bt_confidence": 0.83,
                "gk_confidence": 0.92,
                "region_confidence": 1.00,
                "suggested_bt": "Chocolate",
                "suggested_gk": "Wafer",
                "suggested_region": "Confectionery",
            }
        ],
        rule_version="v1",
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["match_source"] == "Classifier"
    assert r["match_score"] == 0.45
    assert r["matched_catalog_name"] == "Kinder Bueno"
    assert r["bt_confidence"] == 0.83
    assert r["gk_confidence"] == 0.92
    assert r["region_confidence"] == 1.00
    assert r["confidence"] == pytest.approx(0.891, abs=1e-3)
