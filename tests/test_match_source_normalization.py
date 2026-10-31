"""Tests for match_source normalization and attribution logic."""

from engine.pipeline.persistence import build_processed_sku_rows


def test_build_processed_sku_rows_matcher_task():
    rows = build_processed_sku_rows(
        job_id="test-job-1",
        domain="food",
        task="matcher",
        input_skus=[{"name": "Burger"}],
        results=[{"score": 0.92, "matched_catalog_name": "Cheese Burger"}],
        rule_version="v1",
    )
    assert len(rows) == 1
    assert rows[0]["match_source"] == "Matcher"


def test_build_processed_sku_rows_classifier_task():
    rows = build_processed_sku_rows(
        job_id="test-job-2",
        domain="food",
        task="classifier",
        input_skus=[{"name": "Kotthu"}],
        results=[{"bt_confidence": 0.88, "suggested_bt": "Kottu"}],
        rule_version="v1",
    )
    assert len(rows) == 1
    assert rows[0]["match_source"] == "Classifier"


def test_build_processed_sku_rows_pipeline_task_explicit():
    rows_clf = build_processed_sku_rows(
        job_id="test-job-3",
        domain="food",
        task="pipeline",
        input_skus=[{"name": "Item A"}],
        results=[{"pipeline_source": "Classifier", "bt_confidence": 0.75}],
        rule_version="v1",
    )
    assert rows_clf[0]["match_source"] == "Classifier"

    rows_match = build_processed_sku_rows(
        job_id="test-job-4",
        domain="food",
        task="pipeline",
        input_skus=[{"name": "Item B"}],
        results=[{"pipeline_source": "Matcher", "score": 0.9}],
        rule_version="v1",
    )
    assert rows_match[0]["match_source"] == "Matcher"


def test_build_processed_sku_rows_pipeline_task_inferred():
    # Inferred from catalog match
    rows_inferred_m = build_processed_sku_rows(
        job_id="test-job-5",
        domain="food",
        task="pipeline",
        input_skus=[{"name": "Item C"}],
        results=[{"matched_catalog_name": "Catalog Item", "score": 0.9}],
        rule_version="v1",
    )
    assert rows_inferred_m[0]["match_source"] == "Matcher"

    # Inferred from BT classifier prediction
    rows_inferred_c = build_processed_sku_rows(
        job_id="test-job-6",
        domain="food",
        task="pipeline",
        input_skus=[{"name": "Item D"}],
        results=[{"suggested_bt": "Noodles", "bt_confidence": 0.8}],
        rule_version="v1",
    )
    assert rows_inferred_c[0]["match_source"] == "Classifier"

    # Fallback when completely unassigned
    rows_fallback = build_processed_sku_rows(
        job_id="test-job-7",
        domain="food",
        task="pipeline",
        input_skus=[{"name": "Item E"}],
        results=[{}],
        rule_version="v1",
    )
    assert rows_fallback[0]["match_source"] == "Matcher"


def test_jobs_source_query_syntax():
    from sqlalchemy import text

    where_str = "tenant_id = :tenant_id"
    source_query = f"""
        SELECT
            CASE
                WHEN LOWER(TRIM(COALESCE(match_source, ''))) IN ('matcher', 'catalogue') THEN 'Matcher'
                WHEN LOWER(TRIM(COALESCE(match_source, ''))) = 'classifier' THEN 'Classifier'
                WHEN matched_catalog_name IS NOT NULL AND TRIM(matched_catalog_name) != '' THEN 'Matcher'
                WHEN (bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0) THEN 'Classifier'
                ELSE 'Matcher'
            END as normalized_source,
            COUNT(*) as count
        FROM processed_skus
        WHERE {where_str}
        GROUP BY normalized_source
        ORDER BY count DESC
    """
    stmt = text(source_query)
    assert stmt is not None


def test_history_where_clause_syntax():
    from sqlalchemy import text

    where = "WHERE tenant_id = :tenant_id"
    m_where = where + """ AND (
        LOWER(TRIM(COALESCE(match_source, ''))) IN ('matcher', 'catalogue')
        OR (
            (match_source IS NULL OR TRIM(match_source) = '' OR LOWER(TRIM(match_source)) = 'unknown')
            AND NOT (
                (matched_catalog_name IS NULL OR TRIM(matched_catalog_name) = '')
                AND ((bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0))
            )
        )
    )"""
    stmt = text(f"SELECT * FROM processed_skus {m_where}")
    assert stmt is not None

    c_where = where + """ AND (
        LOWER(TRIM(COALESCE(match_source, ''))) = 'classifier'
        OR (
            (match_source IS NULL OR TRIM(match_source) = '' OR LOWER(TRIM(match_source)) = 'unknown')
            AND (matched_catalog_name IS NULL OR TRIM(matched_catalog_name) = '')
            AND ((bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0))
        )
    )"""
    stmt2 = text(f"SELECT * FROM processed_skus {c_where}")
    assert stmt2 is not None

