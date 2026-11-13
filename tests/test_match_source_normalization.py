"""Tests for match_source normalization and attribution logic."""

from backend.app.services.persistence import build_processed_sku_rows


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


def test_jobs_dashboard_stats_normalizes_match_source(pg):
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from backend.main import app

    with pg.begin() as conn:
        conn.execute(text("TRUNCATE processed_skus, jobs, batches CASCADE"))
        conn.execute(
            text(
                "INSERT INTO jobs (id, domain, type, status) VALUES ('j_src_1', 'food', 'pipeline', 'completed')"
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO processed_skus (id, batch_id, sku_index, sku_name, domain, match_source, matched_catalog_name, bt, bt_confidence, confidence)
                VALUES
                    ('ps_src_1', 'j_src_1', 0, 'SKU A', 'food', 'Catalogue', 'Cat A', '', 0.0, 0.90),
                    ('ps_src_2', 'j_src_1', 1, 'SKU B', 'food', 'unknown', '', 'Kotthu', 0.85, 0.85),
                    ('ps_src_3', 'j_src_1', 2, 'SKU C', 'food', '', '', '', 0.0, 0.50)
                """
            )
        )

    client = TestClient(app)
    try:
        res = client.get("/jobs/dashboard-stats", params={"domain": "food", "timeframe": "30d"})
        assert res.status_code == 200
        sources = {item["source"]: item["count"] for item in res.json()["matchSourceDistribution"]}
        assert sources["Matcher"] == 2
        assert sources["Classifier"] == 1
    finally:
        with pg.begin() as conn:
            conn.execute(text("TRUNCATE processed_skus, jobs, batches CASCADE"))


def test_history_match_source_filter_executes_real_sql(pg):
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from backend.main import app

    with pg.begin() as conn:
        conn.execute(text("TRUNCATE processed_skus, jobs, batches CASCADE"))
        conn.execute(
            text(
                "INSERT INTO jobs (id, domain, type, status) VALUES ('j_src_2', 'food', 'pipeline', 'completed')"
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO processed_skus (id, batch_id, sku_index, sku_name, domain, match_source, matched_catalog_name, bt, bt_confidence, confidence)
                VALUES
                    ('ps_src_4', 'j_src_2', 0, 'Explicit Catalogue', 'food', 'Catalogue', 'Burger', '', 0.0, 0.95),
                    ('ps_src_5', 'j_src_2', 1, 'Inferred Classifier', 'food', 'unknown', '', 'Noodles', 0.80, 0.80),
                    ('ps_src_6', 'j_src_2', 2, 'Explicit Classifier', 'food', 'Classifier', '', 'Rice', 0.88, 0.88)
                """
            )
        )

    client = TestClient(app)
    try:
        m_res = client.get("/processed-skus", params={"domain": "food", "match_source": "matcher"})
        assert m_res.status_code == 200
        assert [row["sku_name"] for row in m_res.json()] == ["Explicit Catalogue"]

        c_res = client.get(
            "/processed-skus", params={"domain": "food", "match_source": "classifier"}
        )
        assert c_res.status_code == 200
        assert {row["sku_name"] for row in c_res.json()} == {
            "Inferred Classifier",
            "Explicit Classifier",
        }
    finally:
        with pg.begin() as conn:
            conn.execute(text("TRUNCATE processed_skus, jobs, batches CASCADE"))
