from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.main import app

pytestmark = pytest.mark.usefixtures("pg")

client = TestClient(app)


@pytest.fixture
def seeded_data(pg):
    """Seeds isolated PG rows for jobs, batches, processed_skus, and catalog_items."""
    with pg.begin() as conn:
        conn.execute(
            text("TRUNCATE processed_skus, jobs, batches, api_requests, catalog_items CASCADE")
        )
        conn.execute(
            text(
                """
                INSERT INTO batches (id, source, filename, domain, status, total_skus, created_by)
                VALUES ('batch_1', 'upload', 'supermarket.csv', 'market', 'completed', 2, 'alice')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO jobs (id, batch_id, domain, type, status, current_stage, total_items, sheet_name, created_by, input_skus_json)
                VALUES
                    ('job_1', 'batch_1', 'market', 'pipeline', 'completed', 'done', 2, 'supermarket.csv', 'alice',
                     '[{"name": "Anchor Butter 200g", "price": 850.0}]'),
                    ('job_empty', NULL, 'food', 'matcher', 'failed', 'failed', 1, 'menu.csv', 'bob', NULL)
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO processed_skus
                    (id, batch_id, sku_index, rule_version, sku_name, domain, bt, gk_json, category, confidence, match_source, matched_catalog_name)
                VALUES
                    ('ps_1', 'job_1', 0, 'v1', 'Anchor Butter 200g', 'market', 'Butter', '["Dairy", "Butter"]', 'Dairy', 0.94, 'Matcher', 'Anchor Butter 200g'),
                    ('ps_2', 'job_1', 1, 'v1', 'Sunlight Soap 100g', 'market', 'Soap', '["Cleaning"]', 'Household', 0.72, 'Classifier', ''),
                    ('ps_3', 'job_empty', 0, 'v1', 'Chicken Kottu', 'food', 'Kottu', '["Rice", "Street Food"]', '', 0.45, 'Classifier', '')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO catalog_items
                    (domain, name, brand, basictype, generic_keywords, category, price, clean_text, row_hash)
                VALUES
                    ('market', 'Anchor Butter 200g', 'Anchor', 'Butter', 'Dairy, Butter', 'Dairy', 850.0, 'anchor butter 200g', 'h1'),
                    ('market', 'Anchor Milk Powder 400g', 'Anchor', 'Milk Powder', 'Dairy, Milk', 'Dairy', 1100.0, 'anchor milk powder 400g', 'h2'),
                    ('market', 'Sunlight Soap 100g', 'Sunlight', 'Soap', 'Cleaning, Bath', 'Personal Care', 180.0, 'sunlight soap 100g', 'h3')
                """
            )
        )
    yield
    with pg.begin() as conn:
        conn.execute(
            text("TRUNCATE processed_skus, jobs, batches, api_requests, catalog_items CASCADE")
        )


def test_health_endpoint():
    """Verify that GET /health returns status 200 and 'ok'."""
    with patch(
        "backend.services.engine_gateway_service.get_engine_health",
        return_value={
            "status": "ok",
            "loaded_domains": ["market", "food"],
            "all_models_ready": True,
        },
    ):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("loaded_domains") == ["market", "food"]


def test_rules_crud_lifecycle():
    """Test full CRUD lifecycle for Rules Engine via FastAPI endpoints."""
    rule_payload = {
        "rule_id": "test_rule_api_999",
        "domain": "market",
        "priority": 500,
        "description": "Integration test rule for API suite.",
        "reasoning": "Verify rules endpoint creation and updates.",
        "condition_logic": "AND",
        "is_active": 1,
        "conditions": [
            {"condition_group": 1, "condition_type": "sku_contains", "value": "pepsi", "negate": 0}
        ],
        "actions": [{"action_type": "set_bt", "value": "Soft Drink"}],
    }

    with patch("backend.services.rule_service.sync_rules_everywhere"):
        create_resp = client.post("/rules", json=rule_payload)
        assert create_resp.status_code == 200
        rule_id = create_resp.json()["rule_id"]

        try:
            list_resp = client.get("/rules", params={"domain": "market"})
            assert list_resp.status_code == 200
            assert any(r["rule_id"] == rule_id for r in list_resp.json())

            rule_payload["description"] = "Updated integration test rule description."
            update_resp = client.put(f"/rules/{rule_id}", json=rule_payload)
            assert update_resp.status_code == 200

            reorder_resp = client.put("/rules/reorder", json={"ordered_rule_ids": [rule_id]})
            assert reorder_resp.status_code == 200

            test_resp = client.post(
                f"/rules/{rule_id}/test",
                json={"sample_record": {"sku_name": "Pepsi Max 330ml Can", "bt": "Beverage"}},
            )
            assert test_resp.status_code == 200
            assert test_resp.json()["fires"] is True
            assert test_resp.json()["sample_record_after"]["bt"] == "Soft Drink"
        finally:
            del_resp = client.delete(f"/rules/{rule_id}")
            assert del_resp.status_code == 200


def test_rules_test_draft_endpoint():
    """Verify testing a draft rule before persisting to database."""
    draft_payload = {
        "rule": {
            "rule_id": "draft_test_1",
            "domain": "food",
            "priority": 100,
            "description": "Draft rule test",
            "reasoning": "Test without saving",
            "condition_logic": "AND",
            "is_active": 1,
            "conditions": [
                {
                    "condition_group": 1,
                    "condition_type": "sku_contains",
                    "value": "kottu",
                    "negate": 0,
                }
            ],
            "actions": [{"action_type": "set_bt", "value": "Kottu"}],
        },
        "sample_record": {"sku_name": "Chicken Cheese Kottu", "bt": "Unknown"},
    }
    resp = client.post("/rules/test-draft", json=draft_payload)
    assert resp.status_code == 200
    assert resp.json()["fires"] is True
    assert resp.json()["sample_record_after"]["bt"] == "Kottu"


def test_jobs_list_filters_stats_and_retry(seeded_data):
    """Verify /jobs filtering, X-Total-Count, /jobs/dashboard-stats aggregation, and /jobs/{id}/retry."""
    list_resp = client.get("/jobs", params={"domain": "market", "status": "completed"})
    assert list_resp.status_code == 200
    assert list_resp.headers["X-Total-Count"] == "1"
    jobs_list = list_resp.json()
    assert len(jobs_list) == 1
    assert jobs_list[0]["id"] == "job_1"

    stats_resp = client.get("/jobs/dashboard-stats", params={"domain": "all", "timeframe": "30d"})
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert stats_data["stats"]["totalProcessedSkus"] == 3
    assert stats_data["stats"]["highConfidenceCount"] == 1
    assert stats_data["stats"]["mediumConfidenceCount"] == 1
    assert stats_data["stats"]["lowConfidenceCount"] == 1
    assert {d["domain"] for d in stats_data["domainBreakdown"]} == {"market", "food"}

    with patch(
        "backend.services.job_service.enqueue_job",
        return_value={"job_id": "job_retry_1", "status": "queued"},
    ) as mock_enq:
        retry_ok = client.post("/jobs/job_1/retry")
    assert retry_ok.status_code == 200
    assert retry_ok.json()["new_job_id"] == "job_retry_1"
    mock_enq.assert_called_once()

    retry_no_inputs = client.post("/jobs/job_empty/retry")
    assert retry_no_inputs.status_code == 400

    retry_missing = client.post("/jobs/does_not_exist/retry")
    assert retry_missing.status_code == 404


def test_history_filters_pagination_and_export(seeded_data):
    """Verify /processed-skus filters, X-Total-Count, and CSV export against seeded rows."""
    res_market = client.get(
        "/processed-skus",
        params={"domain": "market", "min_confidence": 0.80, "page": 1, "limit": 1},
    )
    assert res_market.status_code == 200
    assert res_market.headers["X-Total-Count"] == "1"
    assert [r["sku_name"] for r in res_market.json()] == ["Anchor Butter 200g"]

    res_search = client.get("/processed-skus", params={"sku_name": "sunlight", "bt": "Soap"})
    assert res_search.status_code == 200
    assert [r["id"] for r in res_search.json()] == ["ps_2"]

    export_resp = client.get("/processed-skus/export", params={"ids": "ps_1", "format": "csv"})
    assert export_resp.status_code == 200
    assert "text/csv" in export_resp.headers.get("content-type", "")
    assert "Anchor Butter 200g" in export_resp.text
    assert "Sunlight Soap 100g" not in export_resp.text
    assert "Chicken Kottu" not in export_resp.text


def test_audit_logged_inference_endpoints_and_api_requests_inspector(seeded_data):
    """Verify /match, /classify, /pipeline log redacted audit entries retrievable via /api-requests."""
    for path in ("/match", "/classify", "/pipeline"):
        with patch(
            "backend.services.job_service.enqueue_job",
            return_value={
                "job_id": f"job_{path.strip('/')}",
                "status": "queued",
                "total_skus": 1,
                "message": "ok",
            },
        ):
            resp = client.post(
                path,
                headers={"X-Secret-Token": "super-secret-header-key"},
                json={
                    "domain": "market",
                    "callback_url": "",
                    "skus": [{"name": "Anchor Butter 200g", "price": 850.0}],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["job_id"] == f"job_{path.strip('/')}"

    req_resp = client.get("/api-requests", params={"page": 1, "limit": 10})
    assert req_resp.status_code == 200
    requests_list = req_resp.json()
    assert len(requests_list) == 3
    assert {r["path"] for r in requests_list} == {"/match", "/classify", "/pipeline"}

    detail_resp = client.get(f"/api-requests/{requests_list[0]['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert "super-secret-header-key" not in detail["headers_json"]
    assert "***REDACTED***" in detail["headers_json"]

    assert client.get("/api-requests/nonexistent-id-0000").status_code == 404


def test_interactive_suggest_audit_and_rerun_rules():
    """Verify /interactive/suggest, /interactive/audit, and /interactive/rerun-rules."""
    with patch(
        "backend.services.engine_gateway_service.rerun_rules_via_engine",
        return_value={
            "suggested_bt": "Soft Drink",
            "suggested_gk": "Beverage",
            "suggested_region": "",
            "rules_applied": "[]",
        },
    ):
        rerun_resp = client.post(
            "/interactive/rerun-rules",
            json={
                "sku_name": "Coca Cola 500ml",
                "domain": "market",
                "bt": "Drink",
                "gk": "beverage, soda",
                "price": 250.0,
            },
        )
    assert rerun_resp.status_code == 200
    assert rerun_resp.json()["suggested_bt"] == "Soft Drink"
    assert isinstance(rerun_resp.json()["rules_applied"], str)

    with patch(
        "backend.services.engine_gateway_service.suggest_tags",
        return_value={
            "status": "success",
            "sku_name": "Chicken Noodles",
            "suggested_bt": "Noodles",
            "suggested_gk": "Noodles",
            "suggested_region": "Asian",
            "template_used": "chicken_noodles",
            "is_match": True,
        },
    ):
        suggest_resp = client.post(
            "/interactive/suggest",
            json={"sku_name": "Chicken Noodles", "domain": "food"},
        )
    assert suggest_resp.status_code == 200
    assert suggest_resp.json()["suggested_bt"] == "Noodles"
    assert suggest_resp.json()["is_match"] is True

    with patch(
        "backend.services.engine_gateway_service.run_sku_audit",
        return_value={"input": {"sku_name": "Chicken Noodles"}, "final_output": {"score": 0.9}},
    ):
        audit_resp = client.post(
            "/interactive/audit",
            json={"sku_name": "Chicken Noodles", "domain": "food", "task": "pipeline"},
        )
    assert audit_resp.status_code == 200
    assert audit_resp.json()["final_output"]["score"] == 0.9

    with patch(
        "backend.services.engine_gateway_service.suggest_tags",
        side_effect=ValueError("invalid domain"),
    ):
        assert (
            client.post(
                "/interactive/suggest",
                json={"sku_name": "Test", "domain": "bad"},
            ).status_code
            == 400
        )


def test_catalog_search_filters_sorting_and_summary(seeded_data):
    """Verify /catalog dataset='catalog' search, price bounds, sorting, and /catalog/summary-stats."""
    stats_resp = client.get("/catalog/summary-stats", params={"domain": "market"})
    assert stats_resp.status_code == 200
    assert stats_resp.json()["catalog"] == 3

    filtered = client.get(
        "/catalog",
        params={
            "dataset": "catalog",
            "domain": "market",
            "query": "Anchor",
            "min_price": 500.0,
            "max_price": 1200.0,
            "sort_by": "price",
            "sort_order": "desc",
        },
    )
    assert filtered.status_code == 200
    data = filtered.json()
    assert data["total"] == 2
    assert [row["name"] for row in data["results"]] == [
        "Anchor Milk Powder 400g",
        "Anchor Butter 200g",
    ]


def test_catalog_dictionary_datasets(pg):
    """Verify /catalog queries for gk, bt, category (incl. food region mapping), brands, and bt_gk_map."""
    with pg.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE classifier_dictionaries, brand_flavors, bt_gk_map RESTART IDENTITY CASCADE"
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO classifier_dictionaries (domain, tag_type, tag, catalog_count)
                VALUES
                    ('market', 'gk', 'Dairy & Chilled', 42),
                    ('market', 'bt', 'Butter & Spreads', 19),
                    ('market', 'category', 'Fresh Groceries', 25),
                    ('food', 'region', 'Sri Lankan', 30)
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO brand_flavors (domain, name, aliases, is_weak, is_meat, catalog_count, row_hash)
                VALUES
                    ('market', 'Anchor', 'anchr', 0, 0, 50, 'bh1'),
                    ('food', 'Chicken', 'chk', 0, 1, 75, 'bh2')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO bt_gk_map (domain, basictype, generic_keywords, gk_count, catalog_count)
                VALUES ('market', 'Butter', 'Dairy, Spreads', 2, 19)
                """
            )
        )

    try:
        for ds, dom, q, expected_field, expected_val in (
            ("gk", "market", "Dairy", "name", "Dairy & Chilled"),
            ("bt", "market", "Butter", "name", "Butter & Spreads"),
            ("category", "market", "Fresh", "name", "Fresh Groceries"),
            ("category", "food", "Sri", "name", "Sri Lankan"),
            ("brands", "market", "Anchor", "name", "Anchor"),
            ("brands", "food", "Chicken", "name", "Chicken"),
            ("bt_gk_map", "market", "Butter", "bt", "Butter"),
        ):
            r = client.get("/catalog", params={"dataset": ds, "domain": dom, "query": q})
            assert r.status_code == 200, f"Failed for dataset={ds}, domain={dom}"
            payload = r.json()
            assert payload["total"] == 1
            assert payload["results"][0][expected_field] == expected_val
    finally:
        with pg.begin() as conn:
            conn.execute(
                text(
                    "TRUNCATE classifier_dictionaries, brand_flavors, bt_gk_map RESTART IDENTITY CASCADE"
                )
            )


def test_batches_merchant_fetch_models_logs_and_upload(seeded_data, tmp_path):
    """Verify /batches/{id}, /merchant-fetch, /load-models, /models-status, /models/reload, /logs, and /upload."""
    batch_ok = client.get("/batches/batch_1")
    assert batch_ok.status_code == 200
    assert batch_ok.json()["filename"] == "supermarket.csv"
    assert client.get("/batches/missing_batch").status_code == 404

    from backend.schemas.sku_schema import SKUItem

    portal_url = "https://uni-portal-api.pickme.lk/food/v1/t0001/food/place/skus/csv/{merchantid}"
    with patch(
        "backend.services.portal_service.fetch_merchant_csv",
        return_value={
            "skus": [SKUItem(name="Rice Bowl", price=750.0, description="", category="Rice")],
            "auth_failed": False,
            "error": None,
        },
    ):
        mf_ok = client.post(
            "/merchant-fetch",
            json={
                "merchant_id": "M123",
                "bearer_token": "tok",
                "portal_url": portal_url,
                "domain": "food",
            },
        )
    assert mf_ok.status_code == 200
    assert len(mf_ok.json()["rows"]) == 1

    with patch(
        "backend.services.portal_service.fetch_merchant_csv",
        return_value={"skus": [], "auth_failed": True, "error": "MER-4007 token expired"},
    ):
        assert (
            client.post(
                "/merchant-fetch",
                json={
                    "merchant_id": "M123",
                    "bearer_token": "expired",
                    "portal_url": portal_url,
                    "domain": "food",
                },
            ).status_code
            == 401
        )

    assert (
        client.post(
            "/merchant-fetch",
            json={
                "merchant_id": " ",
                "bearer_token": "tok",
                "portal_url": portal_url,
                "domain": "food",
            },
        ).status_code
        == 400
    )

    with (
        patch(
            "backend.services.engine_gateway_service.trigger_load_models",
            return_value={"status": "loading"},
        ),
        patch(
            "backend.services.engine_gateway_service.get_models_status",
            return_value={"ready": True},
        ),
        patch(
            "backend.services.engine_gateway_service.reload_engine_models",
            return_value={"status": "reloaded"},
        ),
    ):
        assert client.post("/load-models").json() == {"status": "loading"}
        assert client.get("/models-status").json() == {"ready": True}
        assert client.post("/models/reload").json() == {"status": "reloaded"}

    fake_log = tmp_path / "backend.log"
    fake_log.write_text("line1\nline2\nline3\n", encoding="utf-8")
    with patch("backend.core.config.LOG_FILE", str(fake_log)):
        logs_resp = client.get("/logs", params={"lines": 2})
    assert logs_resp.status_code == 200
    assert logs_resp.json()["logs"] == "line2\nline3\n"

    # Empty SKUs rejected on /upload
    assert (
        client.post(
            "/upload",
            json={"outlet_id_or_name": "outlet-1", "domain": "market", "skus": []},
        ).status_code
        == 400
    )

    # Non-host-managed upload dispatches to engine client
    with patch("backend.services.engine_gateway_service.dispatch_upload_job") as mock_dispatch:
        up_resp = client.post(
            "/upload",
            json={
                "outlet_id_or_name": "outlet-1",
                "domain": "market",
                "skus": [{"Name": "Butter 200g", "BasicType": "Butter"}],
                "host_managed": False,
            },
        )
    assert up_resp.status_code == 200
    assert up_resp.json()["job_id"]
    mock_dispatch.assert_called_once()
