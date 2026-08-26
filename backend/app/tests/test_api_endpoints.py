from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.app.api.routes import api_router

api_app = FastAPI()
api_app.include_router(api_router)
client = TestClient(api_app)


def test_health_endpoint():
    """Verify that GET /health returns status 200 and 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "loaded_domains" in data


def test_rules_crud_lifecycle():
    """Test full CRUD lifecycle for Rules Engine via FastAPI endpoints."""
    # 1. Create a new test rule
    rule_payload = {
        "rule_id": "test_rule_api_999",
        "domain": "market",
        "module": "bt_override",
        "priority": 500,
        "description": "Integration test rule for API suite.",
        "reasoning": "Verify rules endpoint creation and updates.",
        "condition_logic": "AND",
        "is_active": 1,
        "conditions": [
            {
                "condition_group": 1,
                "condition_type": "sku_contains",
                "value": "pepsi",
                "negate": 0
            }
        ],
        "actions": [
            {
                "action_type": "set_bt",
                "value": "Soft Drink"
            }
        ]
    }

    create_resp = client.post("/rules", json=rule_payload)
    assert create_resp.status_code == 200
    created_data = create_resp.json()
    assert "rule_id" in created_data
    rule_id = created_data["rule_id"]

    try:
        # 2. Fetch rules list and verify test rule is present
        list_resp = client.get("/rules", params={"domain": "market"})
        assert list_resp.status_code == 200
        rules = list_resp.json()
        assert any(r["rule_id"] == rule_id for r in rules)

        # 3. Update the rule
        rule_payload["description"] = "Updated integration test rule description."
        update_resp = client.put(f"/rules/{rule_id}", json=rule_payload)
        assert update_resp.status_code == 200

        # 4. Reorder rules with valid payload model
        reorder_resp = client.put("/rules/reorder", json={"ordered_rule_ids": [rule_id]})
        assert reorder_resp.status_code == 200

        # 5. Test rule against sample record
        test_resp = client.post(f"/rules/{rule_id}/test", json={"sample_record": {"sku_name": "Pepsi Max 330ml Can", "bt": "Beverage"}})
        assert test_resp.status_code == 200
        test_data = test_resp.json()
        assert test_data.get("fires") is True
        assert test_data.get("sample_record_after", {}).get("bt") == "Soft Drink"

    finally:
        # 6. Delete the test rule (cleanup)
        del_resp = client.delete(f"/rules/{rule_id}")
        assert del_resp.status_code == 200


def test_rules_test_draft_endpoint():
    """Verify testing a draft rule before persisting to database."""
    draft_payload = {
        "rule": {
            "rule_id": "draft_test_1",
            "domain": "food",
            "module": "bt_override",
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
                    "negate": 0
                }
            ],
            "actions": [
                {
                    "action_type": "set_bt",
                    "value": "Kottu"
                }
            ]
        },
        "sample_record": {
            "sku_name": "Chicken Cheese Kottu",
            "bt": "Unknown"
        }
    }
    resp = client.post("/rules/test-draft", json=draft_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("fires") is True
    assert data.get("sample_record_after", {}).get("bt") == "Kottu"


def test_jobs_endpoints():
    """Verify /jobs list and /jobs/dashboard-stats endpoints."""
    list_resp = client.get("/jobs", params={"page": 1})
    assert list_resp.status_code == 200
    assert isinstance(list_resp.json(), list)

    stats_resp = client.get("/jobs/dashboard-stats", params={"timeframe": "30d"})
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "stats" in stats_data
    assert "confidenceDistribution" in stats_data
    assert "domainBreakdown" in stats_data


def test_history_and_export_endpoints():
    """Verify /processed-skus and CSV export endpoints."""
    hist_resp = client.get("/processed-skus", params={"page": 1, "limit": 10})
    assert hist_resp.status_code == 200
    assert isinstance(hist_resp.json(), list)

    export_resp = client.get("/processed-skus/export", params={"format": "csv"})
    assert export_resp.status_code == 200
    assert "text/csv" in export_resp.headers.get("content-type", "")


def test_api_requests_endpoints():
    """Verify /api-requests audit log query endpoint and detail inspector."""
    req_resp = client.get("/api-requests", params={"page": 1})
    assert req_resp.status_code == 200
    requests_list = req_resp.json()
    assert isinstance(requests_list, list)

    if len(requests_list) > 0:
        first_id = requests_list[0]["id"]
        detail_resp = client.get(f"/api-requests/{first_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert "id" in detail_data
        assert "headers_json" in detail_data
        assert "query_params_json" in detail_data
        assert "payload_json_redacted" in detail_data
        assert "response_json" in detail_data

    # 404 for nonexistent id
    not_found_resp = client.get("/api-requests/nonexistent-id-0000")
    assert not_found_resp.status_code == 404



def test_interactive_rerun_rules():
    """Verify /interactive/rerun-rules executes rules engine on single record."""
    payload = {
        "sku_name": "Coca Cola 500ml",
        "domain": "market",
        "bt": "Drink",
        "gk": "beverage, soda",
        "price": 250.0
    }
    resp = client.post("/interactive/rerun-rules", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "suggested_bt" in data
    assert "suggested_gk" in data
