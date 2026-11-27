"""Tests for ETagMiddleware, AuditLoggingMiddleware, portal_service, and PG catalog_sync."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.api.middleware.audit_logging_middleware import AuditLoggingMiddleware
from backend.api.middleware.etag_middleware import ETAG_MAX_BODY_SIZE, ETagMiddleware
from backend.repositories import catalog_repository
from backend.services import catalog_sync_service as catalog_sync
from backend.services.portal_service import fetch_merchant_csv, parse_csv_text_to_skus


def _build_middleware_app():
    app = FastAPI()
    app.add_middleware(AuditLoggingMiddleware)
    app.add_middleware(ETagMiddleware)

    @app.get("/small")
    def small_get():
        return {"hello": "world"}

    @app.get("/large")
    def large_get():
        return {"blob": "x" * (ETAG_MAX_BODY_SIZE + 512)}

    @app.get("/jobs/1/stream")
    def stream_get():
        return StreamingResponse(iter(["data: hi\n\n"]), media_type="text/event-stream")

    @app.get("/err")
    def err_get():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="nope")

    @app.post("/match")
    def match_post(payload: dict):
        return {"ok": True, "secret_token": "raw-secret"}

    return app


def test_etag_middleware_conditional_304_and_exclusions():
    app = _build_middleware_app()
    c = TestClient(app)

    res = c.get("/small")
    assert res.status_code == 200
    etag = res.headers.get("etag")
    assert etag and etag.startswith('W/"')

    # Matching If-None-Match short-circuits with 304 Not Modified
    res_304 = c.get("/small", headers={"If-None-Match": etag})
    assert res_304.status_code == 304
    assert res_304.headers.get("etag") == etag
    assert res_304.content == b""

    # Mismatched If-None-Match returns full 200 payload
    res_200 = c.get("/small", headers={"If-None-Match": 'W/"mismatch"'})
    assert res_200.status_code == 200

    # Large bodies, streams, POSTs, and errors bypass ETag generation
    assert "etag" not in c.get("/large").headers
    assert "etag" not in c.get("/jobs/1/stream").headers
    assert "etag" not in c.get("/err").headers


def test_audit_logging_middleware_filters_and_redacts():
    app = _build_middleware_app()
    c = TestClient(app)

    recorded = []
    with patch("backend.services.audit_service.record_api_request", side_effect=recorded.append):
        c.get("/small")
        assert recorded == []  # GETs are never audited

        res = c.post(
            "/match?q=1",
            headers={"X-API-Key": "secret-key", "Authorization": "Bearer abc"},
            json={"skus": ["Milk"], "api_token": "tok123"},
        )
        assert res.status_code == 200

    assert len(recorded) == 1
    row = recorded[0]
    assert row["method"] == "POST"
    assert row["path"] == "/match"
    assert "secret-key" not in row["headers_json"]
    assert "tok123" not in row["payload_json_redacted"]
    assert "raw-secret" not in row["response_json"]
    assert "***REDACTED***" in row["response_json"]


def test_parse_csv_text_to_skus_handles_column_variants_and_bad_prices():
    csv_text = (
        "SKU_Name,MRP,Desc,Type\n"
        "Anchor Butter 200g,850.5,Salted butter,Dairy\n"
        "Bad Price Item,not_a_number,Note,Misc\n"
        "   ,100,Skipped empty name,Misc\n"
    )
    skus = parse_csv_text_to_skus(csv_text)
    assert len(skus) == 2
    assert skus[0].name == "Anchor Butter 200g"
    assert skus[0].price == 850.5
    assert skus[0].description == "Salted butter"
    assert skus[0].category == "Dairy"
    assert skus[1].name == "Bad Price Item"
    assert skus[1].price == 0.0

    # Falls back to first column when no recognized name column header is present
    fallback_skus = parse_csv_text_to_skus("item_label,cost\nBiscuit Pack,120\n")
    assert len(fallback_skus) == 1
    assert fallback_skus[0].name == "Biscuit Pack"
    assert fallback_skus[0].price == 120.0


@pytest.mark.parametrize(
    "error_entry",
    [
        {"code": "MER-4006", "message": "Token invalidated"},
        {"code": "MER-4007", "message": "Session timed out"},
        {"code": "OTHER", "message": "Token Blacklisted by admin"},
    ],
)
def test_fetch_merchant_csv_detects_portal_auth_errors(error_entry):
    resp = MagicMock(status_code=400, ok=False, text='{"errors": []}')
    resp.json.return_value = {"errors": [error_entry]}

    with (
        patch("backend.services.portal_service.validate_outbound_url"),
        patch("backend.services.audit_service.record_outbound_request"),
        patch("backend.services.portal_service.requests.get", return_value=resp),
    ):
        out = fetch_merchant_csv("M1", "tok")

    assert out["auth_failed"] is True
    assert out["skus"] == []
    assert error_entry["message"] in out["error"]


def test_fetch_merchant_csv_handles_http_status_network_and_empty_csv():
    with (
        patch("backend.services.portal_service.validate_outbound_url"),
        patch("backend.services.audit_service.record_outbound_request"),
    ):
        # 401 status flags auth_failed
        r401 = MagicMock(status_code=401, ok=False, text="Unauthorized", reason="Unauthorized")
        r401.json.side_effect = ValueError("not json")
        with patch("backend.services.portal_service.requests.get", return_value=r401):
            out_401 = fetch_merchant_csv("M1", "tok")
        assert out_401["auth_failed"] is True

        # 502 error does not flag auth_failed
        r502 = MagicMock(status_code=502, ok=False, text="Bad Gateway", reason="Bad Gateway")
        r502.json.side_effect = ValueError("not json")
        with patch("backend.services.portal_service.requests.get", return_value=r502):
            out_502 = fetch_merchant_csv("M1", "tok")
        assert out_502["auth_failed"] is False
        assert "502" in out_502["error"]

        # Empty CSV rows error
        r_empty = MagicMock(status_code=200, ok=True, text="Name,Price\n ,100\n")
        r_empty.json.side_effect = ValueError("not json")
        with patch("backend.services.portal_service.requests.get", return_value=r_empty):
            out_empty = fetch_merchant_csv("M1", "tok")
        assert "no valid SKU rows" in out_empty["error"]

        # Network exception
        with patch(
            "backend.services.portal_service.requests.get",
            side_effect=requests.Timeout("timed out"),
        ):
            out_net = fetch_merchant_csv("M1", "tok")
        assert out_net["auth_failed"] is False
        assert "Could not reach" in out_net["error"]


def test_sync_catalog_to_database_incremental_diff(pg, tmp_path, monkeypatch):
    """Verify catalog_sync performs inserts, hash-based updates, and deletes in PostgreSQL."""
    with pg.begin() as conn:
        conn.execute(text("TRUNCATE catalog_items, brand_flavors RESTART IDENTITY CASCADE"))

    cat_path = tmp_path / "Market_Catalog.csv"
    brand_path = tmp_path / "Market_Brands.csv"

    def _fake_staged_path(sheet_name: str) -> str:
        return str(cat_path if "Catalog" in sheet_name else brand_path)

    monkeypatch.setattr(catalog_repository, "resolve_staged_sheet_path", _fake_staged_path)

    try:
        # Pass 1: initial sync inserts 2 catalog rows and 2 brands
        cat_path.write_text(
            "Item Name,Brand,BasicType,Generic keywords,Category,Price,Description\n"
            "Anchor Butter 200g,Anchor,Butter,Dairy,Dairy,850,Salted\n"
            "Removed Juice 1L,Kist,Juice,Beverages,Drinks,450,Mango\n",
            encoding="utf-8",
        )
        brand_path.write_text(
            "Brand Name,Aliases,Is_Weak\nAnchor,anchor,0\nKist,kist,0\n",
            encoding="utf-8",
        )
        cat_df1, brand_df1 = catalog_sync.sync_catalog_to_database(domain="market")
        assert len(cat_df1) == 2
        assert len(brand_df1) == 2

        # Pass 2: keep 1 unchanged, update price/description on Anchor Butter, delete Removed Juice, add Milo
        cat_path.write_text(
            "Item Name,Brand,BasicType,Generic keywords,Category,Price,Description\n"
            "Anchor Butter 200g,Anchor,Butter,Dairy,Dairy,920,Unsalted\n"
            "Milo 400g,Nestle,Malt Drink,Beverages,Drinks,1150,Pack\n",
            encoding="utf-8",
        )
        brand_path.write_text(
            "Brand Name,Aliases,Is_Weak\nAnchor,anchor,0\nNestle,nestle,1\n",
            encoding="utf-8",
        )
        cat_df2, brand_df2 = catalog_sync.sync_catalog_to_database(domain="market")

        assert set(cat_df2["Name"]) == {"Anchor Butter 200g", "Milo 400g"}
        anchor_row = cat_df2[cat_df2["Name"] == "Anchor Butter 200g"].iloc[0]
        assert anchor_row["Price"] == 920.0
        assert set(brand_df2["Brand Name"]) == {"Anchor", "Nestle"}
    finally:
        with pg.begin() as conn:
            conn.execute(text("TRUNCATE catalog_items, brand_flavors RESTART IDENTITY CASCADE"))
