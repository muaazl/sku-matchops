import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import api_router
from scripts.ops import sync_sheets


def test_parse_csv_to_upload_skus_filters_and_cleans():
    csv_text = (
        "Name,RefID,Price,Generic keywords,basictype,region,ExtraIgnoredCol\n"
        'Chicken Kottu,SKU-1,1200,"rice,  kottu, spicy ",Kottu,Sri Lankan,drop_me\n'
        ",SKU-EMPTY,500,ignored,Kottu,Sri Lankan,drop_me\n"
    )
    rows = sync_sheets.parse_csv_to_upload_skus(csv_text)
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row.keys()) == sync_sheets.ALLOWED_UPLOAD_COLUMNS
    assert row["Name"] == "Chicken Kottu"
    assert row["RefID"] == "SKU-1"
    assert row["Generic keywords"] == "rice,kottu,spicy "
    assert row["basictype"] == "Kottu"
    assert row["region"] == "Sri Lankan"
    assert "ExtraIgnoredCol" not in row


def test_parse_csv_to_upload_skus_rejects_html_or_missing_headers():
    with pytest.raises(ValueError, match="Name"):
        sync_sheets.parse_csv_to_upload_skus("<!DOCTYPE html><html>Sign in</html>")


def test_fetch_queue_gviz_reads_type_and_infers_fallback(monkeypatch):
    csv_data = (
        "Job ID,Spreadsheet ID,Sheet Name,Type,Domain,Task,Status,Created At\n"
        "j1,ss_tag,Outlet A,tagging,food,pipeline,PENDING,2026-11-09T10:00:00Z\n"
        "j2,ss_up,91212_Outlet A,uploading,food,upload,PENDING,2026-11-09T10:01:00Z\n"
        "j3,ss_up,91213_Outlet B,,food,upload,PENDING,2026-11-09T10:02:00Z\n"
        "j4,ss_up,91214_Outlet C,uploading,food,upload,DONE,2026-11-09T10:03:00Z\n"
    )
    monkeypatch.setattr(sync_sheets, "fetch_sheet_csv_gviz", lambda *a, **kw: csv_data)

    pending = sync_sheets.fetch_queue_gviz("catalog_id")
    assert [p["job_id"] for p in pending] == ["j1", "j2", "j3"]
    assert pending[0]["type"] == "tagging"
    assert pending[1]["type"] == "uploading"
    assert pending[2]["type"] == "uploading"


def test_run_upload_job_executes_host_playwright_and_marks_done(monkeypatch):
    statuses: list[str] = []
    uploaded_calls: list[dict] = []

    monkeypatch.setattr(
        sync_sheets,
        "update_queue_status",
        lambda _url, _sid, _jid, status: statuses.append(status),
    )
    monkeypatch.setattr(
        sync_sheets,
        "fetch_upload_skus",
        lambda _sid, _sname, _url: [
            {"Name": "Burger", "RefID": "R1", "Generic keywords": "beef,bun"}
        ],
    )
    monkeypatch.setattr(sync_sheets, "check_backend_health", lambda _url: False)

    import engine.pipeline.uploader as uploader

    def fake_process_upload(
        skus, outlet_id, job_id, progress_callback, is_cancelled, has_more_jobs=False
    ):
        progress_callback("uploading", 50.0)
        assert not is_cancelled()
        uploaded_calls.append(
            {"skus": skus, "outlet_id": outlet_id, "job_id": job_id, "has_more_jobs": has_more_jobs}
        )
        return {
            "results": [{"status": "High (Uploaded)", "sku_name": "Burger"}],
            "pages_processed": 1,
        }

    monkeypatch.setattr(uploader, "process_upload", fake_process_upload)

    ok = sync_sheets.run_upload_job(
        sheet_name="91212_Burger Hub",
        sheet_id="upload_ss_id",
        catalog_sheet_id="catalog_ss_id",
        job_id="uuid-123",
        has_more_jobs=True,
    )
    assert ok is True
    assert statuses == ["UPLOADING", "DONE"]
    assert len(uploaded_calls) == 1
    assert uploaded_calls[0]["outlet_id"] == "91212"
    assert uploaded_calls[0]["has_more_jobs"] is True


def test_run_upload_job_rejects_invalid_outlet_prefix(monkeypatch):
    statuses: list[str] = []
    monkeypatch.setattr(
        sync_sheets,
        "update_queue_status",
        lambda _url, _sid, _jid, status: statuses.append(status),
    )
    ok = sync_sheets.run_upload_job(sheet_name="NoIdTabName", job_id="uuid-bad")
    assert ok is False
    assert statuses == ["FAILED (Invalid Outlet ID)"]


def test_target_page_size_thresholds():
    from engine.pipeline.uploader import _target_page_size

    assert _target_page_size(1) is None
    assert _target_page_size(10) is None
    assert _target_page_size(11) == "20"
    assert _target_page_size(20) == "20"
    assert _target_page_size(21) == "50"
    assert _target_page_size(50) == "50"
    assert _target_page_size(51) == "100"
    assert _target_page_size(250) == "100"


def test_host_managed_upload_endpoints_unit(monkeypatch):
    from sqlalchemy import create_engine, event, text
    from sqlalchemy.pool import StaticPool

    import backend.app.api.endpoints.upload as upload_ep
    from backend.app.core.security import Principal
    from backend.app.schemas.models import UploadJobStateUpdate, UploadRequest

    sqlite_eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(sqlite_eng, "connect")
    def _register_now(dbapi_conn, _rec):
        dbapi_conn.create_function("now", 0, lambda: "2026-11-09T12:00:00Z")

    with sqlite_eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE jobs ("
                "id TEXT PRIMARY KEY, status TEXT, current_stage TEXT, "
                "cancel_requested INTEGER DEFAULT 0, domain TEXT, "
                "input_skus_json TEXT, tenant_id TEXT, updated_at TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO jobs VALUES "
                "('77', 'queued', 'queued', 0, 'food', '[{\"Name\":\"Burger\"}]', 'default', '')"
            )
        )

    dispatched = []
    completions = []
    monkeypatch.setattr(upload_ep, "get_engine", lambda: sqlite_eng)
    monkeypatch.setattr(upload_ep, "dispatch_upload_job", lambda **kw: dispatched.append(kw))
    monkeypatch.setattr(
        upload_ep,
        "register_and_dispatch_job",
        lambda **kw: (
            kw["dispatch_fn"]("77"),
            {"job_id": "77", "status": "queued", "total_skus": kw["total_items"]},
        )[1],
    )
    monkeypatch.setattr(upload_ep, "set_job_progress", lambda *a, **kw: None)
    monkeypatch.setattr(upload_ep, "clear_job_progress", lambda *a, **kw: None)
    monkeypatch.setattr(upload_ep, "publish_job_event", lambda *a, **kw: None)
    monkeypatch.setattr(upload_ep, "record_job_completion", lambda **kw: completions.append(kw))

    principal = Principal(username="ops", role="admin", tenant_id="default")

    created = upload_ep.trigger_upload(
        UploadRequest(
            outlet_id_or_name="91212",
            sheet_name="91212_Outlet",
            domain="food",
            host_managed=True,
            skus=[{"Name": "Burger", "RefID": "R1"}],
        ),
        principal=principal,
    )
    assert created["job_id"] == "77"
    assert dispatched == []

    r_prog = upload_ep.update_host_upload_job(
        "77",
        UploadJobStateUpdate(status="running", stage="uploading", progress_pct=25.0),
        principal=principal,
    )
    assert r_prog == {"status": "running", "cancel_requested": False}

    # Simulate user clicking Cancel in Jobs UI
    with sqlite_eng.begin() as conn:
        conn.execute(
            text("UPDATE jobs SET cancel_requested = 1, status = 'cancelled' WHERE id = '77'")
        )

    r_cancel = upload_ep.update_host_upload_job(
        "77",
        UploadJobStateUpdate(status="running", stage="uploading", progress_pct=50.0),
        principal=principal,
    )
    assert r_cancel["cancel_requested"] is True

    # Reset and verify completion recording
    with sqlite_eng.begin() as conn:
        conn.execute(
            text("UPDATE jobs SET cancel_requested = 0, status = 'running' WHERE id = '77'")
        )
    r_done = upload_ep.update_host_upload_job(
        "77",
        UploadJobStateUpdate(status="completed", duration_minutes=0.5),
        principal=principal,
    )
    assert r_done["status"] == "completed"
    assert len(completions) == 1
    assert completions[0]["high_conf"] == 1
    assert completions[0]["match_rate"] == 100.0


@pytest.mark.usefixtures("pg")
def test_host_managed_upload_endpoint_lifecycle(monkeypatch):
    dispatched = []
    monkeypatch.setattr(
        "backend.app.api.endpoints.upload.dispatch_upload_job",
        lambda **kw: dispatched.append(kw),
    )

    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)

    create_resp = client.post(
        "/upload",
        json={
            "outlet_id_or_name": "91212",
            "sheet_name": "91212_Test Outlet",
            "domain": "food",
            "host_managed": True,
            "skus": [{"Name": "Cheese Kottu", "RefID": "REF-1"}],
        },
    )
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]
    assert dispatched == []

    prog_resp = client.patch(
        f"/upload/{job_id}",
        json={"status": "running", "stage": "uploading", "progress_pct": 40.0},
    )
    assert prog_resp.status_code == 200
    assert prog_resp.json()["cancel_requested"] is False

    done_resp = client.patch(
        f"/upload/{job_id}",
        json={"status": "completed", "duration_minutes": 0.25},
    )
    assert done_resp.status_code == 200

    job_resp = client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "completed"
    assert job_resp.json()["target_sheet"] == "91212"
