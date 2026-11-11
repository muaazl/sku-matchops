#!/usr/bin/env python3
"""
SKU MatchOps — Local Google Sheets Pull & Push Worker (Zero-Tunnel)
==================================================================
Eliminates the need for Cloudflare / inbound tunnels on restricted work laptops.

Capabilities:
1. --watch       : Background watcher. Polls the '_queue' tab in Google Sheets.
                   When an operator clicks 'Run' in Google Sheets, this worker
                   pulls the SKUs, runs ML inference locally, and writes back
                   predictions + color coding automatically via Apps Script doPost.
2. CLI / Single  : python scripts/ops/sync_sheets.py --sheet "TabName" --domain food --task pipeline
3. Interactive   : python scripts/ops/sync_sheets.py
                   Fetches available tabs from Google Sheets and prompts for selection.
"""

import argparse
import csv
import io
import os
import queue
import re
import sys
import threading
import time
import urllib.parse
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

# Apps Script /exec randomly hangs ~30% of requests; every call here is safe to repeat.
# Only https (Google) retries; the local backend (http) keeps failing fast.
_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
        )
    ),
)

DEFAULT_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "17wPPEUu0_KbgOPtjDy4UVS8fh76D1ZmQlB5FZMw2JEo")
DEFAULT_APPS_SCRIPT_URL = os.getenv(
    "APPS_SCRIPT_URL",
    "https://script.google.com/macros/s/AKfycbyGbZfZ-JOEdX2RXs5IPRMhcPQd3SVk6Vr29NcmFjclgjdBUQngZYj_CTSkiatQjxY/exec",
)
# 127.0.0.1, not localhost: on Windows localhost tries IPv6 first and stalls ~2s per connection
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
# CLI task name -> Gateway route (same mapping as the Apps Script)
TASK_ENDPOINTS = {"pipeline": "pipeline", "matcher": "match", "classifier": "classify"}
DEFAULT_API_KEY = os.getenv("MATCHOPS_API_KEY", "bismillah")

ALLOWED_UPLOAD_COLUMNS = (
    "Name",
    "RefID",
    "Description",
    "Price",
    "Category",
    "Individually Sellable",
    "Brands",
    "Categories",
    "Generic keywords",
    "Promotions",
    "Special Properties and Ingredients",
    "Uncategrized",
    "basictype",
    "dietarypreference",
    "foodstate",
    "mealtime",
    "other",
    "region",
)


def fetch_sheet_csv_gviz(sheet_id: str, sheet_name: str, timeout: int = 15) -> str:
    """Fetches raw CSV text from Google Sheets via the public/org gviz export."""
    encoded_sheet = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded_sheet}"
    resp = _session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_sheet_data_apps_script(
    apps_script_url: str, sheet_id: str, sheet_name: str, timeout: int = 25
) -> list[dict[str, Any]]:
    """Fallback fetcher using the Apps Script Web App doGet endpoint."""
    params = {
        "action": "get_data",
        "spreadsheet_id": sheet_id,
        "sheet_name": sheet_name,
    }
    resp = _session.get(apps_script_url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "ok":
        raise RuntimeError(payload.get("message", "Failed to fetch data from Apps Script"))

    raw_data = payload.get("data", [])
    if len(raw_data) < 2:
        return []

    headers = [str(h).strip() for h in raw_data[0]]
    rows = []
    for row in raw_data[1:]:
        row_dict = {}
        for idx, val in enumerate(row):
            if idx < len(headers):
                row_dict[headers[idx]] = val
        rows.append(row_dict)
    return rows


def parse_csv_to_skus(csv_text: str) -> list[dict[str, Any]]:
    """Parses raw CSV into normalized SKU dictionaries."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return []

    # Map column headers case-insensitively
    name_col = None
    price_col = None
    desc_col = None
    cat_col = None

    for col in reader.fieldnames:
        low = col.strip().lower()
        if low in ("name", "sku", "dish", "dish name", "sku name", "product"):
            name_col = col
        elif low in ("price", "cost", "rate"):
            price_col = col
        elif low in ("description", "desc", "details"):
            desc_col = col
        elif low in ("category", "seller category", "sellercategory"):
            cat_col = col

    if not name_col:
        raise ValueError(
            f"Could not locate a SKU name column in headers: {reader.fieldnames}. "
            "Expected 'Name', 'Dish', or 'SKU'."
        )

    skus = []
    for row in reader:
        name = str(row.get(name_col, "") or "").strip()
        if not name:
            continue

        price_raw = row.get(price_col, 0) if price_col else 0
        try:
            price = float(price_raw)
        except (ValueError, TypeError):
            price = 0.0

        desc = str(row.get(desc_col, "") or "").strip() if desc_col else ""
        cat = str(row.get(cat_col, "") or "").strip() if cat_col else ""

        skus.append(
            {
                "name": name,
                "price": price,
                "description": desc,
                "category": cat,
            }
        )

    return skus


def normalize_upload_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Filters sheet rows to the 18 PickMe bulk-upload columns and cleans comma-spaces."""
    cleaned: list[dict[str, str]] = []
    for row in rows:
        name_val = str(row.get("Name", "") or "").strip()
        if not name_val:
            continue
        item = {
            col: ("" if row.get(col) is None else str(row.get(col)))
            for col in ALLOWED_UPLOAD_COLUMNS
        }
        item["Generic keywords"] = re.sub(r",\s+", ",", item["Generic keywords"])
        cleaned.append(item)
    return cleaned


def parse_csv_to_upload_skus(csv_text: str) -> list[dict[str, str]]:
    """Parses raw Uploading Sheet CSV into 18-column PickMe bulk-upload dicts."""
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = [f.strip() for f in (reader.fieldnames or [])]
    if "Name" not in fieldnames or "RefID" not in fieldnames:
        raise ValueError(
            f"Uploading sheet missing required 'Name'/'RefID' headers. Found: {fieldnames}"
        )
    return normalize_upload_rows(list(reader))


def fetch_upload_skus(sheet_id: str, sheet_name: str, apps_script_url: str) -> list[dict[str, str]]:
    """Fetches full upload rows via gviz CSV export, falling back to Apps Script doGet."""
    try:
        csv_text = fetch_sheet_csv_gviz(sheet_id, sheet_name)
        return parse_csv_to_upload_skus(csv_text)
    except Exception as gviz_err:
        print(
            f"      [INFO] Direct CSV export unavailable ({gviz_err}). Fetching via Apps Script...",
            flush=True,
        )
        raw_rows = fetch_sheet_data_apps_script(apps_script_url, sheet_id, sheet_name)
        if raw_rows and ("Name" not in raw_rows[0] or "RefID" not in raw_rows[0]):
            raise ValueError(
                f"Uploading sheet missing required 'Name'/'RefID' headers. Found: {list(raw_rows[0].keys())}"
            )
        return normalize_upload_rows(raw_rows)


def list_available_tabs(apps_script_url: str, sheet_id: str) -> list[str]:
    """Queries Apps Script Web App for available non-internal tabs."""
    try:
        resp = _session.get(
            apps_script_url,
            params={"action": "list_sheets", "spreadsheet_id": sheet_id},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok" and data.get("sheets"):
                return data["sheets"]
    except Exception:
        pass
    return []


def fetch_queue_gviz(catalog_sheet_id: str) -> list[dict[str, Any]]:
    """Reads pending jobs directly from the central Catalog Sheet _queue tab via gviz CSV export."""
    try:
        csv_text = fetch_sheet_csv_gviz(catalog_sheet_id, "_queue", timeout=10)
        reader = csv.DictReader(io.StringIO(csv_text))
        pending = []
        for r_idx, row in enumerate(reader, start=2):
            status = str(row.get("Status", "")).strip().upper()
            if status == "PENDING":
                task_val = str(row.get("Task", "pipeline") or "pipeline").strip()
                raw_type = str(row.get("Type", "") or "").strip().lower()
                type_val = raw_type or ("uploading" if task_val.lower() == "upload" else "tagging")
                pending.append(
                    {
                        "row_index": r_idx,
                        "job_id": row.get("Job ID", ""),
                        "spreadsheet_id": row.get("Spreadsheet ID", ""),
                        "sheet_name": row.get("Sheet Name", ""),
                        "type": type_val,
                        "domain": row.get("Domain", "food") or "food",
                        "task": task_val,
                        "status": "PENDING",
                        "created_at": row.get("Created At", ""),
                    }
                )
        return pending
    except Exception:
        return []


def update_queue_status(apps_script_url: str, sheet_id: str, job_id: str, status: str):
    """Sends a queue_update notification to Google Apps Script doPost."""
    try:
        _session.post(
            apps_script_url,
            json={
                "type": "queue_update",
                "spreadsheet_id": sheet_id,
                "job_id": job_id,
                "status": status,
            },
            timeout=10,
        )
    except Exception as e:
        print(f"[WARN] Failed to update queue status to {status}: {e}", flush=True)


def check_backend_health(backend_url: str) -> bool:
    """Verifies that the local FastAPI backend is active."""
    try:
        resp = _session.get(f"{backend_url.rstrip('/')}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def run_single_job(
    sheet_name: str,
    domain: str,
    task: str,
    sheet_id: str = DEFAULT_SHEET_ID,
    apps_script_url: str = DEFAULT_APPS_SCRIPT_URL,
    backend_url: str = DEFAULT_BACKEND_URL,
    api_key: str = DEFAULT_API_KEY,
    job_id: str | None = None,
) -> bool:
    """Pulls data for a specific sheet tab, triggers the backend pipeline, and tracks progress."""
    print(f"\n{'=' * 65}", flush=True)
    print(f"  SKU MatchOps — Processing Tab: '{sheet_name}'", flush=True)
    print(f"  Domain : {domain.upper()} | Task : {task.upper()}", flush=True)
    print(f"{'=' * 65}", flush=True)

    # 1. Fetch SKUs
    print(f"[1/4] Reading data from tab '{sheet_name}' in Google Sheets...", flush=True)
    skus = []
    try:
        csv_text = fetch_sheet_csv_gviz(sheet_id, sheet_name)
        skus = parse_csv_to_skus(csv_text)
    except Exception as gviz_err:
        if "401" in str(gviz_err) or "Unauthorized" in str(gviz_err):
            print("      Sheet is private. Fetching via authenticated Apps Script...", flush=True)
        else:
            print(
                f"      [WARN] Direct CSV export failed ({gviz_err}). Trying Apps Script...",
                flush=True,
            )
        try:
            raw_rows = fetch_sheet_data_apps_script(apps_script_url, sheet_id, sheet_name)
            # Re-serialize to CSV text for common parsing
            if raw_rows:
                keys = raw_rows[0].keys()
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=keys)
                writer.writeheader()
                writer.writerows(raw_rows)
                skus = parse_csv_to_skus(buf.getvalue())
        except Exception as as_err:
            print(f"[ERROR] Could not read tab '{sheet_name}': {as_err}", flush=True)
            if job_id:
                update_queue_status(apps_script_url, sheet_id, job_id, "FAILED")
            return False

    if not skus:
        print(
            f"[ERROR] No valid SKUs found with a 'Name' column in tab '{sheet_name}'.", flush=True
        )
        if job_id:
            update_queue_status(apps_script_url, sheet_id, job_id, "FAILED (No SKUs)")
        return False

    print(f"      Parsed {len(skus)} SKUs successfully.", flush=True)

    # 2. Check backend health
    if not check_backend_health(backend_url):
        print(
            f"[ERROR] Backend at {backend_url} is unreachable!\n"
            f"        Please start the server first:\n"
            f"          docker compose up -d\n"
            f"        OR locally:\n"
            f"          uvicorn backend.main:app --port 8000",
            flush=True,
        )
        if job_id:
            update_queue_status(apps_script_url, sheet_id, job_id, "FAILED (Server Offline)")
        return False

    # 3. Enqueue to backend
    endpoint = f"{backend_url.rstrip('/')}/{TASK_ENDPOINTS[task.lower()]}"
    payload = {
        "skus": skus,
        "domain": domain.lower(),
        "callback_url": apps_script_url,
        "spreadsheet_id": sheet_id,
        "sheet_name": sheet_name,
    }
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    print(f"[2/2] Enqueuing {len(skus)} SKUs to backend ({endpoint})...", flush=True)
    try:
        resp = _session.post(endpoint, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        res_data = resp.json()
        enqueued_job_id = res_data.get("job_id", "")
        print(
            f"      Enqueued to backend successfully! Backend Job ID: {enqueued_job_id}", flush=True
        )
        print(
            "      Backend is processing in background and will update Google Sheets automatically.\n",
            flush=True,
        )
        if job_id:
            update_queue_status(
                apps_script_url, DEFAULT_SHEET_ID, job_id, f"ENQUEUED (Job {enqueued_job_id})"
            )
        return True
    except Exception as req_err:
        print(f"[ERROR] Failed to dispatch job to backend: {req_err}", flush=True)
        if job_id:
            update_queue_status(apps_script_url, DEFAULT_SHEET_ID, job_id, "FAILED")
        return False


def run_upload_job(
    sheet_name: str,
    domain: str = "food",
    sheet_id: str = DEFAULT_SHEET_ID,
    catalog_sheet_id: str = DEFAULT_SHEET_ID,
    apps_script_url: str = DEFAULT_APPS_SCRIPT_URL,
    backend_url: str = DEFAULT_BACKEND_URL,
    api_key: str = DEFAULT_API_KEY,
    job_id: str | None = None,
    has_more_jobs: bool = False,
) -> bool:
    """Pulls full upload rows from Google Sheets and runs Playwright Chrome upload on the host."""
    outlet_id = (sheet_name or "").split("_", 1)[0].strip()
    print(f"\n{'=' * 65}", flush=True)
    print(f"  SKU MatchOps — Uploading Tab: '{sheet_name}' (Outlet: {outlet_id})", flush=True)
    print(f"{'=' * 65}", flush=True)

    if not outlet_id.isdigit():
        print(
            f"[ERROR] Cannot extract numeric Outlet ID from tab '{sheet_name}'. Expected '<id>_<name>'.",
            flush=True,
        )
        if job_id:
            update_queue_status(
                apps_script_url, catalog_sheet_id, job_id, "FAILED (Invalid Outlet ID)"
            )
        return False

    if job_id:
        update_queue_status(apps_script_url, catalog_sheet_id, job_id, "UPLOADING")

    print(f"[1/3] Reading upload rows from tab '{sheet_name}'...", flush=True)
    try:
        skus = fetch_upload_skus(sheet_id, sheet_name, apps_script_url)
    except Exception as read_err:
        print(f"[ERROR] Failed to read upload tab '{sheet_name}': {read_err}", flush=True)
        if job_id:
            update_queue_status(apps_script_url, catalog_sheet_id, job_id, "FAILED (Read Error)")
        return False

    if not skus:
        print(f"[ERROR] No SKUs found to upload in tab '{sheet_name}'.", flush=True)
        if job_id:
            update_queue_status(apps_script_url, catalog_sheet_id, job_id, "FAILED (No SKUs)")
        return False

    print(f"      Loaded {len(skus)} SKUs for Outlet {outlet_id}.", flush=True)

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    base_api = backend_url.rstrip("/")
    backend_job_id = ""
    if check_backend_health(backend_url):
        try:
            reg_resp = _session.post(
                f"{base_api}/upload",
                json={
                    "outlet_id_or_name": outlet_id,
                    "sheet_name": sheet_name,
                    "skus": skus,
                    "domain": domain.lower(),
                    "host_managed": True,
                },
                headers=headers,
                timeout=15,
            )
            if reg_resp.ok:
                backend_job_id = str(reg_resp.json().get("job_id", ""))
                print(
                    f"[2/3] Registered upload job in MatchOps Backend (Job ID: {backend_job_id}).",
                    flush=True,
                )
        except Exception as reg_err:
            print(f"      [WARN] Could not register upload job in Backend: {reg_err}", flush=True)

    cancelled = False
    last_cancel_poll = 0.0
    last_pct = 0.0

    def _patch_backend(payload: dict[str, Any]) -> dict[str, Any]:
        if not backend_job_id:
            return {}
        try:
            r = _session.patch(
                f"{base_api}/upload/{backend_job_id}",
                json=payload,
                headers=headers,
                timeout=5,
            )
            if r.ok:
                return r.json()
        except Exception:
            pass
        return {}

    def progress_cb(stage: str, pct: float) -> None:
        nonlocal cancelled, last_pct, last_cancel_poll
        last_pct = float(pct)
        last_cancel_poll = time.monotonic()
        res = _patch_backend({"status": "running", "stage": stage, "progress_pct": last_pct})
        if res.get("cancel_requested"):
            cancelled = True

    def is_cancelled_cb() -> bool:
        nonlocal cancelled, last_cancel_poll
        if cancelled or not backend_job_id:
            return cancelled
        now = time.monotonic()
        if now - last_cancel_poll >= 2.0:
            last_cancel_poll = now
            res = _patch_backend(
                {"status": "running", "stage": "uploading", "progress_pct": last_pct}
            )
            if res.get("cancel_requested"):
                cancelled = True
        return cancelled

    t0 = time.time()
    print("[3/3] Running Chrome bulk-upload automation on port 9222...", flush=True)
    try:
        from engine.pipeline.uploader import process_upload

        process_upload(
            skus=skus,
            outlet_id=outlet_id,
            job_id=backend_job_id or job_id or outlet_id,
            progress_callback=progress_cb,
            is_cancelled=is_cancelled_cb,
            has_more_jobs=has_more_jobs,
        )
    except Exception as up_err:
        duration_min = round((time.time() - t0) / 60, 4)
        err_str = str(up_err)
        print(f"[ERROR] Upload failed for '{sheet_name}': {err_str}", flush=True)
        if "9222" in err_str:
            print(
                "        -> Please start Chrome via start_chrome.bat before running uploads.",
                flush=True,
            )
        _patch_backend(
            {
                "status": "failed",
                "duration_minutes": duration_min,
                "error_message": err_str[:500],
            }
        )
        if job_id:
            short_reason = "Chrome 9222 Offline" if "9222" in err_str else "Upload Error"
            update_queue_status(
                apps_script_url, catalog_sheet_id, job_id, f"FAILED ({short_reason})"
            )
        return False

    duration_min = round((time.time() - t0) / 60, 4)
    if cancelled:
        print(f"[INFO] Upload cancelled for '{sheet_name}'.\n", flush=True)
        _patch_backend(
            {
                "status": "cancelled",
                "duration_minutes": duration_min,
                "error_message": "Cancelled by user",
            }
        )
        if job_id:
            update_queue_status(apps_script_url, catalog_sheet_id, job_id, "CANCELLED")
        return False

    _patch_backend({"status": "completed", "duration_minutes": duration_min})
    if job_id:
        update_queue_status(apps_script_url, catalog_sheet_id, job_id, "DONE")
    print(
        f"      Upload completed successfully for '{sheet_name}' ({len(skus)} SKUs)!\n", flush=True
    )
    return True


def watch_queue(
    sheet_id: str = DEFAULT_SHEET_ID,
    apps_script_url: str = DEFAULT_APPS_SCRIPT_URL,
    backend_url: str = DEFAULT_BACKEND_URL,
    api_key: str = DEFAULT_API_KEY,
    interval: float = 5.0,
):
    """Continuously polls Google Sheets '_queue' tab for both tagging and uploading jobs."""
    print("=" * 65, flush=True)
    print("  SKU MatchOps — Google Sheets Queue Watcher (Zero-Tunnel)", flush=True)
    print("=" * 65, flush=True)
    print(f"  Spreadsheet ID  : {sheet_id}", flush=True)
    print(f"  Apps Script URL : {apps_script_url[:60]}...", flush=True)
    print(f"  Backend URL     : {backend_url}", flush=True)
    print(f"  Poll Interval   : {interval}s", flush=True)
    print("=" * 65, flush=True)
    print(
        "[WATCHER] Listening for tagging & uploading jobs from Google Sheets... (Press Ctrl+C to stop)\n",
        flush=True,
    )

    processed_job_ids = set()
    upload_q: queue.Queue[dict[str, Any]] = queue.Queue()

    def _upload_worker() -> None:
        while True:
            u_job = upload_q.get()
            try:
                run_upload_job(
                    sheet_name=u_job["sheet_name"],
                    domain=u_job["domain"],
                    sheet_id=u_job["target_spreadsheet_id"],
                    catalog_sheet_id=sheet_id,
                    apps_script_url=apps_script_url,
                    backend_url=backend_url,
                    api_key=api_key,
                    job_id=u_job["job_id"],
                    has_more_jobs=not upload_q.empty(),
                )
            except Exception as worker_err:
                print(f"[UPLOAD WORKER] Unexpected error: {worker_err}", flush=True)
            finally:
                upload_q.task_done()

    threading.Thread(target=_upload_worker, name="host-upload-worker", daemon=True).start()

    while True:
        try:
            pending_jobs = []
            try:
                resp = _session.get(
                    apps_script_url,
                    params={"action": "get_queue", "spreadsheet_id": sheet_id},
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        pending_jobs = data.get("pending", [])
            except Exception:
                pass

            if not pending_jobs:
                pending_jobs = fetch_queue_gviz(sheet_id)

            for job in pending_jobs:
                jid = job.get("job_id")
                if jid in processed_job_ids:
                    continue

                target_spreadsheet_id = job.get("spreadsheet_id") or sheet_id
                sheet_name = job.get("sheet_name")
                domain = job.get("domain", "food")
                task = str(job.get("task", "pipeline") or "pipeline")
                job_type = str(job.get("type") or "").strip().lower() or (
                    "uploading" if task.lower() == "upload" else "tagging"
                )

                print(
                    f"[WATCHER] Found queued {job_type.upper()} job '{jid}' on tab '{sheet_name}' (task: {task}, domain: {domain}).",
                    flush=True,
                )
                processed_job_ids.add(jid)

                if job_type == "uploading" or task.lower() == "upload":
                    if jid:
                        update_queue_status(apps_script_url, sheet_id, jid, "QUEUED")
                    upload_q.put(
                        {
                            "job_id": jid,
                            "target_spreadsheet_id": target_spreadsheet_id,
                            "sheet_name": sheet_name,
                            "domain": domain,
                        }
                    )
                else:
                    # Immediately enqueue tagging job to backend without blocking the watcher
                    run_single_job(
                        sheet_name=sheet_name,
                        domain=domain,
                        task=task,
                        sheet_id=target_spreadsheet_id,
                        apps_script_url=apps_script_url,
                        backend_url=backend_url,
                        api_key=api_key,
                        job_id=jid,
                    )

        except KeyboardInterrupt:
            print("\n[WATCHER] Watcher stopped by user.", flush=True)
            sys.exit(0)
        except Exception as e:
            # Avoid spamming errors on temporary network hiccups
            print(f"[WATCHER] Polling check notice: {e}", flush=True)

        time.sleep(interval)


def interactive_mode(sheet_id: str, apps_script_url: str, backend_url: str, api_key: str):
    """Interactive CLI prompt to select tab, domain, and task."""
    print("=" * 65, flush=True)
    print("         SKU MatchOps — Interactive Google Sheets Sync          ", flush=True)
    print("=" * 65, flush=True)

    print(f"[INFO] Discovering tabs in spreadsheet ({sheet_id})...", flush=True)
    tabs = list_available_tabs(apps_script_url, sheet_id)

    selected_sheet = ""
    if tabs:
        print("\nAvailable Tabs:")
        for idx, t in enumerate(tabs, 1):
            print(f"  [{idx}] {t}")
        print("  [0] Enter custom tab name manually")

        choice = input(f"\nSelect tab [1-{len(tabs)}] (or press Enter for 1): ").strip()
        if not choice:
            choice = "1"

        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(tabs):
                selected_sheet = tabs[val - 1]

    if not selected_sheet:
        selected_sheet = input("Enter Google Sheet tab name: ").strip()

    if not selected_sheet:
        print("[ERROR] Tab name cannot be empty.", flush=True)
        sys.exit(1)

    # Domain selection
    default_domain = "market" if "market" in selected_sheet.lower() else "food"
    domain_input = (
        input(f"Select domain [food/market] (default: {default_domain}): ").strip().lower()
    )
    domain = domain_input if domain_input in ("food", "market") else default_domain

    # Task selection
    task_input = (
        input("Select task [pipeline/matcher/classifier] (default: pipeline): ").strip().lower()
    )
    task = task_input if task_input in ("pipeline", "matcher", "classifier") else "pipeline"

    run_single_job(
        sheet_name=selected_sheet,
        domain=domain,
        task=task,
        sheet_id=sheet_id,
        apps_script_url=apps_script_url,
        backend_url=backend_url,
        api_key=api_key,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SKU MatchOps Google Sheets Sync Worker (Zero-Tunnel)"
    )
    parser.add_argument("--sheet", help="Target Google Sheet tab name to process")
    parser.add_argument("--domain", choices=["food", "market"], help="Domain type (food or market)")
    parser.add_argument(
        "--task",
        choices=["pipeline", "matcher", "classifier"],
        default="pipeline",
        help="Task to run (default: pipeline)",
    )
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID, help="Google Sheet ID")
    parser.add_argument(
        "--apps-script-url", default=DEFAULT_APPS_SCRIPT_URL, help="Google Apps Script Web App URL"
    )
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Backend URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key for MatchOps backend")
    parser.add_argument("--watch", action="store_true", help="Run in continuous queue watcher mode")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Polling interval in seconds for --watch"
    )

    args = parser.parse_args()

    if args.watch:
        watch_queue(
            sheet_id=args.sheet_id,
            apps_script_url=args.apps_script_url,
            backend_url=args.backend_url,
            api_key=args.api_key,
            interval=args.interval,
        )
    elif args.sheet:
        domain = args.domain or ("market" if "market" in args.sheet.lower() else "food")
        run_single_job(
            sheet_name=args.sheet,
            domain=domain,
            task=args.task,
            sheet_id=args.sheet_id,
            apps_script_url=args.apps_script_url,
            backend_url=args.backend_url,
            api_key=args.api_key,
        )
    else:
        interactive_mode(
            sheet_id=args.sheet_id,
            apps_script_url=args.apps_script_url,
            backend_url=args.backend_url,
            api_key=args.api_key,
        )
