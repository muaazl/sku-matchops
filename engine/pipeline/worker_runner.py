"""
SKU MatchOps Engine - Background Job Worker Runner
Executes async batch jobs, persists results directly to PostgreSQL, emits progress/status
HTTP callbacks to the Backend Gateway, and handles Google Sheets webhook callbacks.
"""

import logging
import queue
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from engine.core.db import log_outbound_request
from engine.core.observability import SKUS_PROCESSED, trace_context
from engine.pipeline.persistence import record_job_completion, record_job_failure
from engine.pipeline.processor import process_request
from engine.pipeline.task_queue import get_task_queue, run_worker
from engine.pipeline.uploader import process_upload
from engine.rules_engine.loader import pinned_rules, snapshot_rules, sync_rules_version

logger = logging.getLogger("matchops.engine.worker")

_batch_queue = queue.Queue()
_cancelled_jobs: set[str] = set()
_running_jobs: dict[str, dict[str, Any]] = {}
_worker_thread: threading.Thread | None = None

# Persistent HTTP session with connection pooling.
# allowed_methods must explicitly include POST: urllib3's default excludes it (only
# idempotent verbs retry out of the box), and every call this session makes is a POST
# (progress/complete/fail callbacks to the backend, and the Google Sheets webhook).
_http_session = requests.Session()
_retries = Retry(
    total=1,
    backoff_factor=0.1,
    status_forcelist=[502, 503, 504],
    allowed_methods=["GET", "POST"],
)
_http_session.mount(
    "http://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_retries)
)
_http_session.mount(
    "https://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_retries)
)


_exec_lock = threading.Lock()  # one job at a time per process, whichever queue it came from
_queue_jobs: set[str] = set()  # jobs taken from the Redis queue
_cancel_checked: dict[str, tuple[float, bool]] = {}


def cancel_job(job_id: str) -> bool:
    """Marks a job as cancelled so processing aborts immediately (on any worker, via Redis)."""
    _cancelled_jobs.add(str(job_id))
    try:
        queue = get_task_queue()
        if queue is not None:
            queue.request_cancel(str(job_id))
    except Exception as e:
        logger.debug(f"[ENGINE WORKER] Could not broadcast cancellation of {job_id}: {e}")
    logger.info(f"[ENGINE WORKER] Job {job_id} cancellation requested.")
    return True


def is_job_cancelled(job_id: str) -> bool:
    jid = str(job_id)
    if jid in _cancelled_jobs:
        return True
    if jid not in _queue_jobs:
        return False
    # Queued jobs can be cancelled through any gateway/engine: check Redis, at most once a second
    checked_at, cancelled = _cancel_checked.get(jid, (0.0, False))
    if time.time() - checked_at >= 1.0:
        try:
            cancelled = get_task_queue().is_cancelled(jid)
        except Exception:
            cancelled = False
        _cancel_checked[jid] = (time.time(), cancelled)
        if cancelled:
            _cancelled_jobs.add(jid)
    return cancelled


def enqueue_batch_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy path: puts a job onto this process's in-memory queue (used when the Gateway
    could not reach the Redis queue and dispatched over HTTP)."""
    global _worker_thread
    job_id = str(payload.get("job_id", ""))
    _cancelled_jobs.discard(job_id)
    _batch_queue.put(payload)

    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()

    logger.info(f"[ENGINE WORKER] Enqueued job {job_id} ({len(payload.get('skus', []))} SKUs)")
    return {"status": "accepted", "job_id": job_id}


def start_queue_worker() -> threading.Thread | None:
    """Starts the Redis queue consumer for this process (None when the queue is disabled)."""
    queue = get_task_queue()
    if queue is None:
        return None
    thread = threading.Thread(
        target=run_worker,
        args=(queue, _run_queued_job, _on_dead_task),
        kwargs={"lock": _exec_lock},
        name="queue-worker",
        daemon=True,
    )
    thread.start()
    logger.info("[ENGINE WORKER] Redis queue consumer started.")
    return thread


def _worker_loop():
    logger.info("[ENGINE WORKER] Worker loop active and listening for batch jobs.")
    while True:
        payload = _batch_queue.get()
        if payload is None:
            break
        try:
            with _exec_lock, trace_context(payload.get("trace_id")):
                _run_legacy_job(payload)
        finally:
            _batch_queue.task_done()


def _run_legacy_job(payload: dict[str, Any]) -> None:
    """In-memory queue job: a failure is final (no retries)."""
    job_id = str(payload.get("job_id", ""))
    if is_job_cancelled(job_id):
        logger.info(f"[ENGINE WORKER] Skipped job {job_id} because it was already cancelled.")
        _cancelled_jobs.discard(job_id)
        return

    t0 = time.time()
    try:
        _execute_job(payload)
    except InterruptedError:
        logger.info(f"[ENGINE WORKER] Job {job_id} cancelled successfully.")
    except Exception as e:
        if is_job_cancelled(job_id):
            logger.info(f"[ENGINE WORKER] Job {job_id} cancelled during exception.")
            return
        duration = round((time.time() - t0) / 60, 4)
        logger.error(f"[ENGINE WORKER] Job {job_id} failed: {e}", exc_info=True)
        try:
            record_job_failure(job_id, str(e), duration)
        except Exception as db_err:
            logger.error(f"[ENGINE WORKER] Failed to record failure of job {job_id}: {db_err}")
        _notify_failure(payload, str(e))
    finally:
        _cancelled_jobs.discard(job_id)


def _run_queued_job(payload: dict[str, Any]) -> None:
    """Redis queue job: raising hands the error to the queue, which retries or dead-letters it."""
    job_id = str(payload.get("job_id", ""))
    _queue_jobs.add(job_id)
    try:
        if is_job_cancelled(job_id):
            logger.info(f"[ENGINE WORKER] Skipped job {job_id} because it was already cancelled.")
            return
        try:
            _execute_job(payload)
        except InterruptedError:
            logger.info(f"[ENGINE WORKER] Job {job_id} cancelled successfully.")
        except Exception as e:
            if is_job_cancelled(job_id):
                logger.info(f"[ENGINE WORKER] Job {job_id} cancelled during exception.")
                return
            logger.error(f"[ENGINE WORKER] Job {job_id} attempt failed: {e}", exc_info=True)
            raise
    finally:
        _queue_jobs.discard(job_id)
        _cancelled_jobs.discard(job_id)
        _cancel_checked.pop(job_id, None)


def _on_dead_task(task: dict) -> None:
    """The queue already dead-lettered the task and marked the job failed; tell the clients."""
    _notify_failure(task["payload"], task.get("last_error", "job failed"))


def _execute_job(payload: dict[str, Any]) -> None:
    """Runs one job end to end: process, persist, notify. Raises on failure."""
    job_id = str(payload.get("job_id", ""))
    task = payload.get("task", "pipeline")
    domain = payload.get("domain", "market")
    skus = payload.get("skus", [])
    backend_url = payload.get("backend_url", "http://backend:8000")
    callback_url = payload.get("callback_url")
    sheet_name = payload.get("sheet_name")
    spreadsheet_id = payload.get("spreadsheet_id")

    t0 = time.time()
    logger.info(
        f"[ENGINE WORKER] Starting execution of job {job_id} (task={task}, domain={domain}, {len(skus)} SKUs)"
    )

    last_progress_time = 0.0
    last_progress_pct = -10.0
    last_stage = None

    def send_progress_update(
        stage: str,
        pct: float,
        eta: int | None = None,
        _cur_job_id=job_id,
        _cur_backend_url=backend_url,
    ):
        nonlocal last_progress_time, last_progress_pct, last_stage
        if is_job_cancelled(_cur_job_id):
            return

        now = time.time()
        is_terminal = pct >= 100.0 or pct <= 0.0
        stage_changed = stage != last_stage
        time_elapsed = (now - last_progress_time) >= 0.4
        pct_changed = abs(pct - last_progress_pct) >= 2.0

        if not (is_terminal or stage_changed or (time_elapsed and pct_changed)):
            return

        last_progress_time = now
        last_progress_pct = pct
        last_stage = stage

        progress_endpoint = (
            f"{_cur_backend_url.rstrip('/')}/api/internal/jobs/{_cur_job_id}/progress"
        )
        try:
            _http_session.post(
                progress_endpoint,
                json={"current_stage": stage, "progress_pct": pct, "eta_seconds": eta},
                timeout=2.0,
            )
        except Exception as e:
            logger.debug(f"Failed to post progress to backend: {e}")

    sync_rules_version()
    # Pin one (rules, rule_version) pair for every chunk of this job: a reload mid-job
    # must not change which rules evaluate it or the version its results are saved under
    rule_snapshot = snapshot_rules(payload.get("tenant_id") or "default")
    rule_version = rule_snapshot.version
    # Execute processing
    if task.lower() == "upload":
        result_payload = process_upload(
            skus=skus,
            outlet_id=sheet_name,  # outlet ID is passed in sheet_name
            job_id=job_id,
            progress_callback=send_progress_update,
            is_cancelled=lambda jid=job_id: is_job_cancelled(jid),
            has_more_jobs=not _batch_queue.empty(),
        )
    else:
        with pinned_rules(rule_snapshot):
            result_payload = process_request(
                skus=skus,
                task=task,
                domain=domain,
                job_id=job_id,
                progress_callback=send_progress_update,
                is_cancelled=lambda jid=job_id: is_job_cancelled(jid),
            )

    if is_job_cancelled(job_id):
        logger.info(f"[ENGINE WORKER] Job {job_id} aborted mid-execution due to cancellation.")
        return

    duration = round((time.time() - t0) / 60, 4)
    res_list = result_payload.get("results", [])

    # Compute statistics
    high, med, low = 0, 0, 0
    task_lower = task.lower()
    for res in res_list:
        if task_lower == "matcher":
            s = res.get("status", "")
            if "High" in s:
                high += 1
            elif "Medium" in s:
                med += 1
            else:
                low += 1
        elif task_lower == "classifier":
            s = res.get("bt_status", "")
            if "AUTO" in s:
                high += 1
            elif "REVIEW" in s:
                med += 1
            else:
                low += 1
        else:  # pipeline
            s = res.get("status", "")
            if "High" in s:
                high += 1
            elif "Medium" in s:
                med += 1
            else:
                low += 1

    match_rate = round((high / max(1, len(res_list))) * 100, 2)
    tenant_id = payload.get("tenant_id") or "default"
    for status, count in (("high", high), ("medium", med), ("low", low)):
        if count:
            SKUS_PROCESSED.labels(domain=domain, status=status, tenant_id=tenant_id).inc(count)

    # Enrich result payload for external callback
    result_payload["job_id"] = job_id
    result_payload["sheet_name"] = sheet_name
    result_payload["task"] = task
    result_payload["spreadsheet_id"] = spreadsheet_id

    # Persist results directly; a DB error raises to the caller (retry or final failure)
    record_job_completion(
        job_id=job_id,
        domain=domain,
        task=task,
        results=res_list,
        rule_version=rule_version,
        duration_minutes=duration,
        high_conf=high,
        med_conf=med,
        low_conf=low,
        match_rate=match_rate,
    )

    # Notify the Backend Gateway so its in-memory progress state is finalized
    complete_endpoint = f"{backend_url.rstrip('/')}/api/internal/jobs/{job_id}/complete"
    try:
        _http_session.post(complete_endpoint, json={"status": "completed"}, timeout=10)
        logger.info(f"[ENGINE WORKER] Job {job_id} completed in {duration}m.")
    except Exception as be_err:
        logger.error(
            f"[ENGINE WORKER] Failed to notify backend of completion for job {job_id}: {be_err}"
        )

    # Post direct Google Sheets callback if provided
    if callback_url:
        t0_cb = time.time()
        try:
            logger.info(f"[ENGINE WORKER] Delivering Google Sheets callback to {callback_url}...")
            cb_resp = _http_session.post(callback_url, json=result_payload, timeout=120)
            cb_dur = int((time.time() - t0_cb) * 1000)
            logger.info(
                f"[ENGINE WORKER] Callback responded with status {cb_resp.status_code} in {cb_dur}ms"
            )
            log_outbound_request(
                url=callback_url,
                method="POST",
                payload=result_payload,
                response_status=cb_resp.status_code,
                response_text=cb_resp.text[:10000] if cb_resp.text else "",
                duration_ms=cb_dur,
                path="/doPost",
            )
        except Exception as cb_err:
            cb_dur = int((time.time() - t0_cb) * 1000)
            logger.error(f"[ENGINE WORKER] Google Sheets callback delivery failed: {cb_err}")
            log_outbound_request(
                url=callback_url,
                method="POST",
                payload=result_payload,
                response_status=500,
                response_text=str(cb_err),
                duration_ms=cb_dur,
                path="/doPost",
            )


def _notify_failure(payload: dict[str, Any], error: str) -> None:
    """Tells the Gateway and (if configured) Google Sheets that the job failed."""
    job_id = str(payload.get("job_id", ""))
    task = payload.get("task", "pipeline")
    backend_url = payload.get("backend_url", "http://backend:8000")
    callback_url = payload.get("callback_url")
    sheet_name = payload.get("sheet_name")
    spreadsheet_id = payload.get("spreadsheet_id")

    fail_endpoint = f"{backend_url.rstrip('/')}/api/internal/jobs/{job_id}/fail"
    try:
        _http_session.post(fail_endpoint, json={"status": "failed"}, timeout=15)
    except Exception as be_err:
        logger.error(f"[ENGINE WORKER] Failed to notify backend of job failure: {be_err}")

    if callback_url:
        err_payload = {
            "job_id": job_id,
            "error": error,
            "sheet_name": sheet_name,
            "spreadsheet_id": spreadsheet_id,
            "task": task,
        }
        t0_cb = time.time()
        try:
            cb_resp = _http_session.post(callback_url, json=err_payload, timeout=30)
            cb_dur = int((time.time() - t0_cb) * 1000)
            log_outbound_request(
                url=callback_url,
                method="POST",
                payload=err_payload,
                response_status=cb_resp.status_code,
                response_text=cb_resp.text[:10000] if cb_resp.text else "",
                duration_ms=cb_dur,
                path="/doPost",
            )
        except Exception as cb_err:
            cb_dur = int((time.time() - t0_cb) * 1000)
            log_outbound_request(
                url=callback_url,
                method="POST",
                payload=err_payload,
                response_status=500,
                response_text=str(cb_err),
                duration_ms=cb_dur,
                path="/doPost",
            )
