"""
SKU MatchOps Engine - Background Job Worker Runner
Executes async batch jobs, triggers registered progress/completion/failure callbacks,
and delivers external Google Sheets webhook callbacks.
"""

import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from engine.core import config
from engine.core.observability import SKUS_PROCESSED, bind_trace_context
from engine.core.onnx_factory import trim_process_memory
from engine.core.security import validate_outbound_url
from engine.pipelines.queue_pipeline import (
    get_task_queue,
    register_failure_notifier,
    release_active_tasks,
    run_worker,
)
from engine.pipelines.request_pipeline import process_request
from engine.pipelines.rules_loader_pipeline import (
    pinned_rules,
    snapshot_rules,
    sync_rules_version,
)
from engine.pipelines.upload_pipeline import process_upload

logger = logging.getLogger("matchops.engine.worker")

_batch_queue: queue.Queue = queue.Queue()
_state_lock = threading.Lock()
_cancelled_jobs: set[str] = set()
_worker_threads: list[threading.Thread] = []
_queue_workers: list[threading.Thread] = []

_http_session = requests.Session()
_sheets_retries = Retry(
    total=3, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=["GET", "POST"]
)
_http_session.mount(
    "https://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_sheets_retries)
)

_on_progress_callback: Callable[..., None] | None = None
_on_complete_callback: Callable[..., None] | None = None
_on_fail_callback: Callable[..., None] | None = None


def register_job_callbacks(
    on_progress: Callable[..., None] | None = None,
    on_complete: Callable[..., None] | None = None,
    on_fail: Callable[..., None] | None = None,
) -> None:
    """Registers transport-layer job status callbacks (e.g. from engine.grpc.callback_client)."""
    global _on_progress_callback, _on_complete_callback, _on_fail_callback
    _on_progress_callback = on_progress
    _on_complete_callback = on_complete
    _on_fail_callback = on_fail


class _ExecGate:
    """Thread-aware mutex allowing early release after CPU inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tls = threading.local()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if getattr(self._tls, "held", False):
            return True
        ok = self._lock.acquire(blocking, timeout) if timeout >= 0 else self._lock.acquire(blocking)
        if ok:
            self._tls.held = True
        return ok

    def release(self) -> None:
        if getattr(self._tls, "held", False):
            self._tls.held = False
            self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> "_ExecGate":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


_exec_lock = _ExecGate()
_worker_stop = threading.Event()
_queue_jobs: set[str] = set()
_cancel_checked: dict[str, tuple[float, bool]] = {}


def cancel_job(job_id: str) -> bool:
    """Marks a job as cancelled so processing aborts immediately."""
    jid = str(job_id)
    with _state_lock:
        _cancelled_jobs.add(jid)
    try:
        task_q = get_task_queue()
        if task_q is not None:
            task_q.request_cancel(jid)
    except Exception as e:
        logger.debug(f"[ENGINE WORKER] Could not broadcast cancellation of {job_id}: {e}")
    logger.info(f"[ENGINE WORKER] Job {job_id} cancellation requested.")
    return True


def is_job_cancelled(job_id: str) -> bool:
    jid = str(job_id)
    with _state_lock:
        if jid in _cancelled_jobs:
            return True
        if jid not in _queue_jobs:
            return False
        checked_at, prev_cancelled = _cancel_checked.get(jid, (0.0, False))
    if time.time() - checked_at >= 1.0:
        try:
            task_q = get_task_queue()
            cancelled = task_q.is_cancelled(jid) if task_q is not None else False
        except Exception:
            cancelled = False
        with _state_lock:
            _cancel_checked[jid] = (time.time(), cancelled)
            if cancelled:
                _cancelled_jobs.add(jid)
        return cancelled
    return prev_cancelled


def enqueue_batch_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Fallback path placing a job onto this process's in-memory queue."""
    global _worker_threads
    job_id = str(payload.get("job_id", ""))
    with _state_lock:
        _cancelled_jobs.discard(job_id)
    _batch_queue.put(payload)

    with _state_lock:
        _worker_threads = [t for t in _worker_threads if t.is_alive()]
        while len(_worker_threads) < 2:
            t = threading.Thread(
                target=_worker_loop, name=f"batch-worker-{len(_worker_threads)}", daemon=True
            )
            t.start()
            _worker_threads.append(t)

    logger.info(f"[ENGINE WORKER] Enqueued job {job_id} ({len(payload.get('skus', []))} SKUs)")
    return {"status": "accepted", "job_id": job_id}


def shutdown_workers() -> None:
    """Signals Redis queue consumers to stop and releases in-flight idempotent leases."""
    _worker_stop.set()
    release_active_tasks()


def start_queue_worker() -> threading.Thread | None:
    """Ensures 2 Redis queue consumer threads are running; restarts only dead workers."""
    global _queue_workers
    task_q = get_task_queue()
    if task_q is None:
        return None
    with _state_lock:
        _worker_stop.clear()
        _queue_workers = [t for t in _queue_workers if t.is_alive()]
        while len(_queue_workers) < 2:
            existing_names = {t.name for t in _queue_workers}
            slot = next(i for i in range(2) if f"queue-worker-{i}" not in existing_names)
            thread = threading.Thread(
                target=run_worker,
                args=(task_q, _run_queued_job),
                kwargs={"lock": _exec_lock, "stop": _worker_stop},
                name=f"queue-worker-{slot}",
                daemon=True,
            )
            thread.start()
            _queue_workers.append(thread)
        logger.info("[ENGINE WORKER] Redis queue consumers started.")
        return _queue_workers[0] if _queue_workers else None


def _worker_loop() -> None:
    logger.info("[ENGINE WORKER] Worker loop active and listening for batch jobs.")
    while True:
        payload = _batch_queue.get()
        if payload is None:
            break
        try:
            with (
                _exec_lock,
                bind_trace_context(payload.get("trace_id")),
            ):
                _run_legacy_job(payload)
        finally:
            _batch_queue.task_done()


def _run_legacy_job(payload: dict[str, Any]) -> None:
    """In-memory queue job: a failure is final (no retries)."""
    job_id = str(payload.get("job_id", ""))
    if is_job_cancelled(job_id):
        logger.info(f"[ENGINE WORKER] Skipped job {job_id} because it was already cancelled.")
        with _state_lock:
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
        logger.error(f"[ENGINE WORKER] Job {job_id} failed after {duration}m: {e}", exc_info=True)
        _notify_failure(payload, str(e))
    finally:
        with _state_lock:
            _cancelled_jobs.discard(job_id)
            _cancel_checked.pop(job_id, None)
        trim_process_memory()


def _run_queued_job(payload: dict[str, Any]) -> None:
    """Redis queue job: raising hands the error to the queue, which retries or dead-letters it."""
    with bind_trace_context(payload.get("trace_id")):
        job_id = str(payload.get("job_id", ""))
        with _state_lock:
            _queue_jobs.add(job_id)
        try:
            if _worker_stop.is_set():
                raise InterruptedError("Worker shutting down")
            if is_job_cancelled(job_id):
                logger.info(
                    f"[ENGINE WORKER] Skipped job {job_id} because it was already cancelled."
                )
                return
            try:
                _execute_job(payload)
            except InterruptedError:
                if _worker_stop.is_set():
                    raise
                logger.info(f"[ENGINE WORKER] Job {job_id} cancelled successfully.")
            except Exception as e:
                if _worker_stop.is_set():
                    raise
                if is_job_cancelled(job_id):
                    logger.info(f"[ENGINE WORKER] Job {job_id} cancelled during exception.")
                    return
                logger.error(f"[ENGINE WORKER] Job {job_id} attempt failed: {e}", exc_info=True)
                raise
        finally:
            with _state_lock:
                _queue_jobs.discard(job_id)
                _cancelled_jobs.discard(job_id)
                _cancel_checked.pop(job_id, None)
            trim_process_memory()


def _execute_job(payload: dict[str, Any]) -> None:
    """Runs one job end to end: process, deliver webhook, notify registered completion callback."""
    job_id = str(payload.get("job_id", ""))
    task = payload.get("task", "pipeline")
    domain = payload.get("domain", "market")
    skus = payload.get("skus", [])
    backend_grpc_url = payload.get("backend_grpc_url") or config.BACKEND_GRPC_URL
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
        _cur_backend_grpc_url=backend_grpc_url,
    ):
        nonlocal last_progress_time, last_progress_pct, last_stage
        if _worker_stop.is_set() or is_job_cancelled(_cur_job_id):
            return

        now = time.time()
        is_terminal = pct >= 100.0 or pct <= 0.0
        stage_changed = stage != last_stage
        time_elapsed = (now - last_progress_time) >= 0.4
        pct_changed = abs(pct - last_progress_pct) >= 0.25 or (now - last_progress_time) >= 1.5

        if not (is_terminal or stage_changed or (time_elapsed and pct_changed)):
            return

        last_progress_time = now
        last_progress_pct = pct
        last_stage = stage

        if _on_progress_callback is not None:
            _on_progress_callback(_cur_backend_grpc_url, _cur_job_id, stage, float(pct), eta)

    send_progress_update("uploading" if task.lower() == "upload" else "embedding", 0.0)
    with _exec_lock:
        sync_rules_version()
        rule_snapshot = snapshot_rules()
        rule_version = rule_snapshot.version
        if task.lower() == "upload":
            result_payload = process_upload(
                skus=skus,
                outlet_id=sheet_name,
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
                    is_cancelled=lambda jid=job_id: _worker_stop.is_set() or is_job_cancelled(jid),
                )

    if _worker_stop.is_set() and task.lower() != "upload":
        raise InterruptedError("Worker shutting down")
    if is_job_cancelled(job_id):
        logger.info(f"[ENGINE WORKER] Job {job_id} aborted mid-execution due to cancellation.")
        return

    duration = round((time.time() - t0) / 60, 4)
    res_list = result_payload.get("results", [])

    high, med, low = 0, 0, 0
    task_lower = task.lower()
    for res in res_list:
        s = res.get("bt_status", "") if task_lower == "classifier" else res.get("status", "")
        if ("AUTO" in s) if task_lower == "classifier" else ("High" in s):
            high += 1
        elif ("REVIEW" in s) if task_lower == "classifier" else ("Medium" in s):
            med += 1
        else:
            low += 1

    match_rate = round((high / max(1, len(res_list))) * 100, 2)
    for status, count in (("high", high), ("medium", med), ("low", low)):
        if count:
            SKUS_PROCESSED.labels(domain=domain, status=status).inc(count)

    result_payload["job_id"] = job_id
    result_payload["sheet_name"] = sheet_name
    result_payload["task"] = task
    result_payload["spreadsheet_id"] = spreadsheet_id

    webhook_log = None
    if callback_url:
        t0_cb = time.time()
        try:
            validate_outbound_url(callback_url)
            logger.info(f"[ENGINE WORKER] Delivering Google Sheets callback to {callback_url}...")
            cb_resp = _http_session.post(callback_url, json=result_payload, timeout=45)
            cb_dur = int((time.time() - t0_cb) * 1000)
            logger.info(
                f"[ENGINE WORKER] Callback responded with status {cb_resp.status_code} in {cb_dur}ms"
            )
            webhook_log = {
                "url": callback_url,
                "method": "POST",
                "payload": result_payload,
                "response_status": cb_resp.status_code,
                "response_text": cb_resp.text[:10000] if cb_resp.text else "",
                "duration_ms": cb_dur,
                "path": "/doPost",
            }
        except Exception as cb_err:
            cb_dur = int((time.time() - t0_cb) * 1000)
            logger.error(f"[ENGINE WORKER] Google Sheets callback delivery failed: {cb_err}")
            webhook_log = {
                "url": callback_url,
                "method": "POST",
                "payload": result_payload,
                "response_status": 500,
                "response_text": str(cb_err),
                "duration_ms": cb_dur,
                "path": "/doPost",
            }

    if _on_complete_callback is not None:
        try:
            _on_complete_callback(
                backend_grpc_url=backend_grpc_url,
                job_id=job_id,
                domain=domain,
                task=task,
                rule_version=rule_version,
                duration_minutes=float(duration),
                high_conf=high,
                med_conf=med,
                low_conf=low,
                match_rate=float(match_rate),
                webhook_log_json=json.dumps(webhook_log, default=str) if webhook_log else "",
                results=res_list,
            )
            logger.info(f"[ENGINE WORKER] Job {job_id} completed in {duration}m.")
        except Exception as be_err:
            logger.error(
                f"[ENGINE WORKER] Failed to notify Gateway of completion for job {job_id}: {be_err}"
            )
            raise


def _notify_failure(payload: dict[str, Any], error: str) -> None:
    """Delivers external error webhook and invokes the registered terminal failure callback."""
    job_id = str(payload.get("job_id", ""))
    task = payload.get("task", "pipeline")
    backend_grpc_url = payload.get("backend_grpc_url") or config.BACKEND_GRPC_URL
    callback_url = payload.get("callback_url")
    sheet_name = payload.get("sheet_name")
    spreadsheet_id = payload.get("spreadsheet_id")

    if callback_url:
        err_payload = {
            "job_id": job_id,
            "error": error,
            "sheet_name": sheet_name,
            "spreadsheet_id": spreadsheet_id,
            "task": task,
        }
        try:
            validate_outbound_url(callback_url)
            _http_session.post(callback_url, json=err_payload, timeout=30)
        except Exception as cb_err:
            logger.error(f"[ENGINE WORKER] Google Sheets callback delivery failed: {cb_err}")

    if _on_fail_callback is not None:
        _on_fail_callback(backend_grpc_url=backend_grpc_url, job_id=job_id, error_message=error)


register_failure_notifier(_notify_failure)
