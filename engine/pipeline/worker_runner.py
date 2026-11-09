"""
SKU MatchOps Engine - Background Job Worker Runner
Executes async batch jobs, emits progress/completion/failure gRPC callbacks to the Gateway,
and delivers external Google Sheets webhook callbacks.
"""

import json
import logging
import queue
import threading
import time
from typing import Any

import grpc
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from engine import config
from engine.core.grpc_utils import GRPC_CHANNEL_OPTIONS, grpc_metadata, result_to_pipeline_proto
from engine.core.observability import SKUS_PROCESSED, trace_context
from engine.pipeline.processor import process_request
from engine.pipeline.task_queue import get_task_queue, release_active_tasks, run_worker
from engine.pipeline.uploader import process_upload
from engine.protos import engine_service_pb2 as pb2
from engine.protos import engine_service_pb2_grpc as pb2_grpc
from engine.rules_engine.loader import pinned_rules, snapshot_rules, sync_rules_version

logger = logging.getLogger("matchops.engine.worker")

_batch_queue = queue.Queue()
_cancelled_jobs: set[str] = set()
_worker_threads: list[threading.Thread] = []

# Persistent HTTP session used exclusively for external Google Apps Script webhook delivery.
# Apps Script /exec randomly hangs ~30% of requests; doPost overwrites the same cells, so re-POSTing is safe.
_http_session = requests.Session()
_sheets_retries = Retry(
    total=3, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=["GET", "POST"]
)
_http_session.mount(
    "http://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_sheets_retries)
)
_http_session.mount(
    "https://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_sheets_retries)
)

_stub_lock = threading.Lock()
_callback_channels: dict[str, grpc.Channel] = {}
_callback_stubs: dict[str, pb2_grpc.BackendCallbackServiceStub] = {}


class _ExecGate:
    """Thread-aware mutex allowing early release after CPU inference so slow HTTP webhooks never block queued jobs."""

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


_exec_lock = _ExecGate()  # one CPU inference job at a time per process; released before webhook I/O
_worker_stop = threading.Event()  # set on SIGTERM/SIGINT so workers abort and release leases
_queue_jobs: set[str] = set()  # jobs taken from the Redis queue
_cancel_checked: dict[str, tuple[float, bool]] = {}


def _get_backend_stub(target: str | None = None) -> pb2_grpc.BackendCallbackServiceStub:
    """Returns a cached gRPC stub for the Gateway's BackendCallbackService."""
    addr = target or config.BACKEND_GRPC_URL
    with _stub_lock:
        stub = _callback_stubs.get(addr)
        if stub is None:
            channel = grpc.insecure_channel(addr, options=list(GRPC_CHANNEL_OPTIONS))
            stub = pb2_grpc.BackendCallbackServiceStub(channel)
            _callback_channels[addr] = channel
            _callback_stubs[addr] = stub
        return stub


def cancel_job(job_id: str) -> bool:
    """Marks a job as cancelled so processing aborts immediately (on any worker, via Redis)."""
    _cancelled_jobs.add(str(job_id))
    try:
        task_q = get_task_queue()
        if task_q is not None:
            task_q.request_cancel(str(job_id))
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
            task_q = get_task_queue()
            cancelled = task_q.is_cancelled(jid) if task_q is not None else False
        except Exception:
            cancelled = False
        _cancel_checked[jid] = (time.time(), cancelled)
        if cancelled:
            _cancelled_jobs.add(jid)
    return cancelled


def enqueue_batch_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Fallback path: puts a job onto this process's in-memory queue (used when the Gateway
    could not reach the Redis queue and dispatched over gRPC)."""
    global _worker_threads
    job_id = str(payload.get("job_id", ""))
    _cancelled_jobs.discard(job_id)
    _batch_queue.put(payload)

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
    """Signals Redis queue consumers to stop and immediately releases in-flight idempotent leases."""
    _worker_stop.set()
    release_active_tasks()


def start_queue_worker() -> threading.Thread | None:
    """Starts Redis queue consumers (2 threads gated by _exec_lock so webhook I/O overlaps with next job's CPU run)."""
    task_q = get_task_queue()
    if task_q is None:
        return None
    _worker_stop.clear()
    threads: list[threading.Thread] = []
    for idx in range(2):
        thread = threading.Thread(
            target=run_worker,
            args=(task_q, _run_queued_job),
            kwargs={"lock": _exec_lock, "stop": _worker_stop},
            name=f"queue-worker-{idx}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    logger.info("[ENGINE WORKER] Redis queue consumers started.")
    return threads[0]


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
        logger.error(f"[ENGINE WORKER] Job {job_id} failed after {duration}m: {e}", exc_info=True)
        _notify_failure(payload, str(e))
    finally:
        _cancelled_jobs.discard(job_id)
        _cancel_checked.pop(job_id, None)


def _run_queued_job(payload: dict[str, Any]) -> None:
    """Redis queue job: raising hands the error to the queue, which retries or dead-letters it."""
    job_id = str(payload.get("job_id", ""))
    _queue_jobs.add(job_id)
    try:
        if _worker_stop.is_set():
            raise InterruptedError("Worker shutting down")
        if is_job_cancelled(job_id):
            logger.info(f"[ENGINE WORKER] Skipped job {job_id} because it was already cancelled.")
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
        _queue_jobs.discard(job_id)
        _cancelled_jobs.discard(job_id)
        _cancel_checked.pop(job_id, None)


def _execute_job(payload: dict[str, Any]) -> None:
    """Runs one job end to end: process, deliver webhook, notify Gateway via gRPC."""
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
        pct_changed = abs(pct - last_progress_pct) >= 2.0

        if not (is_terminal or stage_changed or (time_elapsed and pct_changed)):
            return

        last_progress_time = now
        last_progress_pct = pct
        last_stage = stage

        try:
            stub = _get_backend_stub(_cur_backend_grpc_url)
            stub.ReportJobProgress(
                pb2.JobProgressRequest(
                    job_id=_cur_job_id,
                    current_stage=stage,
                    progress_pct=float(pct),
                    eta_seconds=int(eta or 0),
                    has_eta=eta is not None,
                ),
                metadata=grpc_metadata(),
                timeout=2.0,
            )
        except Exception as e:
            logger.debug(f"Failed to report progress via gRPC: {e}")

    send_progress_update("uploading" if task.lower() == "upload" else "embedding", 0.0)
    with _exec_lock:
        sync_rules_version()
        # Pin one (rules, rule_version) pair for every chunk of this job: a reload mid-job
        # must not change which rules evaluate it or the version its results are saved under
        rule_snapshot = snapshot_rules(payload.get("tenant_id") or "default")
        rule_version = rule_snapshot.version
        # Execute CPU inference; _exec_lock is released immediately upon exit so slow outbound
        # Google Sheets webhooks never stall CPU execution for the next queued job.
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
                    is_cancelled=lambda jid=job_id: _worker_stop.is_set() or is_job_cancelled(jid),
                )

    if _worker_stop.is_set() and task.lower() != "upload":
        raise InterruptedError("Worker shutting down")
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

    webhook_log = None
    # Post direct Google Sheets callback if provided
    if callback_url:
        t0_cb = time.time()
        try:
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

    # Notify the Gateway over gRPC so it persists results and finalizes progress
    try:
        stub = _get_backend_stub(backend_grpc_url)
        stub.CompleteJob(
            pb2.JobCompleteRequest(
                job_id=job_id,
                status="completed",
                domain=domain,
                task=task,
                rule_version=rule_version,
                duration_minutes=float(duration),
                high_conf=high,
                med_conf=med,
                low_conf=low,
                match_rate=float(match_rate),
                webhook_log_json=json.dumps(webhook_log, default=str) if webhook_log else "",
                results=[result_to_pipeline_proto(r) for r in res_list],
            ),
            metadata=grpc_metadata(),
            timeout=30.0,
        )
        logger.info(f"[ENGINE WORKER] Job {job_id} completed in {duration}m.")
    except Exception as be_err:
        logger.error(
            f"[ENGINE WORKER] Failed to notify Gateway of completion for job {job_id}: {be_err}"
        )
        raise


def _notify_failure(payload: dict[str, Any], error: str) -> None:
    """Tells the Gateway via gRPC and (if configured) Google Sheets that the job failed."""
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
            _http_session.post(callback_url, json=err_payload, timeout=30)
        except Exception as cb_err:
            logger.error(f"[ENGINE WORKER] Google Sheets callback delivery failed: {cb_err}")

    try:
        stub = _get_backend_stub(backend_grpc_url)
        stub.FailJob(
            pb2.JobFailRequest(
                job_id=job_id,
                status="failed",
                error_message=error,
                duration_minutes=0.0,
                is_cancelled=False,
            ),
            metadata=grpc_metadata(),
            timeout=15.0,
        )
    except Exception as be_err:
        logger.error(f"[ENGINE WORKER] Failed to notify Gateway of job failure: {be_err}")
