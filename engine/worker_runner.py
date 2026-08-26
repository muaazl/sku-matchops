"""
SKU MatchOps Engine - Background Job Worker Runner
Executes async batch jobs, emits incremental HTTP callbacks to the Backend Gateway,
and handles Google Sheets webhook callbacks.
"""

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from engine.processor import process_request

logger = logging.getLogger("matchops.engine.worker")

_batch_queue = queue.Queue()
_cancelled_jobs: Set[str] = set()
_running_jobs: Dict[str, Dict[str, Any]] = {}
_worker_thread: Optional[threading.Thread] = None

# Persistent HTTP session with connection pooling
_http_session = requests.Session()
_retries = Retry(total=1, backoff_factor=0.1, status_forcelist=[502, 503, 504])
_http_session.mount("http://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_retries))
_http_session.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_retries))


def cancel_job(job_id: str) -> bool:
    """Marks a job as cancelled so processing aborts immediately."""
    _cancelled_jobs.add(str(job_id))
    logger.info(f"[ENGINE WORKER] Job {job_id} cancellation requested.")
    return True


def is_job_cancelled(job_id: str) -> bool:
    return str(job_id) in _cancelled_jobs


def enqueue_batch_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Puts a batch job onto the engine execution queue."""
    global _worker_thread
    job_id = str(payload.get("job_id", ""))
    _cancelled_jobs.discard(job_id)
    _batch_queue.put(payload)
    
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()
        
    logger.info(f"[ENGINE WORKER] Enqueued job {job_id} ({len(payload.get('skus', []))} SKUs)")
    return {"status": "accepted", "job_id": job_id}


def _worker_loop():
    logger.info("[ENGINE WORKER] Worker loop active and listening for batch jobs.")
    while True:
        payload = _batch_queue.get()
        if payload is None:
            break

        job_id = str(payload.get("job_id", ""))
        task = payload.get("task", "pipeline")
        domain = payload.get("domain", "market")
        skus = payload.get("skus", [])
        backend_url = payload.get("backend_url", "http://backend:8000")
        callback_url = payload.get("callback_url")
        sheet_name = payload.get("sheet_name")
        spreadsheet_id = payload.get("spreadsheet_id")

        if is_job_cancelled(job_id):
            logger.info(f"[ENGINE WORKER] Skipped job {job_id} because it was already cancelled.")
            _batch_queue.task_done()
            continue

        t0 = time.time()
        logger.info(f"[ENGINE WORKER] Starting execution of job {job_id} (task={task}, domain={domain}, {len(skus)} SKUs)")

        last_progress_time = 0.0
        last_progress_pct = -10.0
        last_stage = None

        def send_progress_update(stage: str, pct: float, eta: Optional[int] = None):
            nonlocal last_progress_time, last_progress_pct, last_stage
            if is_job_cancelled(job_id):
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

            progress_endpoint = f"{backend_url.rstrip('/')}/api/internal/jobs/{job_id}/progress"
            try:
                _http_session.post(
                    progress_endpoint,
                    json={"current_stage": stage, "progress_pct": pct, "eta_seconds": eta},
                    timeout=2.0
                )
            except Exception as e:
                logger.debug(f"Failed to post progress to backend: {e}")

        try:
            # Execute processing
            result_payload = process_request(
                skus=skus,
                task=task,
                domain=domain,
                job_id=job_id,
                progress_callback=send_progress_update,
                is_cancelled=lambda: is_job_cancelled(job_id)
            )

            if is_job_cancelled(job_id):
                logger.info(f"[ENGINE WORKER] Job {job_id} aborted mid-execution due to cancellation.")
                _batch_queue.task_done()
                continue

            duration = round((time.time() - t0) / 60, 4)
            res_list = result_payload.get("results", [])

            # Compute statistics
            high, med, low = 0, 0, 0
            task_lower = task.lower()
            for res in res_list:
                if task_lower == "matcher":
                    s = res.get("status", "")
                    if "High" in s: high += 1
                    elif "Medium" in s: med += 1
                    else: low += 1
                elif task_lower == "classifier":
                    s = res.get("bt_status", "")
                    if "AUTO" in s: high += 1
                    elif "REVIEW" in s: med += 1
                    else: low += 1
                else: # pipeline
                    s = res.get("status", "")
                    if "High" in s: high += 1
                    elif "Medium" in s: med += 1
                    else: low += 1

            match_rate = round((high / max(1, len(res_list))) * 100, 2)

            # Enrich result payload for external callback
            result_payload["job_id"] = job_id
            result_payload["sheet_name"] = sheet_name
            result_payload["task"] = task
            result_payload["spreadsheet_id"] = spreadsheet_id

            # Post completion to Backend Gateway
            complete_endpoint = f"{backend_url.rstrip('/')}/api/internal/jobs/{job_id}/complete"
            try:
                _http_session.post(
                    complete_endpoint,
                    json={
                        "status": "completed",
                        "duration_minutes": duration,
                        "high_conf": high,
                        "med_conf": med,
                        "low_conf": low,
                        "match_rate": match_rate,
                        "results": res_list,
                        "raw_payload": result_payload,
                        "callback_url": callback_url
                    },
                    timeout=30
                )
                logger.info(f"[ENGINE WORKER] Job {job_id} completed in {duration}m. Results delivered to backend.")
            except Exception as be_err:
                logger.error(f"[ENGINE WORKER] Failed to deliver completion to backend for job {job_id}: {be_err}")

            # Post direct Google Sheets callback if provided
            if callback_url:
                try:
                    logger.info(f"[ENGINE WORKER] Delivering Google Sheets callback to {callback_url}...")
                    cb_resp = _http_session.post(callback_url, json=result_payload, timeout=120)
                    logger.info(f"[ENGINE WORKER] Callback responded with status {cb_resp.status_code}")
                except Exception as cb_err:
                    logger.error(f"[ENGINE WORKER] Google Sheets callback delivery failed: {cb_err}")

        except InterruptedError:
            logger.info(f"[ENGINE WORKER] Job {job_id} cancelled successfully.")
        except Exception as e:
            if is_job_cancelled(job_id):
                logger.info(f"[ENGINE WORKER] Job {job_id} cancelled during exception.")
                _batch_queue.task_done()
                continue

            duration = round((time.time() - t0) / 60, 4)
            logger.error(f"[ENGINE WORKER] Job {job_id} failed: {e}", exc_info=True)

            fail_endpoint = f"{backend_url.rstrip('/')}/api/internal/jobs/{job_id}/fail"
            try:
                _http_session.post(
                    fail_endpoint,
                    json={
                        "status": "failed",
                        "error_message": str(e),
                        "duration_minutes": duration,
                        "callback_url": callback_url
                    },
                    timeout=15
                )
            except Exception as be_err:
                logger.error(f"[ENGINE WORKER] Failed to notify backend of job failure: {be_err}")

            if callback_url:
                err_payload = {
                    "job_id": job_id,
                    "error": str(e),
                    "sheet_name": sheet_name,
                    "spreadsheet_id": spreadsheet_id,
                    "task": task
                }
                try:
                    _http_session.post(callback_url, json=err_payload, timeout=30)
                except Exception:
                    pass

        finally:
            _batch_queue.task_done()
            _cancelled_jobs.discard(job_id)
