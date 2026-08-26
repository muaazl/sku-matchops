"""
SKU MatchOps Backend - ML Engine HTTP Client
Dispatches inference and batch processing workloads to the dedicated ML Engine microservice.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.app.schemas.models import BaseRequest

logger = logging.getLogger("matchops.engine_client")

ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8001").rstrip("/")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://localhost:8000").rstrip("/")

# Set up requests session with connection pooling and light retry policy
_session = requests.Session()
_retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[502, 503, 504])
_session.mount("http://", HTTPAdapter(max_retries=_retries))
_session.mount("https://", HTTPAdapter(max_retries=_retries))

_models_status_cache: Dict[str, Any] = {}
_models_status_cache_time: float = 0.0


def is_engine_reachable() -> bool:
    """Checks if the ML Engine microservice is responding to health checks."""
    try:
        resp = _session.get(f"{ENGINE_URL}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def get_engine_health() -> Dict[str, Any]:
    """Fetches health and loaded model status from the ML Engine."""
    try:
        resp = _session.get(f"{ENGINE_URL}/health", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "unhealthy", "status_code": resp.status_code}
    except Exception as e:
        logger.warning(f"Failed to fetch health from ML Engine ({ENGINE_URL}): {e}")
        return {"status": "offline", "error": str(e), "engine_url": ENGINE_URL}


def get_models_status() -> Dict[str, Any]:
    """Fetches detailed model statuses from the ML Engine with short TTL cache."""
    global _models_status_cache, _models_status_cache_time
    now = time.time()
    if _models_status_cache and (now - _models_status_cache_time) < 3.0:
        return _models_status_cache

    try:
        resp = _session.get(f"{ENGINE_URL}/models-status", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            _models_status_cache = data
            _models_status_cache_time = now
            return data
        return {"status": "error", "status_code": resp.status_code}
    except Exception as e:
        return {"status": "offline", "error": str(e)}


def trigger_load_models() -> Dict[str, Any]:
    """Triggers background model pre-loading on the ML Engine."""
    global _models_status_cache_time
    _models_status_cache_time = 0.0  # Invalidate cache immediately on load trigger
    try:
        resp = _session.post(f"{ENGINE_URL}/load-models", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "detail": resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def reload_engine_models() -> Dict[str, Any]:
    """Forces cache clearing and model reload on the ML Engine."""
    global _models_status_cache_time
    _models_status_cache_time = 0.0
    try:
        resp = _session.post(f"{ENGINE_URL}/engine/reload-models", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "detail": resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def dispatch_batch_job(job_id: str, request: BaseRequest, task: str) -> Dict[str, Any]:
    """
    Dispatches a batch job to the dedicated ML Engine microservice for async execution.
    """
    skus_payload = [
        {
            "name": sku.name,
            "price": sku.price or 0.0,
            "description": sku.description or "",
            "category": sku.category or ""
        }
        for sku in request.skus
    ]

    payload = {
        "job_id": str(job_id),
        "task": task,
        "domain": request.domain or "market",
        "skus": skus_payload,
        "backend_url": BACKEND_INTERNAL_URL,
        "callback_url": request.callback_url,
        "sheet_name": request.sheet_name,
        "spreadsheet_id": request.spreadsheet_id
    }

    try:
        logger.info(f"Dispatching job {job_id} ({len(skus_payload)} SKUs) to Engine at {ENGINE_URL}/engine/process-batch")
        resp = _session.post(f"{ENGINE_URL}/engine/process-batch", json=payload, timeout=10.0)
        if resp.status_code in (200, 202):
            return resp.json()
        error_msg = f"ML Engine rejected job {job_id} with status {resp.status_code}: {resp.text}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except requests.RequestException as re:
        error_msg = f"Failed to connect to ML Engine at {ENGINE_URL}: {re}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def run_single(
    sku_name: str,
    domain: str = "market",
    task: str = "pipeline",
    price: float = 0.0,
    description: str = "",
    category: str = ""
) -> Dict[str, Any]:
    """Runs synchronous single-SKU inference on the ML Engine."""
    payload = {
        "sku_name": sku_name,
        "domain": domain,
        "task": task,
        "price": price,
        "description": description,
        "category": category
    }

    try:
        resp = _session.post(f"{ENGINE_URL}/engine/run-single", json=payload, timeout=60.0)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Engine single-run returned {resp.status_code}: {resp.text}")
    except requests.RequestException as re:
        raise RuntimeError(f"Could not reach ML Engine at {ENGINE_URL}: {re}")


def suggest_tags(
    sku_name: str,
    domain: str = "market",
    current_bt: str = "",
    exclude_bt: str = "",
    exclude_gk: str = ""
) -> Dict[str, Any]:
    """Requests template-aware tag suggestions from the ML Engine."""
    payload = {
        "sku_name": sku_name,
        "domain": domain,
        "current_bt": current_bt,
        "exclude_bt": exclude_bt,
        "exclude_gk": exclude_gk
    }

    try:
        resp = _session.post(f"{ENGINE_URL}/engine/suggest", json=payload, timeout=15.0)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Engine suggest returned {resp.status_code}: {resp.text}")
    except requests.RequestException as re:
        raise RuntimeError(f"Could not reach ML Engine at {ENGINE_URL}: {re}")


def cancel_engine_job(job_id: str) -> bool:
    """Notifies the ML Engine to abort processing for the given job."""
    try:
        resp = _session.post(f"{ENGINE_URL}/engine/jobs/{job_id}/cancel", timeout=5.0)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Could not notify ML Engine of job {job_id} cancellation: {e}")
        return False


def list_vector_collections() -> Dict[str, Any]:
    """Fetches vector database collection names from the ML Engine."""
    try:
        resp = _session.get(f"{ENGINE_URL}/engine/vector-db/collections", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Engine vector collections returned {resp.status_code}: {resp.text}")
    except requests.RequestException as re:
        raise RuntimeError(f"Could not reach ML Engine at {ENGINE_URL}: {re}")


def get_vector_collection(name: str) -> Dict[str, Any]:
    """Fetches info for a specific vector collection from the ML Engine."""
    try:
        resp = _session.get(f"{ENGINE_URL}/engine/vector-db/collections/{name}", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Engine get vector collection returned {resp.status_code}: {resp.text}")
    except requests.RequestException as re:
        raise RuntimeError(f"Could not reach ML Engine at {ENGINE_URL}: {re}")


def search_vector_collection(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatches vector search request to the ML Engine."""
    try:
        resp = _session.post(f"{ENGINE_URL}/engine/vector-db/collections/{name}/search", json=payload, timeout=15.0)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Engine vector search returned {resp.status_code}: {resp.text}")
    except requests.RequestException as re:
        raise RuntimeError(f"Could not reach ML Engine at {ENGINE_URL}: {re}")


def run_sku_audit(
    sku_name: str,
    domain: str = "market",
    task: str = "pipeline",
    price: float = 0.0,
    description: str = "",
    category: str = ""
) -> Dict[str, Any]:
    """Runs single-SKU diagnostic audit on the ML Engine."""
    payload = {
        "sku_name": sku_name,
        "domain": domain,
        "task": task,
        "price": price,
        "description": description,
        "category": category
    }

    try:
        resp = _session.post(f"{ENGINE_URL}/engine/audit", json=payload, timeout=60.0)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Engine audit returned {resp.status_code}: {resp.text}")
    except requests.RequestException as re:
        raise RuntimeError(f"Could not reach ML Engine at {ENGINE_URL}: {re}")

