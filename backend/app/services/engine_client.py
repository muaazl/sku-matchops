"""
SKU MatchOps Backend - ML Engine gRPC Client
Dispatches inference and batch processing workloads to the dedicated ML Engine microservice via gRPC.
"""

import json
import logging
import os
import time
from typing import Any

import grpc
import requests
from requests.adapters import HTTPAdapter, Retry

from backend.app.core.config import DEFAULT_TENANT
from backend.app.schemas.models import BaseRequest
from engine.core.observability import TRACE_ID
from engine.pipeline.task_queue import get_task_queue
from protos import engine_service_pb2 as pb2
from protos import engine_service_pb2_grpc as pb2_grpc

logger = logging.getLogger("matchops.engine_client")

ENGINE_GRPC_URL = os.getenv("ENGINE_GRPC_URL", "127.0.0.1:50051")
ENGINE_URL = os.getenv("ENGINE_URL", "http://127.0.0.1:8001").rstrip("/")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://127.0.0.1:8000").rstrip("/")

_session = requests.Session()
adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=50,
    max_retries=Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE"],
    ),
)
_session.mount("http://", adapter)
_session.mount("https://", adapter)

_channel: grpc.Channel | None = None
_stub: pb2_grpc.EngineInferenceServiceStub | None = None


def get_stub() -> pb2_grpc.EngineInferenceServiceStub:
    """Returns singleton gRPC stub connected to the ML Engine."""
    global _channel, _stub
    if _stub is None:
        _channel = grpc.insecure_channel(
            ENGINE_GRPC_URL,
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        _stub = pb2_grpc.EngineInferenceServiceStub(_channel)
    return _stub


def _metadata() -> tuple[tuple[str, str], ...]:
    trace_id = TRACE_ID.get()
    return (("x-request-id", trace_id),) if trace_id != "-" else ()


_models_status_cache: dict[str, Any] = {}
_models_status_cache_time: float = 0.0


def is_engine_reachable() -> bool:
    """Checks if the ML Engine microservice is responding to health checks."""
    try:
        stub = get_stub()
        resp = stub.GetHealth(pb2.Empty(), timeout=2.0)
        return resp.status == "ok"
    except Exception:
        return False


def get_engine_health() -> dict[str, Any]:
    """Fetches health and loaded model status from the ML Engine."""
    try:
        stub = get_stub()
        resp = stub.GetHealth(pb2.Empty(), timeout=5.0)
        return {
            "status": resp.status,
            "service": resp.service,
            "loaded_domains": list(resp.loaded_domains),
            "all_models_ready": resp.all_models_ready,
        }
    except Exception as e:
        logger.warning(f"Failed to fetch health from ML Engine ({ENGINE_GRPC_URL}): {e}")
        return {"status": "offline", "error": str(e), "engine_url": ENGINE_GRPC_URL}


def get_models_status() -> dict[str, Any]:
    """Fetches detailed model statuses from the ML Engine with short TTL cache."""
    global _models_status_cache, _models_status_cache_time
    now = time.time()
    ttl = 1.0 if _models_status_cache.get("loading_in_progress") else 3.0
    if _models_status_cache and (now - _models_status_cache_time) < ttl:
        return _models_status_cache

    try:
        stub = get_stub()
        resp = stub.GetModelsStatus(pb2.Empty(), timeout=5.0)
        details = {
            domain: {"pipeline": s.pipeline, "classifier": s.classifier}
            for domain, s in resp.domains.items()
        }
        loading_in_progress = resp.status == "loading" or any(
            s["pipeline"] == "loading" or s["classifier"] == "training"
            for s in details.values()
        )
        market_ready = (
            details.get("market", {}).get("pipeline") == "ready"
            and details.get("market", {}).get("classifier") == "ready"
        )
        food_ready = (
            details.get("food", {}).get("pipeline") == "ready"
            and details.get("food", {}).get("classifier") == "ready"
        )
        is_ready = bool(market_ready and food_ready and not loading_in_progress)
        status_str = "ready" if is_ready else ("loading" if loading_in_progress else "unloaded")

        data = {
            "status": status_str,
            "loaded": is_ready,
            "all_models_ready": is_ready,
            "loading_in_progress": loading_in_progress,
            "details": details,
        }
        _models_status_cache = data
        _models_status_cache_time = now
        return data
    except Exception as e:
        return {
            "status": "offline",
            "loaded": False,
            "all_models_ready": False,
            "loading_in_progress": False,
            "error": str(e),
        }


def trigger_load_models() -> dict[str, Any]:
    """Triggers background model pre-loading on the ML Engine."""
    global _models_status_cache_time
    _models_status_cache_time = 0.0
    try:
        stub = get_stub()
        resp = stub.ManageModels(pb2.ManageModelsRequest(action="load"), timeout=5.0)
        return {"status": resp.status, "message": resp.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def reload_engine_models() -> dict[str, Any]:
    """Forces cache clearing and model reload on the ML Engine."""
    global _models_status_cache_time
    _models_status_cache_time = 0.0
    try:
        stub = get_stub()
        resp = stub.ManageModels(pb2.ManageModelsRequest(action="reload"), timeout=5.0)
        return {"status": resp.status, "message": resp.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def refresh_engine_rules(rules_data: list[dict] | None = None, version: str = "") -> dict[str, Any]:
    """Tells the ML Engine to reload rules via gRPC (fallback for when Redis pub/sub is down)."""
    try:
        stub = get_stub()
        req = pb2.RefreshRulesRequest(
            rules_json=json.dumps(rules_data) if rules_data else "", version=version
        )
        resp = stub.RefreshRules(req, timeout=5.0)
        return {"status": resp.status, "message": resp.message}
    except Exception as e:
        logger.warning(f"Failed to notify ML Engine via gRPC to refresh rules: {e}")
        return {"status": "error", "error": str(e)}


def _enqueue_or_post(job_id: str, payload: dict[str, Any], label: str) -> dict[str, Any]:
    """Puts the job on the Redis queue. Falls back to direct HTTP dispatch if queue is disabled."""
    try:
        queue = get_task_queue()
        if queue is not None:
            task_id = queue.enqueue(payload)
            return {"status": "queued", "job_id": str(job_id), "task_id": task_id}
    except Exception as e:
        logger.warning(
            f"Redis queue unavailable ({e}); falling back to HTTP dispatch for {label} {job_id}."
        )

    try:
        r = _session.post(
            f"{ENGINE_URL}/engine/process-batch",
            json=payload,
            timeout=(5.0, 30.0),
        )
        if r.status_code not in (200, 202):
            error_msg = f"Engine rejected {label} {job_id} with status {r.status_code}: {r.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return r.json()
    except requests.exceptions.RequestException as re:
        error_msg = f"Failed to connect to ML Engine at {ENGINE_URL}: {re}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def dispatch_batch_job(
    job_id: str, request: BaseRequest, task: str, tenant_id: str = DEFAULT_TENANT
) -> dict[str, Any]:
    """Dispatches a batch job to the dedicated ML Engine microservice for async execution."""
    skus_payload = [
        {
            "name": sku.name,
            "price": sku.price or 0.0,
            "description": sku.description or "",
            "category": sku.category or "",
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
        "spreadsheet_id": request.spreadsheet_id,
        "tenant_id": tenant_id,
        "trace_id": TRACE_ID.get(),
    }
    return _enqueue_or_post(job_id, payload, "job")


def dispatch_upload_job(
    job_id: str, request: Any, task: str, tenant_id: str = DEFAULT_TENANT
) -> dict[str, Any]:
    """Dispatches an upload job to the ML Engine microservice for async execution."""
    allowed_columns = [
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
    ]

    filtered_skus = []
    for sku in request.skus:
        filtered_sku = {col: sku.get(col, "") for col in allowed_columns}
        filtered_skus.append(filtered_sku)

    payload = {
        "job_id": str(job_id),
        "task": task,
        "domain": request.domain or "market",
        "skus": filtered_skus,
        "backend_url": BACKEND_INTERNAL_URL,
        "sheet_name": request.outlet_id_or_name,
        "callback_url": None,
        "spreadsheet_id": None,
        "tenant_id": tenant_id,
        "trace_id": TRACE_ID.get(),
    }
    return _enqueue_or_post(job_id, payload, "upload job")


def match_skus(
    skus: list[dict[str, Any]],
    domain: str = "market",
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Runs synchronous SKU matching via gRPC with typed Protobuf messages."""
    try:
        stub = get_stub()
        sku_inputs = [
            pb2.SKUInput(
                name=s.get("name", s.get("Name", "")),
                price=float(s.get("price", s.get("Price", 0.0)) or 0.0),
                description=str(s.get("description", s.get("Description", "")) or ""),
                category=str(s.get("category", s.get("Category", "")) or ""),
            )
            for s in skus
        ]
        req = pb2.MatchRequest(
            skus=sku_inputs,
            domain=domain,
            rules_json="",
            tenant_id=tenant_id,
        )
        resp = stub.Match(req, metadata=_metadata(), timeout=60.0)
        results = [
            {
                "matched_catalog_name": r.matched_catalog_name,
                "score": r.score,
                "status": r.status,
                "logic_notes": r.logic_notes,
                "suggested_bt": r.suggested_bt,
                "suggested_gk": r.suggested_gk,
                "suggested_region": r.suggested_region,
                "rules_applied": r.rules_applied,
            }
            for r in resp.results
        ]
        return {
            "status": resp.status,
            "domain": resp.domain,
            "total": resp.total,
            "results": results,
            "stage_timings": dict(resp.stage_timings),
            "duration_seconds": resp.duration_seconds,
            "rule_version": resp.rule_version,
        }
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")


def classify_skus(
    skus: list[dict[str, Any]],
    domain: str = "market",
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Runs synchronous SKU classification via gRPC with typed Protobuf messages."""
    try:
        stub = get_stub()
        sku_inputs = [
            pb2.SKUInput(
                name=s.get("name", s.get("Name", "")),
                price=float(s.get("price", s.get("Price", 0.0)) or 0.0),
                description=str(s.get("description", s.get("Description", "")) or ""),
                category=str(s.get("category", s.get("Category", "")) or ""),
            )
            for s in skus
        ]
        req = pb2.ClassifyRequest(
            skus=sku_inputs,
            domain=domain,
            rules_json="",
            tenant_id=tenant_id,
        )
        resp = stub.Classify(req, metadata=_metadata(), timeout=60.0)
        results = [
            {
                "suggested_bt": r.suggested_bt,
                "bt_confidence": r.bt_confidence,
                "bt_status": r.bt_status,
                "bt_source": r.bt_source,
                "model": r.model,
                "bt_model": r.bt_model,
                "suggested_gk": r.suggested_gk,
                "gk_confidence": r.gk_confidence,
                "gk_status": r.gk_status,
                "suggested_region": r.suggested_region,
                "region_confidence": r.region_confidence,
                "region_status": r.region_status,
                "region_source": r.region_source,
                "rules_applied": r.rules_applied,
                "logic_notes": r.logic_notes,
            }
            for r in resp.results
        ]
        return {
            "status": resp.status,
            "domain": resp.domain,
            "total": resp.total,
            "results": results,
            "stage_timings": dict(resp.stage_timings),
            "duration_seconds": resp.duration_seconds,
            "rule_version": resp.rule_version,
        }
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")


def run_single(
    sku_name: str,
    domain: str = "market",
    task: str = "pipeline",
    price: float = 0.0,
    description: str = "",
    category: str = "",
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Runs synchronous single-SKU inference on the ML Engine via gRPC."""
    try:
        stub = get_stub()
        req = pb2.SingleRunRequest(
            task=task,
            domain=domain,
            skus=[
                pb2.SKUInput(
                    name=sku_name,
                    price=float(price or 0.0),
                    description=description or "",
                    category=category or "",
                )
            ],
            rules_json="",
            tenant_id=tenant_id,
        )
        resp = stub.RunSingle(req, metadata=_metadata(), timeout=60.0)
        results = [
            {
                "matched_catalog_name": r.matched_catalog_name,
                "score": r.score,
                "status": r.status,
                "logic_notes": r.logic_notes,
                "suggested_bt": r.suggested_bt,
                "bt_confidence": r.bt_confidence,
                "bt_status": r.bt_status,
                "bt_source": r.bt_source,
                "model": r.model,
                "bt_model": r.bt_model,
                "suggested_gk": r.suggested_gk,
                "gk_confidence": r.gk_confidence,
                "gk_status": r.gk_status,
                "suggested_region": r.suggested_region,
                "region_confidence": r.region_confidence,
                "region_status": r.region_status,
                "region_source": r.region_source,
                "rules_applied": r.rules_applied,
            }
            for r in resp.results
        ]
        return {
            "results": results,
            "domain": resp.domain,
            "task": resp.task,
            "total": resp.total,
            "stage_timings": dict(resp.stage_timings),
            "duration_seconds": resp.duration_seconds,
            "rule_version": resp.rule_version,
            "status": resp.status,
        }
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")


def suggest_tags(
    sku_name: str,
    domain: str = "market",
    current_bt: str = "",
    exclude_bt: str = "",
    exclude_gk: str = "",
) -> dict[str, Any]:
    """Requests template-aware tag suggestions from the ML Engine via gRPC."""
    try:
        stub = get_stub()
        req = pb2.SuggestRequest(
            sku_name=sku_name,
            domain=domain,
            current_bt=current_bt or "",
            exclude_bt=exclude_bt or "",
            exclude_gk=exclude_gk or "",
        )
        resp = stub.SuggestTags(req, metadata=_metadata(), timeout=15.0)
        return {
            "status": resp.status,
            "sku_name": resp.sku_name,
            "suggested_bt": resp.suggested_bt,
            "suggested_gk": resp.suggested_gk,
            "suggested_region": resp.suggested_region,
            "template_used": resp.template_used,
            "is_match": resp.is_match,
        }
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")


def cancel_engine_job(job_id: str) -> bool:
    """Notifies the ML Engine to abort processing for the given job via Redis and gRPC."""
    try:
        queue = get_task_queue()
        if queue is not None:
            queue.request_cancel(str(job_id))
    except Exception as e:
        logger.warning(f"Could not set Redis cancellation flag for job {job_id}: {e}")
    try:
        stub = get_stub()
        resp = stub.CancelJob(pb2.CancelJobRequest(job_id=str(job_id)), timeout=5.0)
        return resp.status == "cancelled"
    except Exception as e:
        logger.warning(f"Could not notify ML Engine of job {job_id} cancellation via gRPC: {e}")
        return False


def list_vector_collections() -> dict[str, Any]:
    """Fetches vector database collection names from the ML Engine via gRPC."""
    try:
        stub = get_stub()
        resp = stub.ListVectorCollections(pb2.Empty(), timeout=5.0)
        return json.loads(resp.payload_json)
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")


def get_vector_collection(name: str) -> dict[str, Any]:
    """Fetches info for a specific vector collection from the ML Engine via gRPC."""
    try:
        stub = get_stub()
        resp = stub.GetVectorCollection(pb2.VectorCollectionRequest(name=name), timeout=5.0)
        return json.loads(resp.payload_json)
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")


def search_vector_collection(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatches vector search request to the ML Engine via gRPC."""
    try:
        stub = get_stub()
        req = pb2.VectorSearchRequest(
            name=name,
            query=payload.get("query", ""),
            limit=int(payload.get("top_k", 10)),
        )
        resp = stub.SearchVectorCollection(req, timeout=15.0)
        return json.loads(resp.payload_json)
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")


def run_sku_audit(
    sku_name: str,
    domain: str = "market",
    task: str = "pipeline",
    price: float = 0.0,
    description: str = "",
    category: str = "",
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Runs single-SKU diagnostic audit on the ML Engine via gRPC."""
    try:
        stub = get_stub()
        req = pb2.AuditRequest(
            sku_name=sku_name,
            domain=domain,
            task=task,
            price=float(price or 0.0),
            description=description or "",
            category=category or "",
            tenant_id=tenant_id,
        )
        resp = stub.RunAudit(req, metadata=_metadata(), timeout=60.0)
        return json.loads(resp.audit_json)
    except Exception as re:
        raise RuntimeError(f"Could not reach ML Engine via gRPC at {ENGINE_GRPC_URL}: {re}")
