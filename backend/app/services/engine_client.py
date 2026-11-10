"""
SKU MatchOps Backend - ML Engine gRPC Client
Dispatches inference and batch processing workloads to the dedicated ML Engine microservice via gRPC.
"""

import json
import logging
import threading
import time
from typing import Any

import grpc

from backend.app.core import config as backend_config
from backend.app.core.config import DEFAULT_TENANT
from backend.app.core.grpc_utils import (
    GRPC_CHANNEL_OPTIONS,
    dict_to_sku_proto,
    grpc_metadata,
    pipeline_proto_to_result,
    unwrap_rpc_error,
)
from backend.app.core.observability import get_trace_id
from backend.app.protos import engine_service_pb2 as pb2
from backend.app.protos import engine_service_pb2_grpc as pb2_grpc
from backend.app.schemas.models import BaseRequest
from backend.app.services.task_queue import get_task_queue

logger = logging.getLogger("matchops.engine_client")

ENGINE_GRPC_URL = backend_config.ENGINE_GRPC_URL
BACKEND_GRPC_URL = backend_config.BACKEND_GRPC_URL

_client_lock = threading.Lock()
_channel: grpc.Channel | None = None
_channel_target: str | None = None
_stub: pb2_grpc.EngineInferenceServiceStub | None = None


def get_stub() -> pb2_grpc.EngineInferenceServiceStub:
    """Returns thread-safe singleton gRPC stub connected to the ML Engine."""
    global _channel, _channel_target, _stub
    with _client_lock:
        if _stub is None or _channel_target != ENGINE_GRPC_URL:
            if _channel is not None:
                try:
                    _channel.close()
                except Exception:
                    pass
            _channel = grpc.insecure_channel(ENGINE_GRPC_URL, options=list(GRPC_CHANNEL_OPTIONS))
            _channel_target = ENGINE_GRPC_URL
            _stub = pb2_grpc.EngineInferenceServiceStub(_channel)
        return _stub


_models_status_cache: dict[str, Any] = {}
_models_status_cache_time: float = 0.0


def _invalidate_models_cache() -> None:
    global _models_status_cache_time
    with _client_lock:
        _models_status_cache_time = 0.0


def is_engine_reachable() -> bool:
    """Checks if the ML Engine microservice is responding to health checks."""
    try:
        stub = get_stub()
        resp = stub.GetHealth(pb2.Empty(), metadata=grpc_metadata(), timeout=2.0)
        return resp.status == "ok"
    except Exception:
        return False


def get_engine_health() -> dict[str, Any]:
    """Fetches health and loaded model status from the ML Engine."""
    try:
        stub = get_stub()
        resp = stub.GetHealth(pb2.Empty(), metadata=grpc_metadata(), timeout=5.0)
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
    with _client_lock:
        ttl = 1.0 if _models_status_cache.get("loading_in_progress") else 3.0
        if _models_status_cache and (now - _models_status_cache_time) < ttl:
            return _models_status_cache

    try:
        stub = get_stub()
        resp = stub.GetModelsStatus(pb2.Empty(), metadata=grpc_metadata(), timeout=5.0)
        details = {
            domain: {"pipeline": s.pipeline, "classifier": s.classifier}
            for domain, s in resp.domains.items()
        }
        loading_in_progress = resp.status == "loading" or any(
            s["pipeline"] == "loading" or s["classifier"] == "training" for s in details.values()
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
        with _client_lock:
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
    _invalidate_models_cache()
    try:
        stub = get_stub()
        resp = stub.ManageModels(
            pb2.ManageModelsRequest(action="load"), metadata=grpc_metadata(), timeout=5.0
        )
        return {"status": resp.status, "message": resp.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def reload_engine_models() -> dict[str, Any]:
    """Forces cache clearing and model reload on the ML Engine."""
    _invalidate_models_cache()
    try:
        stub = get_stub()
        resp = stub.ManageModels(
            pb2.ManageModelsRequest(action="reload"), metadata=grpc_metadata(), timeout=5.0
        )
        return {"status": resp.status, "message": resp.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def refresh_engine_rules(rules_data: list[dict] | None = None, version: str = "") -> dict[str, Any]:
    """Tells the ML Engine to reload rules via gRPC (fallback for when Redis pub/sub is down)."""
    try:
        stub = get_stub()
        req = pb2.RefreshRulesRequest(
            rules_json=json.dumps(rules_data) if rules_data is not None else "", version=version
        )
        resp = stub.RefreshRules(req, metadata=grpc_metadata(), timeout=5.0)
        return {"status": resp.status, "message": resp.message}
    except Exception as e:
        logger.warning(f"Failed to notify ML Engine via gRPC to refresh rules: {e}")
        return {"status": "error", "error": str(e)}


def _try_enqueue_redis(job_id: str, payload: dict[str, Any], label: str) -> dict[str, Any] | None:
    """Attempts to enqueue the job onto the Redis distributed queue. Returns None on fallback."""
    try:
        queue = get_task_queue()
        if queue is not None:
            task_id = queue.enqueue(payload)
            return {"status": "queued", "job_id": str(job_id), "task_id": task_id}
    except Exception as e:
        logger.warning(
            f"Redis queue unavailable ({e}); falling back to gRPC dispatch for {label} {job_id}."
        )
    return None


def dispatch_batch_job(
    job_id: str, request: BaseRequest, task: str, tenant_id: str = DEFAULT_TENANT
) -> dict[str, Any]:
    """Dispatches a batch job via Redis queue, falling back to direct gRPC DispatchBatchJob."""
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
        "backend_grpc_url": BACKEND_GRPC_URL,
        "callback_url": request.callback_url,
        "sheet_name": request.sheet_name,
        "spreadsheet_id": request.spreadsheet_id,
        "tenant_id": tenant_id,
        "trace_id": get_trace_id(),
    }
    queued = _try_enqueue_redis(job_id, payload, "job")
    if queued is not None:
        return queued

    try:
        stub = get_stub()
        req = pb2.BatchJobRequest(
            job_id=str(job_id),
            task=task,
            domain=request.domain or "market",
            skus=[dict_to_sku_proto(s) for s in skus_payload],
            callback_url=request.callback_url or "",
            sheet_name=request.sheet_name or "",
            spreadsheet_id=request.spreadsheet_id or "",
            backend_grpc_url=BACKEND_GRPC_URL,
            tenant_id=tenant_id,
        )
        resp = stub.DispatchBatchJob(req, metadata=grpc_metadata(), timeout=30.0)
        return {"status": resp.status, "job_id": resp.job_id}
    except Exception as e:
        error_msg = (
            f"Failed to dispatch job {job_id} to ML Engine via gRPC ({ENGINE_GRPC_URL}): {e}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def dispatch_upload_job(
    job_id: str, request: Any, task: str, tenant_id: str = DEFAULT_TENANT
) -> dict[str, Any]:
    """Dispatches an upload job via Redis queue, falling back to direct gRPC DispatchUploadJob."""
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

    filtered_skus = [{col: sku.get(col, "") for col in allowed_columns} for sku in request.skus]

    payload = {
        "job_id": str(job_id),
        "task": task,
        "domain": request.domain or "market",
        "skus": filtered_skus,
        "backend_grpc_url": BACKEND_GRPC_URL,
        "sheet_name": request.outlet_id_or_name,
        "callback_url": None,
        "spreadsheet_id": None,
        "tenant_id": tenant_id,
        "trace_id": get_trace_id(),
    }
    queued = _try_enqueue_redis(job_id, payload, "upload job")
    if queued is not None:
        return queued

    try:
        stub = get_stub()
        req = pb2.UploadJobRequest(
            job_id=str(job_id),
            task=task,
            domain=request.domain or "market",
            sheet_name=request.outlet_id_or_name or "",
            tenant_id=tenant_id,
            backend_grpc_url=BACKEND_GRPC_URL,
            skus=[
                pb2.UploadSkuItem(fields={k: "" if v is None else str(v) for k, v in s.items()})
                for s in filtered_skus
            ],
        )
        resp = stub.DispatchUploadJob(req, metadata=grpc_metadata(), timeout=30.0)
        return {"status": resp.status, "job_id": resp.job_id}
    except Exception as e:
        error_msg = (
            f"Failed to dispatch upload job {job_id} to ML Engine via gRPC ({ENGINE_GRPC_URL}): {e}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def match_skus(
    skus: list[dict[str, Any]],
    domain: str = "market",
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Runs synchronous SKU matching via gRPC with typed Protobuf messages."""
    try:
        stub = get_stub()
        req = pb2.MatchRequest(
            skus=[dict_to_sku_proto(s) for s in skus],
            domain=domain,
            rules_json="",
            tenant_id=tenant_id,
        )
        resp = stub.Match(req, metadata=grpc_metadata(), timeout=60.0)
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
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


def classify_skus(
    skus: list[dict[str, Any]],
    domain: str = "market",
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Runs synchronous SKU classification via gRPC with typed Protobuf messages."""
    try:
        stub = get_stub()
        req = pb2.ClassifyRequest(
            skus=[dict_to_sku_proto(s) for s in skus],
            domain=domain,
            rules_json="",
            tenant_id=tenant_id,
        )
        resp = stub.Classify(req, metadata=grpc_metadata(), timeout=60.0)
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
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


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
        resp = stub.RunSingle(req, metadata=grpc_metadata(), timeout=60.0)
        return {
            "results": [pipeline_proto_to_result(r) for r in resp.results],
            "domain": resp.domain,
            "task": resp.task,
            "total": resp.total,
            "stage_timings": dict(resp.stage_timings),
            "duration_seconds": resp.duration_seconds,
            "rule_version": resp.rule_version,
            "status": resp.status,
        }
    except Exception as re:
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


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
        resp = stub.SuggestTags(req, metadata=grpc_metadata(), timeout=15.0)
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
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


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
        resp = stub.CancelJob(
            pb2.CancelJobRequest(job_id=str(job_id)), metadata=grpc_metadata(), timeout=5.0
        )
        return resp.status == "cancelled"
    except Exception as e:
        logger.warning(f"Could not notify ML Engine of job {job_id} cancellation via gRPC: {e}")
        return False


def list_vector_collections() -> dict[str, Any]:
    """Fetches vector database collection names from the ML Engine via gRPC."""
    try:
        stub = get_stub()
        resp = stub.ListVectorCollections(pb2.Empty(), metadata=grpc_metadata(), timeout=5.0)
        return json.loads(resp.payload_json)
    except Exception as re:
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


def get_vector_collection(name: str) -> dict[str, Any]:
    """Fetches info for a specific vector collection from the ML Engine via gRPC."""
    try:
        stub = get_stub()
        resp = stub.GetVectorCollection(
            pb2.VectorCollectionRequest(name=name), metadata=grpc_metadata(), timeout=5.0
        )
        return json.loads(resp.payload_json)
    except Exception as re:
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


def search_vector_collection(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatches vector search request to the ML Engine via gRPC."""
    try:
        stub = get_stub()
        st = payload.get("score_threshold")
        filters = payload.get("filters")
        mode = payload.get("mode", "dense")
        req = pb2.VectorSearchRequest(
            name=name,
            query=payload.get("query", ""),
            limit=int(payload.get("top_k", 10)),
            score_threshold=float(st) if st is not None else 0.0,
            has_score_threshold=st is not None,
            filters_json=json.dumps(filters) if filters else "",
            mode=mode,
        )
        resp = stub.SearchVectorCollection(req, metadata=grpc_metadata(), timeout=15.0)
        return json.loads(resp.payload_json)
    except Exception as re:
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


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
        resp = stub.RunAudit(req, metadata=grpc_metadata(), timeout=60.0)
        return json.loads(resp.audit_json)
    except Exception as re:
        unwrap_rpc_error(re, ENGINE_GRPC_URL)


def rerun_rules_via_engine(
    sku_name: str,
    domain: str,
    bt: str,
    gk: str,
    region: str = "",
    category: str = "",
    price: float = 0.0,
    confidence: float = 1.0,
    match_source: str = "Classifier",
    matched_sku: str = "",
    reasoning: str = "",
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Evaluates business rules for a single SKU prediction on the ML Engine via gRPC."""
    stub = get_stub()
    req = pb2.RerunRulesRequest(
        sku_name=sku_name,
        domain=domain,
        bt=bt,
        gk=gk,
        region=region or "",
        category=category or "",
        price=float(price or 0.0),
        confidence=float(confidence or 1.0),
        match_source=match_source or "Classifier",
        matched_sku=matched_sku or "",
        reasoning=reasoning or "",
        tenant_id=tenant_id or DEFAULT_TENANT,
    )
    try:
        resp = stub.RerunRules(req, metadata=grpc_metadata(), timeout=10.0)
        return {
            "suggested_bt": resp.suggested_bt,
            "suggested_gk": resp.suggested_gk,
            "suggested_region": resp.suggested_region,
            "rules_applied": resp.rules_applied,
        }
    except Exception as e:
        try:
            from backend.app.core.rules_evaluator import rerun_rules_locally

            return rerun_rules_locally(
                sku_name=sku_name,
                domain=domain,
                bt=bt,
                gk=gk,
                region=region,
                category=category,
                price=price,
                confidence=confidence,
                match_source=match_source,
                matched_sku=matched_sku,
                reasoning=reasoning,
                tenant_id=tenant_id,
            )
        except Exception:
            unwrap_rpc_error(e, ENGINE_GRPC_URL)
            raise


def test_rule_via_engine(
    rule_dict: dict[str, Any],
    sample_record: dict[str, Any],
    tenant_id: str = DEFAULT_TENANT,
) -> dict[str, Any]:
    """Tests a rule against a sample record on the ML Engine via gRPC."""
    stub = get_stub()
    req = pb2.TestRuleRequest(
        rule_json=json.dumps(rule_dict, default=str),
        sample_record_json=json.dumps(sample_record, default=str),
        tenant_id=tenant_id or DEFAULT_TENANT,
    )
    try:
        resp = stub.TestRule(req, metadata=grpc_metadata(), timeout=10.0)
        return {
            "rule_id": resp.rule_id,
            "fires": resp.fires,
            "matched_conditions": json.loads(resp.matched_conditions_json or "[]"),
            "actions_applied": json.loads(resp.actions_applied_json or "[]"),
            "changes": resp.changes,
            "sample_record_after": json.loads(resp.sample_record_after_json or "{}"),
        }
    except Exception as e:
        try:
            from backend.app.core.rules_evaluator import evaluate_rule_locally

            return evaluate_rule_locally(rule_dict, sample_record)
        except Exception:
            unwrap_rpc_error(e, ENGINE_GRPC_URL)
            raise
