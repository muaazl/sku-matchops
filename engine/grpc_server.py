"""
gRPC Server implementation for SKU MatchOps ML Engine.
Serves high-performance Protobuf-based inference endpoints on port 50051.
"""

import json
import logging
import time
from concurrent import futures

import grpc
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

from engine import config
from engine.core.grpc_utils import (
    GRPC_CHANNEL_OPTIONS,
    GrpcContextInterceptor,
    abort_rpc_error,
    result_to_pipeline_proto,
    sku_proto_to_dict,
    stage_timings_map,
)
from engine.core.observability import TRACE_ID
from engine.core.resource_loader import (
    _model_statuses,
    are_all_models_ready,
    is_model_loading,
    trigger_model_load,
)
from engine.data_pipeline.vector_store import VectorStore
from engine.pipeline.audit_engine import run_sku_audit
from engine.pipeline.processor import process_request
from engine.pipeline.worker_runner import cancel_job, enqueue_batch_job
from engine.protos import engine_service_pb2 as pb2
from engine.protos import engine_service_pb2_grpc as pb2_grpc
from engine.rules_engine import refresh_rules_cache
from engine.rules_engine.loader import (
    pinned_rules,
    set_rules_from_dicts,
    snapshot_rules,
)
from engine.templates.template_suggest import suggest_tags_from_template

logger = logging.getLogger("matchops.engine.grpc")


def _build_qdrant_filter(filters_json: str) -> Filter | None:
    if not filters_json:
        return None
    raw = json.loads(filters_json)
    if not isinstance(raw, dict):
        raise ValueError("filters must be a JSON object.")
    if not raw:
        return None
    if any(k in raw for k in ("must", "should", "must_not")):
        return Filter.model_validate(raw)
    must = [
        FieldCondition(
            key=str(k),
            match=MatchAny(any=v) if isinstance(v, list) else MatchValue(value=v),
        )
        for k, v in raw.items()
        if v is not None
    ]
    return Filter(must=must) if must else None


class EngineInferenceServicer(pb2_grpc.EngineInferenceServiceServicer):
    def Match(self, request, context):
        t0 = time.time()
        skus = [sku_proto_to_dict(s) for s in request.skus]
        tenant = request.tenant_id or "default"
        try:
            with pinned_rules(snapshot_rules(tenant)):
                res = process_request(skus=skus, task="matcher", domain=request.domain)
            items = [
                pb2.MatchResultItem(
                    matched_catalog_name=str(item.get("matched_catalog_name") or ""),
                    score=float(item.get("score") or 0.0),
                    status=str(item.get("status") or ""),
                    logic_notes=str(item.get("logic_notes") or ""),
                    suggested_bt=str(item.get("suggested_bt") or ""),
                    suggested_gk=str(item.get("suggested_gk") or ""),
                    suggested_region=str(item.get("suggested_region") or ""),
                    rules_applied=str(item.get("rules_applied") or ""),
                )
                for item in res.get("results", [])
            ]
            return pb2.MatchResponse(
                status="success",
                domain=str(res.get("domain", request.domain)),
                total=int(res.get("total", len(items))),
                results=items,
                stage_timings=stage_timings_map(res),
                duration_seconds=time.time() - t0,
                rule_version=str(res.get("rule_version", "")),
            )
        except Exception as e:
            abort_rpc_error(context, logger, "Match", e)

    def Classify(self, request, context):
        t0 = time.time()
        skus = [sku_proto_to_dict(s) for s in request.skus]
        tenant = request.tenant_id or "default"
        try:
            with pinned_rules(snapshot_rules(tenant)):
                res = process_request(skus=skus, task="classifier", domain=request.domain)
            items = [
                pb2.ClassifyResultItem(
                    suggested_bt=str(item.get("suggested_bt") or ""),
                    bt_confidence=float(item.get("bt_confidence") or 0.0),
                    bt_status=str(item.get("bt_status") or ""),
                    bt_source=str(item.get("bt_source") or ""),
                    model=str(item.get("model") or "logreg"),
                    bt_model=str(item.get("bt_model") or "logreg"),
                    suggested_gk=str(item.get("suggested_gk") or ""),
                    gk_confidence=float(item.get("gk_confidence") or 0.0),
                    gk_status=str(item.get("gk_status") or ""),
                    suggested_region=str(item.get("suggested_region") or ""),
                    region_confidence=float(item.get("region_confidence") or 0.0),
                    region_status=str(item.get("region_status") or ""),
                    region_source=str(item.get("region_source") or ""),
                    rules_applied=str(item.get("rules_applied") or ""),
                    logic_notes=str(item.get("logic_notes") or ""),
                )
                for item in res.get("results", [])
            ]
            return pb2.ClassifyResponse(
                status="success",
                domain=str(res.get("domain", request.domain)),
                total=int(res.get("total", len(items))),
                results=items,
                stage_timings=stage_timings_map(res),
                duration_seconds=time.time() - t0,
                rule_version=str(res.get("rule_version", "")),
            )
        except Exception as e:
            abort_rpc_error(context, logger, "Classify", e)

    def RunSingle(self, request, context):
        t0 = time.time()
        skus = [sku_proto_to_dict(s) for s in request.skus]
        tenant = request.tenant_id or "default"
        try:
            with pinned_rules(snapshot_rules(tenant)):
                res = process_request(skus=skus, task=request.task, domain=request.domain)
            items = [result_to_pipeline_proto(item) for item in res.get("results", [])]
            return pb2.SingleRunResponse(
                status="success",
                task=request.task,
                domain=str(res.get("domain", request.domain)),
                total=int(res.get("total", len(items))),
                results=items,
                stage_timings=stage_timings_map(res),
                duration_seconds=time.time() - t0,
                rule_version=str(res.get("rule_version", "")),
            )
        except Exception as e:
            abort_rpc_error(context, logger, "RunSingle", e)

    def SuggestTags(self, request, context):
        try:
            res = suggest_tags_from_template(
                sku_name=request.sku_name,
                domain=request.domain,
                current_bt=request.current_bt,
                exclude_bt=request.exclude_bt,
                exclude_gk=request.exclude_gk,
            )
            return pb2.SuggestResponse(
                status="success",
                sku_name=str(res.get("sku_name") or request.sku_name),
                suggested_bt=str(res.get("suggested_bt") or ""),
                suggested_gk=str(res.get("suggested_gk") or ""),
                suggested_region=str(res.get("suggested_region") or ""),
                template_used=str(res.get("template_used") or ""),
                is_match=bool(res.get("is_match", False)),
            )
        except Exception as e:
            abort_rpc_error(context, logger, "SuggestTags", e)

    def RunAudit(self, request, context):
        tenant = request.tenant_id or "default"
        try:
            with pinned_rules(snapshot_rules(tenant)):
                res = run_sku_audit(
                    sku_name=request.sku_name,
                    domain=request.domain,
                    task=request.task,
                    price=request.price,
                    description=request.description,
                    category=request.category,
                )
            return pb2.AuditResponse(audit_json=json.dumps(res))
        except Exception as e:
            abort_rpc_error(context, logger, "RunAudit", e)

    def DispatchBatchJob(self, request, context):
        if not request.skus:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty SKU list provided.")
        payload = {
            "job_id": request.job_id,
            "task": request.task or "pipeline",
            "domain": request.domain or "market",
            "skus": [sku_proto_to_dict(s) for s in request.skus],
            "callback_url": request.callback_url or None,
            "sheet_name": request.sheet_name or None,
            "spreadsheet_id": request.spreadsheet_id or None,
            "backend_grpc_url": config.BACKEND_GRPC_URL,
            "created_by": request.created_by,
            "tenant_id": request.tenant_id or "default",
            "trace_id": TRACE_ID.get(),
        }
        try:
            enqueue_batch_job(payload)
            return pb2.JobDispatchResponse(status="accepted", job_id=request.job_id)
        except Exception as e:
            abort_rpc_error(context, logger, "DispatchBatchJob", e)

    def DispatchUploadJob(self, request, context):
        if request.skus:
            skus = [dict(item.fields) for item in request.skus]
        else:
            try:
                skus = json.loads(request.skus_json) if request.skus_json else []
            except Exception as e:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid skus_json: {e}")
        payload = {
            "job_id": request.job_id,
            "task": request.task or "upload",
            "domain": request.domain or "market",
            "skus": skus,
            "file_path": request.file_path,
            "callback_url": request.callback_url or None,
            "sheet_name": request.sheet_name or None,
            "spreadsheet_id": None,
            "backend_grpc_url": config.BACKEND_GRPC_URL,
            "created_by": request.created_by,
            "tenant_id": request.tenant_id or "default",
            "trace_id": TRACE_ID.get(),
        }
        try:
            enqueue_batch_job(payload)
            return pb2.JobDispatchResponse(status="accepted", job_id=request.job_id)
        except Exception as e:
            abort_rpc_error(context, logger, "DispatchUploadJob", e)

    def CancelJob(self, request, context):
        cancel_job(request.job_id)
        return pb2.CancelJobResponse(status="cancelled", message=f"Job {request.job_id} cancelled.")

    def GetHealth(self, request, context):
        loaded = [d for d in _model_statuses if _model_statuses[d].get("pipeline") == "ready"]
        return pb2.HealthResponse(
            status="ok",
            service="matchops-engine",
            loaded_domains=loaded,
            all_models_ready=are_all_models_ready(),
        )

    def GetModelsStatus(self, request, context):
        domain_statuses = {
            d: pb2.DomainModelStatus(
                pipeline=st.get("pipeline", "idle"),
                classifier=st.get("classifier", "idle"),
            )
            for d, st in _model_statuses.items()
        }
        loading = is_model_loading()
        ready = are_all_models_ready() and not loading
        status_str = "ready" if ready else ("loading" if loading else "unloaded")
        return pb2.ModelsStatusResponse(domains=domain_statuses, status=status_str)

    def ManageModels(self, request, context):
        status, message = trigger_model_load(reload=request.action == "reload")
        return pb2.ManageModelsResponse(status=status, message=message)

    def RefreshRules(self, request, context):
        try:
            if request.rules_json:
                data = json.loads(request.rules_json)
                set_rules_from_dicts(data, version=request.version or None)
            else:
                refresh_rules_cache()
            return pb2.RefreshRulesResponse(status="success", message="Rules refreshed.")
        except Exception as e:
            abort_rpc_error(context, logger, "RefreshRules", e)

    def ListVectorCollections(self, request, context):
        try:
            client = VectorStore.get_client()
            cols = [c.name for c in client.get_collections().collections]
            return pb2.JsonPayloadResponse(payload_json=json.dumps({"collections": cols}))
        except Exception as e:
            abort_rpc_error(context, logger, "ListVectorCollections", e)

    def GetVectorCollection(self, request, context):
        try:
            client = VectorStore.get_client()
            info = client.get_collection(collection_name=request.name)
            payload = info.model_dump(mode="json") if hasattr(info, "model_dump") else info.dict()
            return pb2.JsonPayloadResponse(payload_json=json.dumps(payload))
        except Exception as e:
            abort_rpc_error(context, logger, "GetVectorCollection", e)

    def SearchVectorCollection(self, request, context):
        try:
            query_filter = _build_qdrant_filter(request.filters_json)
            score_threshold = request.score_threshold if request.has_score_threshold else None
            client = VectorStore.get_client()
            from engine.core.resource_loader import _get_shared_models

            embed_engine, _ = _get_shared_models()
            encoded = embed_engine.encode([request.query])
            query_vector = encoded["dense"][0]

            points = client.query_points(
                collection_name=request.name,
                query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
                using="dense",
                limit=request.limit or 10,
                query_filter=query_filter,
                score_threshold=score_threshold,
            ).points
            res = [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in points]
            return pb2.JsonPayloadResponse(payload_json=json.dumps({"results": res}))
        except Exception as e:
            abort_rpc_error(context, logger, "SearchVectorCollection", e)


def start_grpc_server(port: int | None = None, host: str = "0.0.0.0") -> grpc.Server:
    """Starts the gRPC inference server on the configured port."""
    target_port = port if port is not None else config.ENGINE_GRPC_PORT
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=getattr(config, "MAX_CONCURRENT_TASKS", config.MAX_CPU_CORES) * 2
        ),
        interceptors=[GrpcContextInterceptor()],
        options=list(GRPC_CHANNEL_OPTIONS),
    )
    pb2_grpc.add_EngineInferenceServiceServicer_to_server(EngineInferenceServicer(), server)
    bind_addr = f"{host}:{target_port}"
    bound_port = server.add_insecure_port(bind_addr)
    server._bound_port = bound_port  # type: ignore[attr-defined]
    server.start()
    logger.info(f"[gRPC] SKU MatchOps Engine gRPC Server listening on {host}:{bound_port}")
    return server
