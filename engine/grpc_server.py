"""
gRPC Server implementation for SKU MatchOps ML Engine.
Serves high-performance Protobuf-based inference endpoints on port 50051.
"""

import json
import logging
import threading
import time
from concurrent import futures

import grpc

from engine import config
from engine.core.observability import TRACE_ID
from engine.core.resource_loader import (
    _model_statuses,
    reset_statuses,
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


class EngineInferenceServicer(pb2_grpc.EngineInferenceServiceServicer):
    def _extract_request_id(self, context) -> str:
        for key, val in context.invocation_metadata():
            if key.lower() == "x-request-id":
                TRACE_ID.set(val)
                return val
        return "-"

    def Match(self, request, context):
        self._extract_request_id(context)
        t0 = time.time()
        skus = [
            {"name": s.name, "price": s.price, "description": s.description, "category": s.category}
            for s in request.skus
        ]
        tenant = request.tenant_id or "default"
        try:
            with pinned_rules(snapshot_rules(tenant)):
                res = process_request(skus=skus, task="matcher", domain=request.domain)
            items = []
            for item in res.get("results", []):
                items.append(
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
                )
            stage_timings = {
                k: float(v)
                for k, v in res.get("stage_timings", {}).items()
                if isinstance(v, (int, float))
            }
            return pb2.MatchResponse(
                status="success",
                domain=str(res.get("domain", request.domain)),
                total=int(res.get("total", len(items))),
                results=items,
                stage_timings=stage_timings,
                duration_seconds=time.time() - t0,
                rule_version=str(res.get("rule_version", "")),
            )
        except Exception as e:
            logger.error(f"[gRPC] Match failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def Classify(self, request, context):
        self._extract_request_id(context)
        t0 = time.time()
        skus = [
            {"name": s.name, "price": s.price, "description": s.description, "category": s.category}
            for s in request.skus
        ]
        tenant = request.tenant_id or "default"
        try:
            with pinned_rules(snapshot_rules(tenant)):
                res = process_request(skus=skus, task="classifier", domain=request.domain)
            items = []
            for item in res.get("results", []):
                items.append(
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
                )
            stage_timings = {
                k: float(v)
                for k, v in res.get("stage_timings", {}).items()
                if isinstance(v, (int, float))
            }
            return pb2.ClassifyResponse(
                status="success",
                domain=str(res.get("domain", request.domain)),
                total=int(res.get("total", len(items))),
                results=items,
                stage_timings=stage_timings,
                duration_seconds=time.time() - t0,
                rule_version=str(res.get("rule_version", "")),
            )
        except Exception as e:
            logger.error(f"[gRPC] Classify failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def RunSingle(self, request, context):
        self._extract_request_id(context)
        t0 = time.time()
        skus = [
            {"name": s.name, "price": s.price, "description": s.description, "category": s.category}
            for s in request.skus
        ]
        tenant = request.tenant_id or "default"
        try:
            with pinned_rules(snapshot_rules(tenant)):
                res = process_request(skus=skus, task=request.task, domain=request.domain)
            items = []
            for item in res.get("results", []):
                items.append(
                    pb2.PipelineResultItem(
                        matched_catalog_name=str(item.get("matched_catalog_name") or ""),
                        score=float(item.get("score") or 0.0),
                        status=str(item.get("status") or ""),
                        logic_notes=str(item.get("logic_notes") or ""),
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
                    )
                )
            stage_timings = {
                k: float(v)
                for k, v in res.get("stage_timings", {}).items()
                if isinstance(v, (int, float))
            }
            return pb2.SingleRunResponse(
                status="success",
                task=request.task,
                domain=str(res.get("domain", request.domain)),
                total=int(res.get("total", len(items))),
                results=items,
                stage_timings=stage_timings,
                duration_seconds=time.time() - t0,
                rule_version=str(res.get("rule_version", "")),
            )
        except Exception as e:
            logger.error(f"[gRPC] RunSingle failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def SuggestTags(self, request, context):
        self._extract_request_id(context)
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
            logger.error(f"[gRPC] SuggestTags failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def RunAudit(self, request, context):
        self._extract_request_id(context)
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
            logger.error(f"[gRPC] RunAudit failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def DispatchBatchJob(self, request, context):
        self._extract_request_id(context)
        payload = {
            "job_id": request.job_id,
            "task": request.task,
            "domain": request.domain,
            "skus": [
                {
                    "name": s.name,
                    "price": s.price,
                    "description": s.description,
                    "category": s.category,
                }
                for s in request.skus
            ],
            "callback_url": request.callback_url or None,
            "sheet_name": request.sheet_name or None,
            "created_by": request.created_by,
            "tenant_id": request.tenant_id or "default",
            "trace_id": TRACE_ID.get(),
        }
        try:
            enqueue_batch_job(payload)
            return pb2.JobDispatchResponse(status="accepted", job_id=request.job_id)
        except Exception as e:
            logger.error(f"[gRPC] DispatchBatchJob failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def DispatchUploadJob(self, request, context):
        self._extract_request_id(context)
        payload = {
            "job_id": request.job_id,
            "task": request.task,
            "domain": request.domain,
            "file_path": request.file_path,
            "callback_url": request.callback_url or None,
            "sheet_name": request.sheet_name or None,
            "created_by": request.created_by,
            "tenant_id": request.tenant_id or "default",
            "trace_id": TRACE_ID.get(),
        }
        try:
            enqueue_batch_job(payload)
            return pb2.JobDispatchResponse(status="accepted", job_id=request.job_id)
        except Exception as e:
            logger.error(f"[gRPC] DispatchUploadJob failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def CancelJob(self, request, context):
        self._extract_request_id(context)
        cancel_job(request.job_id)
        return pb2.CancelJobResponse(status="cancelled", message=f"Job {request.job_id} cancelled.")

    def GetHealth(self, request, context):
        market_ready = (
            _model_statuses.get("market", {}).get("pipeline") == "ready"
            and _model_statuses.get("market", {}).get("classifier") == "ready"
        )
        food_ready = (
            _model_statuses.get("food", {}).get("pipeline") == "ready"
            and _model_statuses.get("food", {}).get("classifier") == "ready"
        )
        loaded = [d for d in _model_statuses if _model_statuses[d].get("pipeline") == "ready"]
        return pb2.HealthResponse(
            status="ok",
            service="matchops-engine",
            loaded_domains=loaded,
            all_models_ready=market_ready and food_ready,
        )

    def GetModelsStatus(self, request, context):
        domain_statuses = {}
        for d, s in _model_statuses.items():
            domain_statuses[d] = pb2.DomainModelStatus(
                pipeline=s.get("pipeline", "idle"),
                classifier=s.get("classifier", "idle"),
            )
        import engine.server as s

        is_loading = getattr(s, "_loading_in_progress", False) or any(
            st.pipeline == "loading" or st.classifier == "training"
            for st in domain_statuses.values()
        )
        market_ready = (
            domain_statuses.get("market")
            and domain_statuses["market"].pipeline == "ready"
            and domain_statuses["market"].classifier == "ready"
        )
        food_ready = (
            domain_statuses.get("food")
            and domain_statuses["food"].pipeline == "ready"
            and domain_statuses["food"].classifier == "ready"
        )
        is_ready = bool(market_ready and food_ready and not is_loading)
        status_str = "ready" if is_ready else ("loading" if is_loading else "unloaded")
        return pb2.ModelsStatusResponse(domains=domain_statuses, status=status_str)

    def ManageModels(self, request, context):
        import engine.server as s

        market_ready = (
            _model_statuses.get("market", {}).get("pipeline") == "ready"
            and _model_statuses.get("market", {}).get("classifier") == "ready"
        )
        food_ready = (
            _model_statuses.get("food", {}).get("pipeline") == "ready"
            and _model_statuses.get("food", {}).get("classifier") == "ready"
        )
        if market_ready and food_ready and request.action != "reload":
            return pb2.ManageModelsResponse(
                status="success", message="Models are already fully loaded."
            )

        if getattr(s, "_loading_in_progress", False) and request.action != "reload":
            return pb2.ManageModelsResponse(
                status="ignored", message="Model loading is already in progress."
            )

        s._loading_in_progress = True
        if request.action == "reload":
            reset_statuses()
        threading.Thread(target=s._bg_load_models, daemon=True).start()
        return pb2.ManageModelsResponse(status="started", message="Model loading triggered.")

    def RefreshRules(self, request, context):
        try:
            if request.rules_json:
                data = json.loads(request.rules_json)
                set_rules_from_dicts(data, version=request.version or None)
            else:
                refresh_rules_cache()
            return pb2.RefreshRulesResponse(status="success", message="Rules refreshed.")
        except Exception as e:
            logger.error(f"[gRPC] RefreshRules failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def ListVectorCollections(self, request, context):
        try:
            client = VectorStore.get_client()
            cols = [c.name for c in client.get_collections().collections]
            return pb2.JsonPayloadResponse(payload_json=json.dumps({"collections": cols}))
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetVectorCollection(self, request, context):
        try:
            client = VectorStore.get_client()
            info = client.get_collection(collection_name=request.name)
            return pb2.JsonPayloadResponse(payload_json=json.dumps(info.dict()))
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def SearchVectorCollection(self, request, context):
        try:
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
            ).points
            res = [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in points]
            return pb2.JsonPayloadResponse(payload_json=json.dumps({"results": res}))
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))


def start_grpc_server(port: int | None = None) -> grpc.Server:
    """Starts the gRPC inference server on the configured port."""
    target_port = port if port is not None else config.ENGINE_GRPC_PORT
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=getattr(config, "MAX_CONCURRENT_TASKS", config.MAX_CPU_CORES) * 2
        ),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    pb2_grpc.add_EngineInferenceServiceServicer_to_server(EngineInferenceServicer(), server)
    bind_addr = f"0.0.0.0:{target_port}"
    server.add_insecure_port(bind_addr)
    server.start()
    logger.info(f"[gRPC] SKU MatchOps Engine gRPC Server listening on {bind_addr}")
    return server
