"""
gRPC Server implementation for SKU MatchOps ML Engine.
Serves high-performance Protobuf-based inference endpoints on port 50051.
"""

import json
import logging
import time
from concurrent import futures
from typing import Any

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
from engine.rules_engine import refresh_rules_cache, run_rules_engine
from engine.rules_engine.actions import apply_actions
from engine.rules_engine.evaluator import evaluate_conditions
from engine.rules_engine.loader import (
    Rule,
    pinned_rules,
    set_rules_from_dicts,
    snapshot_rules,
)
from engine.templates.template_suggest import suggest_tags_from_template

logger = logging.getLogger("matchops.engine.grpc")


def _get_vector_store() -> VectorStore:
    vs = VectorStore.__new__(VectorStore)
    vs.client = VectorStore.get_client()
    return vs


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
        try:
            with pinned_rules(snapshot_rules()):
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
        try:
            with pinned_rules(snapshot_rules()):
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
        try:
            with pinned_rules(snapshot_rules()):
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
        try:
            with pinned_rules(snapshot_rules()):
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
            vs = _get_vector_store()
            cols = [
                c.name
                for c in vs._call_with_retry(
                    "get_collections", vs.client.get_collections
                ).collections
            ]
            details: dict[str, Any] = {}
            if cols and hasattr(vs.client, "get_collection"):

                def _fetch_info(col_name: str) -> tuple[str, dict[str, Any] | None]:
                    try:
                        info = vs._call_with_retry(
                            "get_collection", vs.client.get_collection, collection_name=col_name
                        )
                        payload = (
                            info.model_dump(mode="json")
                            if hasattr(info, "model_dump")
                            else (info.dict() if hasattr(info, "dict") else None)
                        )
                        return col_name, payload
                    except Exception:
                        return col_name, None

                with futures.ThreadPoolExecutor(max_workers=min(len(cols), 6)) as pool:
                    for col_name, payload in pool.map(_fetch_info, cols):
                        if isinstance(payload, dict):
                            details[col_name] = payload

            return pb2.JsonPayloadResponse(
                payload_json=json.dumps({"collections": cols, "details": details})
            )
        except Exception as e:
            abort_rpc_error(context, logger, "ListVectorCollections", e)

    def GetVectorCollection(self, request, context):
        try:
            vs = _get_vector_store()
            info = vs._call_with_retry(
                "get_collection", vs.client.get_collection, collection_name=request.name
            )
            payload = info.model_dump(mode="json") if hasattr(info, "model_dump") else info.dict()
            return pb2.JsonPayloadResponse(payload_json=json.dumps(payload))
        except Exception as e:
            abort_rpc_error(context, logger, "GetVectorCollection", e)

    def SearchVectorCollection(self, request, context):
        try:
            # Strongly typed mode field on proto, with fallback to JSON metadata
            mode = getattr(request, "mode", None) or "dense"
            filters_dict = json.loads(request.filters_json) if request.filters_json else {}
            if isinstance(filters_dict, dict) and "_search_mode" in filters_dict:
                mode = filters_dict.pop("_search_mode")
                cleaned_filters_json = json.dumps(filters_dict) if filters_dict else ""
            else:
                cleaned_filters_json = request.filters_json

            query_filter = _build_qdrant_filter(cleaned_filters_json)
            # RRF fusion scores are ~0.016-0.033; only apply score_threshold for dense cosine and sparse dot-product search
            score_threshold = (
                request.score_threshold
                if (request.has_score_threshold and mode in ("dense", "sparse"))
                else None
            )
            vs = _get_vector_store()
            from engine.core.resource_loader import _get_shared_models
            from engine.nlp.text_cleaner import TextPipeline

            embed_engine, _ = _get_shared_models()
            raw_query = (request.query or "").strip()
            normalized_query = TextPipeline.normalize_final(
                TextPipeline.standardize_units(raw_query)
            )

            # Collections ending in _catalog (or generic collections) store clean_text which is normalized lowercase.
            # Use normalized query for catalog dense matching; for sparse/hybrid, encode both raw and normalized
            # so BGE-M3's case-sensitive subword vocabulary matches regardless of user input casing.
            is_catalog = request.name.endswith("_catalog") or not request.name.endswith("_tags")
            dense_target = normalized_query if (is_catalog and normalized_query) else raw_query

            # Strictly encode the normalized query so BGE-M3 produces consistent subword token IDs
            # regardless of query casing or script (case-insensitive for Latin, Sinhala, etc.)
            encoded = embed_engine.encode([dense_target])
            dense_vec = encoded["dense"][0]
            dense_list = dense_vec.tolist() if hasattr(dense_vec, "tolist") else dense_vec

            if mode in ("sparse", "hybrid"):
                sparse_list = encoded.get("sparse", [{}])
                sparse_dict = sparse_list[0] if sparse_list else {}

                from qdrant_client.http.models import SparseVector

                indices, values = [], []
                for k, v in sparse_dict.items():
                    try:
                        indices.append(int(k))
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass
                sparse_vector = SparseVector(indices=indices, values=values)

                if mode == "sparse":
                    q_words = normalized_query.split()
                    q_len = len(q_words)
                    fetch_limit = max((request.limit or 10) * 3, 30)

                    raw_points = vs._call_with_retry(
                        "query_points (sparse collection search)",
                        vs.client.query_points,
                        collection_name=request.name,
                        query=sparse_vector,
                        using="sparse",
                        limit=fetch_limit,
                        query_filter=query_filter,
                        score_threshold=score_threshold,
                        with_payload=True,
                    ).points

                    # Apply length normalization penalty: min(1.0, |Q| / |D|) with exact-match priority
                    scored_points = []
                    for hit in raw_points:
                        p_load = hit.payload or {}
                        clean_name = str(p_load.get("clean_text") or "").strip().lower()
                        doc_len = p_load.get("token_count") or (
                            len(clean_name.split()) if clean_name else q_len
                        )
                        len_ratio = min(1.0, q_len / max(1, doc_len)) if q_len > 0 else 1.0

                        is_exact = bool(clean_name and clean_name == normalized_query)
                        boost = 1.25 if is_exact else 1.0
                        adjusted_score = hit.score * len_ratio * boost
                        hit.score = float(adjusted_score)
                        scored_points.append((adjusted_score, is_exact, hit))

                    scored_points.sort(key=lambda x: (x[0], 1 if x[1] else 0), reverse=True)
                    final_limit = request.limit or 10
                    points = [sp[2] for sp in scored_points[:final_limit]]
                else:  # hybrid
                    from qdrant_client.http.models import Fusion, FusionQuery, Prefetch

                    points = vs._call_with_retry(
                        "query_points (hybrid collection search)",
                        vs.client.query_points,
                        collection_name=request.name,
                        prefetch=[
                            Prefetch(query=dense_list, using="dense", limit=request.limit or 10),
                            Prefetch(
                                query=sparse_vector, using="sparse", limit=request.limit or 10
                            ),
                        ],
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=request.limit or 10,
                        query_filter=query_filter,
                        with_payload=True,
                    ).points
            else:  # dense
                points = vs._call_with_retry(
                    "query_points (dense collection search)",
                    vs.client.query_points,
                    collection_name=request.name,
                    query=dense_list,
                    using="dense",
                    limit=request.limit or 10,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                ).points

            res = [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in points]
            return pb2.JsonPayloadResponse(payload_json=json.dumps({"results": res}))
        except Exception as e:
            abort_rpc_error(context, logger, "SearchVectorCollection", e)

    def RerunRules(self, request, context):
        try:
            gk_list = [x.strip() for x in request.gk.split(",") if x.strip()]
            record = {
                "sku_name": request.sku_name,
                "domain": request.domain,
                "bt": request.bt,
                "gk": gk_list,
                "region": request.region if request.domain == "food" else None,
                "category": request.category if request.domain == "market" else None,
                "price": request.price,
                "confidence": request.confidence,
                "match_source": request.match_source,
                "matched_sku": request.matched_sku,
                "reasoning": request.reasoning,
                "rules_applied": [],
            }
            with pinned_rules(snapshot_rules()):
                aug_record = run_rules_engine(record)

            rules_applied = aug_record.get("rules_applied", [])
            rules_applied_str = json.dumps(rules_applied) if rules_applied else ""

            return pb2.RerunRulesResponse(
                suggested_bt=str(aug_record.get("bt") or ""),
                suggested_gk=", ".join(aug_record.get("gk", [])),
                suggested_region=str(aug_record.get("region") or aug_record.get("category") or ""),
                rules_applied=rules_applied_str,
            )
        except Exception as e:
            abort_rpc_error(context, logger, "RerunRules", e)

    def TestRule(self, request, context):
        try:
            rule_dict = json.loads(request.rule_json)
            sample_record = json.loads(request.sample_record_json)

            row_values = (
                rule_dict.get("id", 0),
                rule_dict["rule_id"],
                rule_dict["domain"],
                rule_dict["priority"],
                rule_dict["description"],
                rule_dict["reasoning"],
                rule_dict["condition_logic"],
                rule_dict.get("is_active", 1),
            )
            engine_rule = Rule(row_values)
            engine_rule.conditions = rule_dict.get("conditions", [])
            engine_rule.actions = rule_dict.get("actions", [])

            record = dict(sample_record)
            fires = evaluate_conditions(engine_rule, record)
            changes = ""
            if fires:
                changes = apply_actions(engine_rule, record)

            return pb2.TestRuleResponse(
                rule_id=rule_dict["rule_id"],
                fires=bool(fires),
                matched_conditions_json=json.dumps(
                    rule_dict.get("conditions", []) if fires else []
                ),
                actions_applied_json=json.dumps(rule_dict.get("actions", []) if fires else []),
                changes=str(changes),
                sample_record_after_json=json.dumps(record),
            )
        except Exception as e:
            abort_rpc_error(context, logger, "TestRule", e)


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
