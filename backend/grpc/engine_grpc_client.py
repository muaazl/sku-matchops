"""Outbound gRPC client implementing `EngineGatewayProtocol` against the ML Engine."""

import json
import logging
import threading
from typing import Any

import grpc

from backend.core import config
from backend.grpc.grpc_utils import (
    build_grpc_channel_options,
    build_grpc_metadata,
    unwrap_rpc_error,
)
from backend.protos import engine_service_pb2, engine_service_pb2_grpc
from backend.services import engine_gateway_service, rule_service

logger = logging.getLogger("matchops.grpc.client")

_channel_lock = threading.Lock()
_channel: grpc.Channel | None = None
_channel_target: str | None = None
_stub: engine_service_pb2_grpc.EngineInferenceServiceStub | None = None


def get_engine_stub() -> engine_service_pb2_grpc.EngineInferenceServiceStub:
    """Returns a cached gRPC stub targeting `config.ENGINE_GRPC_URL`, recreating if changed."""
    global _channel, _channel_target, _stub
    target = config.ENGINE_GRPC_URL
    with _channel_lock:
        if _stub is None or _channel_target != target:
            if _channel is not None:
                try:
                    _channel.close()
                except Exception:
                    pass
            _channel = grpc.insecure_channel(target, options=build_grpc_channel_options())
            _channel_target = target
            _stub = engine_service_pb2_grpc.EngineInferenceServiceStub(_channel)
        return _stub


def close_engine_channel() -> None:
    """Closes the cached outbound gRPC channel to the ML Engine."""
    global _channel, _channel_target, _stub
    with _channel_lock:
        if _channel is not None:
            try:
                _channel.close()
            except Exception:
                pass
        _channel = None
        _channel_target = None
        _stub = None


def _pipeline_item_to_dict(item: engine_service_pb2.PipelineResultItem) -> dict[str, Any]:
    return {
        "sku_name": item.sku_name,
        "matched_catalog_name": item.matched_catalog_name,
        "score": float(item.score),
        "status": item.status,
        "logic_notes": item.logic_notes,
        "suggested_bt": item.suggested_bt,
        "bt_confidence": float(item.bt_confidence),
        "bt_status": item.bt_status,
        "bt_source": item.bt_source,
        "model": item.model,
        "bt_model": item.bt_model,
        "suggested_gk": item.suggested_gk,
        "gk_confidence": float(item.gk_confidence),
        "gk_status": item.gk_status,
        "suggested_region": item.suggested_region,
        "region_confidence": float(item.region_confidence),
        "region_status": item.region_status,
        "region_source": item.region_source,
        "rules_applied": item.rules_applied,
        "pipeline_source": item.pipeline_source,
        "extracted_flavors": list(item.extracted_flavors),
        "extracted_brands": list(item.extracted_brands),
    }


class EngineGrpcGateway:
    """Concrete gRPC gateway adapter registered into `engine_gateway_service`."""

    def get_engine_health(self) -> dict[str, Any]:
        try:
            resp = get_engine_stub().GetHealth(
                engine_service_pb2.Empty(),
                metadata=build_grpc_metadata(),
                timeout=5.0,
            )
            return {
                "status": resp.status,
                "service": resp.service,
                "loaded_domains": list(resp.loaded_domains),
                "all_models_ready": bool(resp.all_models_ready),
            }
        except Exception as exc:
            return {
                "status": "offline",
                "loaded_domains": [],
                "all_models_ready": False,
                "error": str(exc),
            }

    def get_models_status(self) -> dict[str, Any]:
        try:
            resp = get_engine_stub().GetModelsStatus(
                engine_service_pb2.Empty(),
                metadata=build_grpc_metadata(),
                timeout=5.0,
            )
            domains_dict = {
                dom: {"pipeline": val.pipeline, "classifier": val.classifier}
                for dom, val in resp.domains.items()
            }
            is_loaded = resp.status in ("loaded", "ready", "ok", "all_ready") or any(
                d.get("pipeline") == "loaded" or d.get("classifier") == "loaded"
                for d in domains_dict.values()
            )
            return {
                "status": resp.status,
                "domains": domains_dict,
                "loaded": is_loaded,
                "all_models_ready": resp.status in ("loaded", "ready", "all_ready"),
            }
        except Exception as exc:
            return {
                "status": "offline",
                "domains": {},
                "loaded": False,
                "all_models_ready": False,
                "loading_in_progress": False,
                "error": str(exc),
            }

    def trigger_load_models(self) -> dict[str, Any]:
        try:
            resp = get_engine_stub().ManageModels(
                engine_service_pb2.ManageModelsRequest(action="load"),
                metadata=build_grpc_metadata(),
                timeout=15.0,
            )
            return {"status": resp.status, "message": resp.message}
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def reload_engine_models(self) -> dict[str, Any]:
        try:
            resp = get_engine_stub().ManageModels(
                engine_service_pb2.ManageModelsRequest(action="reload"),
                metadata=build_grpc_metadata(),
                timeout=30.0,
            )
            return {"status": resp.status, "message": resp.message}
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def refresh_engine_rules(
        self, rules_data: list[dict] | None = None, version: str = ""
    ) -> dict[str, Any]:
        if rules_data is None:
            rules_data = rule_service.load_active_rules()
        if not version:
            version = rule_service.compute_rules_version(rules_data)
        try:
            resp = get_engine_stub().RefreshRules(
                engine_service_pb2.RefreshRulesRequest(
                    rules_json=json.dumps(rules_data, default=str),
                    version=version,
                ),
                metadata=build_grpc_metadata(),
                timeout=10.0,
            )
            return {"status": resp.status, "message": resp.message}
        except Exception as exc:
            logger.warning(f"Failed to push rules snapshot to ML Engine via gRPC: {exc}")
            return {"status": "offline", "message": str(exc)}

    def dispatch_batch_job(self, job_id: str, request: Any, task: str) -> dict[str, Any]:
        try:
            rules_data = rule_service.load_active_rules()
            rules_json = json.dumps(rules_data, default=str)
        except Exception as err:
            logger.debug(f"Dispatching batch job {job_id} without inline DB rules: {err}")
            rules_json = ""
        pb_skus = [
            engine_service_pb2.SKUInput(
                name=s.name,
                price=float(s.price or 0.0),
                description=s.description or "",
                category=s.category or "",
            )
            for s in request.skus
        ]
        req = engine_service_pb2.BatchJobRequest(
            job_id=str(job_id),
            task=task,
            domain=request.domain or "market",
            skus=pb_skus,
            callback_url=request.callback_url or "",
            sheet_name=request.sheet_name or "",
            created_by="",
            rules_json=rules_json,
            spreadsheet_id=request.spreadsheet_id or "",
            backend_grpc_url=config.BACKEND_GRPC_URL,
        )
        try:
            resp = get_engine_stub().DispatchBatchJob(
                req,
                metadata=build_grpc_metadata(),
                timeout=15.0,
            )
            return {"status": resp.status, "job_id": resp.job_id}
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def dispatch_upload_job(
        self,
        job_id: str,
        skus_payload: list[dict[str, Any]],
        domain: str,
        outlet_id_or_name: str,
        task: str,
    ) -> dict[str, Any]:
        pb_skus = [
            engine_service_pb2.UploadSkuItem(
                fields={str(k): "" if v is None else str(v) for k, v in row.items()}
            )
            for row in skus_payload
        ]
        req = engine_service_pb2.UploadJobRequest(
            job_id=str(job_id),
            task=task,
            domain=domain or "market",
            file_path="",
            callback_url="",
            sheet_name=outlet_id_or_name or "",
            created_by="",
            skus_json="",
            backend_grpc_url=config.BACKEND_GRPC_URL,
            skus=pb_skus,
        )
        try:
            resp = get_engine_stub().DispatchUploadJob(
                req,
                metadata=build_grpc_metadata(),
                timeout=15.0,
            )
            return {"status": resp.status, "job_id": resp.job_id}
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def cancel_engine_job(self, job_id: str) -> bool:
        try:
            resp = get_engine_stub().CancelJob(
                engine_service_pb2.CancelJobRequest(job_id=str(job_id)),
                metadata=build_grpc_metadata(),
                timeout=5.0,
            )
            return resp.status in ("cancelled", "cancelling", "ok")
        except Exception as exc:
            logger.debug(f"Could not cancel job {job_id} on ML Engine: {exc}")
            return False

    def run_single(
        self,
        sku_name: str,
        domain: str = "market",
        task: str = "pipeline",
        price: float = 0.0,
        description: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        try:
            rules_data = rule_service.load_active_rules()
            rules_json = json.dumps(rules_data, default=str)
        except Exception as err:
            logger.debug(f"Running single SKU inference without inline DB rules: {err}")
            rules_json = ""
        req = engine_service_pb2.SingleRunRequest(
            task=task,
            domain=domain,
            skus=[
                engine_service_pb2.SKUInput(
                    name=sku_name,
                    price=float(price or 0.0),
                    description=description or "",
                    category=category or "",
                )
            ],
            rules_json=rules_json,
        )
        try:
            resp = get_engine_stub().RunSingle(
                req,
                metadata=build_grpc_metadata(),
                timeout=60.0,
            )
            return {
                "status": resp.status,
                "task": resp.task,
                "domain": resp.domain,
                "total": resp.total,
                "results": [_pipeline_item_to_dict(item) for item in resp.results],
                "stage_timings": dict(resp.stage_timings),
                "duration_seconds": float(resp.duration_seconds),
                "rule_version": resp.rule_version,
            }
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def suggest_tags(
        self,
        sku_name: str,
        domain: str = "market",
        current_bt: str = "",
        exclude_bt: str = "",
        exclude_gk: str = "",
    ) -> dict[str, Any]:
        req = engine_service_pb2.SuggestRequest(
            sku_name=sku_name,
            domain=domain,
            current_bt=current_bt or "",
            exclude_bt=exclude_bt or "",
            exclude_gk=exclude_gk or "",
        )
        try:
            resp = get_engine_stub().SuggestTags(
                req,
                metadata=build_grpc_metadata(),
                timeout=30.0,
            )
            return {
                "status": resp.status,
                "sku_name": resp.sku_name,
                "suggested_bt": resp.suggested_bt,
                "suggested_gk": resp.suggested_gk,
                "suggested_region": resp.suggested_region,
                "template_used": resp.template_used,
                "is_match": bool(resp.is_match),
            }
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def run_sku_audit(
        self,
        sku_name: str,
        domain: str = "market",
        task: str = "pipeline",
        price: float = 0.0,
        description: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        req = engine_service_pb2.AuditRequest(
            sku_name=sku_name,
            domain=domain,
            task=task,
            price=float(price or 0.0),
            description=description or "",
            category=category or "",
        )
        try:
            resp = get_engine_stub().RunAudit(
                req,
                metadata=build_grpc_metadata(),
                timeout=30.0,
            )
            return json.loads(resp.audit_json) if resp.audit_json else {}
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def rerun_rules_via_engine(
        self,
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
    ) -> dict[str, Any]:
        req = engine_service_pb2.RerunRulesRequest(
            sku_name=sku_name,
            domain=domain,
            bt=bt,
            gk=gk,
            region=region or "",
            category=category or "",
            price=float(price or 0.0),
            confidence=float(confidence if confidence is not None else 1.0),
            match_source=match_source or "Classifier",
            matched_sku=matched_sku or "",
            reasoning=reasoning or "",
        )
        try:
            resp = get_engine_stub().RerunRules(
                req,
                metadata=build_grpc_metadata(),
                timeout=15.0,
            )
            return {
                "suggested_bt": resp.suggested_bt,
                "suggested_gk": resp.suggested_gk,
                "suggested_region": resp.suggested_region,
                "rules_applied": resp.rules_applied,
            }
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def test_rule_via_engine(
        self,
        rule_dict: dict[str, Any],
        sample_record: dict[str, Any],
    ) -> dict[str, Any]:
        req = engine_service_pb2.TestRuleRequest(
            rule_json=json.dumps(rule_dict, default=str),
            sample_record_json=json.dumps(sample_record, default=str),
        )
        try:
            resp = get_engine_stub().TestRule(
                req,
                metadata=build_grpc_metadata(),
                timeout=15.0,
            )
            return {
                "rule_id": resp.rule_id,
                "fires": bool(resp.fires),
                "matched_conditions": json.loads(resp.matched_conditions_json or "[]"),
                "actions_applied": json.loads(resp.actions_applied_json or "[]"),
                "changes": resp.changes,
                "sample_record_after": json.loads(resp.sample_record_after_json or "{}"),
            }
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def list_vector_collections(self) -> dict[str, Any]:
        try:
            resp = get_engine_stub().ListVectorCollections(
                engine_service_pb2.Empty(),
                metadata=build_grpc_metadata(),
                timeout=10.0,
            )
            return json.loads(resp.payload_json or "{}")
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def get_vector_collection(self, name: str) -> dict[str, Any]:
        try:
            resp = get_engine_stub().GetVectorCollection(
                engine_service_pb2.VectorCollectionRequest(name=name),
                metadata=build_grpc_metadata(),
                timeout=10.0,
            )
            return json.loads(resp.payload_json or "{}")
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc

    def search_vector_collection(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        score_threshold = payload.get("score_threshold")
        req = engine_service_pb2.VectorSearchRequest(
            name=name,
            query=payload.get("query", ""),
            limit=int(payload.get("top_k", 10)),
            score_threshold=float(score_threshold) if score_threshold is not None else 0.0,
            has_score_threshold=score_threshold is not None,
            filters_json=json.dumps(payload.get("filters") or {}),
            mode=payload.get("mode", "dense"),
        )
        try:
            resp = get_engine_stub().SearchVectorCollection(
                req,
                metadata=build_grpc_metadata(),
                timeout=15.0,
            )
            return json.loads(resp.payload_json or "{}")
        except Exception as exc:
            raise unwrap_rpc_error(exc) from exc


def register_engine_grpc_gateway() -> EngineGrpcGateway:
    """Instantiates and registers the default gRPC Engine gateway adapter."""
    gateway = EngineGrpcGateway()
    engine_gateway_service.register_engine_gateway(gateway)
    return gateway
