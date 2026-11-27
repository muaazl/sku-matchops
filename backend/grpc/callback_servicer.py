"""Inbound gRPC Callback Server receiving asynchronous job lifecycle events from ML Engine."""

import json
import logging
import threading
from concurrent import futures
from typing import Any

import grpc

from backend.core import config
from backend.grpc.grpc_utils import GrpcContextInterceptor, build_grpc_channel_options
from backend.protos import engine_service_pb2, engine_service_pb2_grpc
from backend.services import audit_service, job_callback_service

logger = logging.getLogger("matchops.grpc.callback_server")

_grpc_server: grpc.Server | None = None
_server_lock = threading.Lock()


def _proto_result_to_dict(item: engine_service_pb2.PipelineResultItem) -> dict[str, Any]:
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
        "input_entities": {
            "flavor": list(item.extracted_flavors),
            "brand": list(item.extracted_brands),
        },
    }


class BackendCallbackServicer(engine_service_pb2_grpc.BackendCallbackServiceServicer):
    """Implements BackendCallbackService RPCs and delegates persistence to Gateway services."""

    def ReportJobProgress(self, request, context):
        try:
            job_callback_service.handle_job_progress(
                job_id=request.job_id,
                progress_pct=float(request.progress_pct),
                eta=int(request.eta_seconds) if request.has_eta else None,
                current_stage=request.current_stage,
            )
        except Exception as exc:
            logger.exception(f"[JOB {request.job_id}] ReportJobProgress failed: {exc}")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return engine_service_pb2.JobCallbackResponse(status="ok")

    def CompleteJob(self, request, context):
        try:
            if request.results:
                results = [_proto_result_to_dict(item) for item in request.results]
            elif request.results_json:
                results = json.loads(request.results_json)
            else:
                results = []

            written = job_callback_service.record_job_completion(
                job_id=request.job_id,
                domain=request.domain,
                task=request.task,
                results=results,
                rule_version=request.rule_version,
                duration_minutes=float(request.duration_minutes),
                high_conf=int(request.high_conf),
                med_conf=int(request.med_conf),
                low_conf=int(request.low_conf),
                match_rate=float(request.match_rate),
            )
        except Exception as exc:
            logger.exception(f"[JOB {request.job_id}] CompleteJob persistence failed: {exc}")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        if request.webhook_log_json:
            try:
                webhook = json.loads(request.webhook_log_json)
                if isinstance(webhook, dict):
                    audit_service.record_outbound_request(
                        url=str(webhook.get("url", "")),
                        method=str(webhook.get("method", "POST")),
                        payload=webhook.get("payload", {}),
                        response_status=int(webhook.get("response_status", 0)),
                        response_text=str(webhook.get("response_text", "")),
                        duration_ms=int(webhook.get("duration_ms", 0)),
                        path=str(webhook.get("path", "/doPost")),
                    )
            except Exception as webhook_err:
                logger.warning(f"Failed to parse webhook audit payload: {webhook_err}")

        return engine_service_pb2.JobCallbackResponse(status="ignored" if written == -1 else "ok")

    def FailJob(self, request, context):
        try:
            is_cancelled = bool(request.is_cancelled or request.status == "cancelled")
            job_callback_service.record_job_failure(
                job_id=request.job_id,
                error_message=request.error_message,
                duration_minutes=float(request.duration_minutes),
                is_cancelled=is_cancelled,
            )
        except Exception as exc:
            logger.exception(f"[JOB {request.job_id}] FailJob persistence failed: {exc}")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return engine_service_pb2.JobCallbackResponse(status="ok")


def start_backend_grpc_server(
    port: int | None = None,
    max_workers: int = 8,
    host: str = "[::]",
) -> grpc.Server:
    """Starts and binds the inbound Backend callback gRPC server."""
    bind_port = port if port is not None else config.BACKEND_GRPC_PORT
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=[GrpcContextInterceptor()],
        options=build_grpc_channel_options(),
    )
    engine_service_pb2_grpc.add_BackendCallbackServiceServicer_to_server(
        BackendCallbackServicer(),
        server,
    )
    bound_port = server.add_insecure_port(f"{host}:{bind_port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind Backend callback gRPC server on {host}:{bind_port}")
    server._bound_port = bound_port  # type: ignore[attr-defined]
    server.start()
    logger.info(f"Backend callback gRPC server listening on {host}:{bound_port}")
    return server


def ensure_backend_grpc_server(port: int | None = None) -> grpc.Server:
    """Starts the singleton Backend callback gRPC server if not already running."""
    global _grpc_server
    with _server_lock:
        if _grpc_server is None:
            _grpc_server = start_backend_grpc_server(port=port)
        return _grpc_server


def stop_backend_grpc_server(grace: float = 2.0) -> None:
    """Stops the singleton Backend callback gRPC server if running."""
    global _grpc_server
    with _server_lock:
        if _grpc_server is not None:
            _grpc_server.stop(grace=grace)
            _grpc_server = None
            logger.info("Backend callback gRPC server stopped.")
