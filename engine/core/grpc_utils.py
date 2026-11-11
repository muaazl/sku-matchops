"""
Shared gRPC utilities for the Gateway and ML Engine: channel limits, request-trace + internal-secret
propagation/interception, error mapping, and typed Protobuf <-> dict converters.
"""

import hmac
import logging
from typing import Any

import grpc

from engine import config
from engine.core.observability import _VALID_TRACE, TRACE_ID
from engine.protos import engine_service_pb2 as pb2

GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
)


def grpc_metadata() -> tuple[tuple[str, str], ...]:
    """Outbound gRPC metadata carrying active trace_id and optional shared internal secret."""
    meta: list[tuple[str, str]] = []
    trace_id = TRACE_ID.get()
    if trace_id != "-":
        meta.append(("x-request-id", trace_id))
    if config.INTERNAL_GRPC_SECRET:
        meta.append(("x-internal-secret", config.INTERNAL_GRPC_SECRET))
    return tuple(meta)


class GrpcContextInterceptor(grpc.ServerInterceptor):
    """Validates optional INTERNAL_GRPC_SECRET and scopes TRACE_ID cleanly per RPC."""

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler

        meta = {k.lower(): v for k, v in (handler_call_details.invocation_metadata or ())}
        secret = config.INTERNAL_GRPC_SECRET
        if secret:
            incoming = str(meta.get("x-internal-secret", ""))
            if not incoming or not hmac.compare_digest(incoming, secret):

                def _unauthenticated(_request, context):
                    context.abort(
                        grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing internal gRPC secret."
                    )

                return grpc.unary_unary_rpc_method_handler(
                    _unauthenticated,
                    request_deserializer=handler.request_deserializer,
                    response_serializer=handler.response_serializer,
                )

        raw_trace = str(meta.get("x-request-id", ""))
        trace_id = raw_trace if _VALID_TRACE.match(raw_trace) else "-"
        inner = handler.unary_unary

        def _scoped(request, context):
            token = TRACE_ID.set(trace_id)
            try:
                return inner(request, context)
            finally:
                TRACE_ID.reset(token)

        return grpc.unary_unary_rpc_method_handler(
            _scoped,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def is_not_found_error(err: Exception) -> bool:
    if isinstance(err, KeyError) or getattr(err, "status_code", None) == 404:
        return True
    if isinstance(err, grpc.RpcError) and err.code() == grpc.StatusCode.NOT_FOUND:
        return True
    msg = str(err).lower()
    return "collection" in msg and ("not found" in msg or "doesn't exist" in msg)


def abort_rpc_error(context, logger: logging.Logger, op: str, err: Exception) -> None:
    """Maps domain/validation/not-found errors to typed gRPC StatusCodes."""
    if is_not_found_error(err):
        context.abort(grpc.StatusCode.NOT_FOUND, str(err))
    if isinstance(err, ValueError):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))
    logger.error(f"[gRPC] {op} failed: {err}", exc_info=True)
    context.abort(grpc.StatusCode.INTERNAL, str(err))


def unwrap_rpc_error(err: Exception, target: str) -> None:
    """Re-raises client gRPC status codes as ValueError / KeyError; wraps transport errors."""
    if isinstance(err, grpc.RpcError):
        code = err.code()
        detail = err.details() or str(err)
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            raise ValueError(detail)
        if code == grpc.StatusCode.NOT_FOUND:
            raise KeyError(detail)
    raise RuntimeError(f"Could not reach ML Engine via gRPC at {target}: {err}")


def dict_to_sku_proto(s: dict[str, Any]) -> pb2.SKUInput:
    return pb2.SKUInput(
        name=str(s.get("name", s.get("Name", "")) or ""),
        price=float(s.get("price", s.get("Price", 0.0)) or 0.0),
        description=str(s.get("description", s.get("Description", "")) or ""),
        category=str(s.get("category", s.get("Category", "")) or ""),
    )


def sku_proto_to_dict(s: pb2.SKUInput) -> dict[str, Any]:
    return {
        "name": s.name,
        "price": s.price,
        "description": s.description,
        "category": s.category,
    }


def stage_timings_map(res: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v) for k, v in res.get("stage_timings", {}).items() if isinstance(v, (int, float))
    }


def result_to_pipeline_proto(item: dict[str, Any]) -> pb2.PipelineResultItem:
    entities = item.get("input_entities") or {}
    reg_conf = item.get("region_confidence")
    if reg_conf is None:
        reg_conf = item.get("category_confidence", 0.0)
    return pb2.PipelineResultItem(
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
        region_confidence=float(reg_conf or 0.0),
        region_status=str(item.get("region_status") or ""),
        region_source=str(item.get("region_source") or ""),
        rules_applied=str(item.get("rules_applied") or ""),
        pipeline_source=str(item.get("pipeline_source") or ""),
        extracted_flavors=[str(x) for x in (entities.get("flavor") or [])],
        extracted_brands=[str(x) for x in (entities.get("brand") or [])],
        sku_name=str(item.get("sku_name") or ""),
    )


def pipeline_proto_to_result(
    r: pb2.PipelineResultItem, include_internal: bool = False
) -> dict[str, Any]:
    out: dict[str, Any] = {
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
    if include_internal:
        if r.pipeline_source:
            out["pipeline_source"] = r.pipeline_source
        if r.sku_name:
            out["sku_name"] = r.sku_name
        if r.extracted_flavors or r.extracted_brands:
            out["input_entities"] = {
                "flavor": list(r.extracted_flavors),
                "brand": list(r.extracted_brands),
            }
    return out
