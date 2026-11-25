"""Shared gRPC channel options, metadata builders, error mappers, and server interceptors."""

import hmac
import sys
from collections.abc import Callable
from typing import Any

import grpc

from backend.core import config
from backend.core.exceptions import (
    ResourceNotFoundError,
    ServiceValidationError,
    UpstreamGatewayError,
)
from backend.core.observability import VALID_TRACE_PATTERN, bind_trace_context, get_trace_id

DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64 MB


def build_grpc_channel_options() -> list[tuple[str, Any]]:
    """Returns gRPC send/receive message size limits for large SKU batches."""
    return [
        ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE_BYTES),
    ]


def _resolve_internal_secret() -> str:
    secret = config.INTERNAL_GRPC_SECRET
    if not secret and "engine.core.config" in sys.modules:
        secret = getattr(sys.modules["engine.core.config"], "INTERNAL_GRPC_SECRET", "") or ""
    return secret


def build_grpc_metadata(trace_id: str | None = None) -> list[tuple[str, str]]:
    """Constructs outbound gRPC metadata carrying trace context and optional internal secret."""
    metadata: list[tuple[str, str]] = []
    tid = trace_id or get_trace_id()
    if tid and tid != "-" and VALID_TRACE_PATTERN.match(tid):
        metadata.append(("x-request-id", tid))
    secret = _resolve_internal_secret()
    if secret:
        metadata.append(("x-internal-secret", secret))
    return metadata


def unwrap_rpc_error(exc: Exception) -> Exception:
    """Translates a gRPC RpcError into the corresponding Gateway domain exception."""
    if not isinstance(exc, grpc.RpcError):
        return exc

    code = exc.code()
    details = exc.details() or str(exc)

    if code == grpc.StatusCode.INVALID_ARGUMENT:
        return ServiceValidationError(details)
    if code == grpc.StatusCode.NOT_FOUND:
        return ResourceNotFoundError(details)
    if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
        return PermissionError(details)
    return UpstreamGatewayError(
        f"Engine gRPC error ({code.name if code else 'UNKNOWN'}): {details}"
    )


class GrpcContextInterceptor(grpc.ServerInterceptor):
    """Enforces shared internal gRPC authentication and binds incoming trace IDs."""

    def intercept_service(self, continuation: Callable, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or ())
        expected_secret = _resolve_internal_secret()
        if expected_secret:
            provided = metadata.get("x-internal-secret", "")
            if not provided or not hmac.compare_digest(provided, expected_secret):

                def _abort_unauthenticated(_request, context):
                    context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "Invalid or missing internal gRPC secret.",
                    )

                return grpc.unary_unary_rpc_method_handler(_abort_unauthenticated)

        raw_trace = metadata.get("x-request-id")
        trace_id = raw_trace if (raw_trace and VALID_TRACE_PATTERN.match(raw_trace)) else None

        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler

        inner = handler.unary_unary

        def _wrapped(request, context):
            with bind_trace_context(trace_id):
                return inner(request, context)

        return grpc.unary_unary_rpc_method_handler(
            _wrapped,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )
