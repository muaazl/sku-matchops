"""Backend gRPC transport package (outbound Engine client and inbound callback server)."""

from backend.grpc.callback_servicer import (
    BackendCallbackServicer,
    ensure_backend_grpc_server,
    start_backend_grpc_server,
    stop_backend_grpc_server,
)
from backend.grpc.engine_grpc_client import (
    EngineGrpcGateway,
    close_engine_channel,
    get_engine_stub,
    register_engine_grpc_gateway,
)

register_engine_grpc_gateway()

__all__ = [
    "BackendCallbackServicer",
    "EngineGrpcGateway",
    "close_engine_channel",
    "ensure_backend_grpc_server",
    "get_engine_stub",
    "register_engine_grpc_gateway",
    "start_backend_grpc_server",
    "stop_backend_grpc_server",
]
