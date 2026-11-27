"""Backend API Gateway FastAPI application composition root."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.middleware.audit_logging_middleware import AuditLoggingMiddleware
from backend.api.middleware.etag_middleware import ETagMiddleware
from backend.api.middleware.trace_id_middleware import TraceIdMiddleware
from backend.api.router import api_router
from backend.core import config
from backend.core.observability import configure_logging
from backend.grpc.callback_servicer import (
    ensure_backend_grpc_server,
    stop_backend_grpc_server,
)
from backend.services import system_service


def configure_app_logging() -> None:
    """Configures console logging for the Backend Gateway."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(type(h) is logging.StreamHandler for h in root_logger.handlers):
        root_logger.addHandler(logging.StreamHandler())
    configure_logging(root_logger, log_format=config.LOG_FORMAT)


configure_app_logging()
logger = logging.getLogger("matchops.main")


def startup_event() -> None:
    """Starts the callback gRPC server and executes Gateway startup checks."""
    ensure_backend_grpc_server()
    system_service.run_startup_sequence()


def shutdown_event() -> None:
    """Stops the Backend callback gRPC server on process exit."""
    stop_backend_grpc_server()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup_event()
    try:
        yield
    finally:
        shutdown_event()
        try:
            await system_service.close_async_resources()
        except Exception:
            pass


app = FastAPI(
    title="MatchOps API Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(ETagMiddleware)
# Binds trace id before audit logging and every request handler
app.add_middleware(TraceIdMiddleware)
# Outermost: ensures CORS headers apply even when inner middleware short-circuits
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Request-ID", "ETag"],
)

app.include_router(api_router)
