"""
gRPC Server implementation for SKU MatchOps Backend Gateway.
Receives progress, completion, and failure callbacks from ML Engine workers over gRPC
(port 50052 by default) and persists terminal job outcomes to PostgreSQL.
"""

import json
import logging
import threading
from concurrent import futures

import grpc
from sqlalchemy import text

from backend.app.core import config
from backend.app.db.session import get_engine, log_outbound_request
from backend.app.services.persistence import record_job_completion, record_job_failure
from engine.core.grpc_utils import (
    GRPC_CHANNEL_OPTIONS,
    GrpcContextInterceptor,
    pipeline_proto_to_result,
)
from engine.protos import engine_service_pb2 as pb2
from engine.protos import engine_service_pb2_grpc as pb2_grpc

logger = logging.getLogger("matchops.backend.grpc")

# In-memory store for fast job polling & dynamic ETA computation (active jobs only)
_job_progress: dict[str, float] = {}
_job_eta: dict[str, int | None] = {}
_job_stage: dict[str, str] = {}

_server_lock = threading.Lock()
_active_server: grpc.Server | None = None


def clear_job_progress(job_id: str) -> None:
    """Evicts terminal job progress/ETA/stage entries so memory stays O(J_active)."""
    jid = str(job_id)
    _job_progress.pop(jid, None)
    _job_eta.pop(jid, None)
    _job_stage.pop(jid, None)


class BackendCallbackServicer(pb2_grpc.BackendCallbackServiceServicer):
    def ReportJobProgress(self, request, context):
        j_id = str(request.job_id)
        _job_progress[j_id] = float(request.progress_pct)
        _job_eta[j_id] = int(request.eta_seconds) if request.has_eta else None

        # Only perform a DB write when the stage changes
        last_stage = _job_stage.get(j_id)
        if last_stage != request.current_stage:
            _job_stage[j_id] = request.current_stage
            try:
                with get_engine().begin() as conn:
                    # Never resurrect a job that already reached a terminal state
                    cur = conn.execute(
                        text(
                            "UPDATE jobs SET status = 'running', current_stage = :stage, updated_at = now() "
                            "WHERE id = :id AND status IN ('queued', 'running')"
                        ),
                        {"stage": request.current_stage, "id": j_id},
                    )
                    if cur.rowcount == 0:
                        clear_job_progress(j_id)
            except Exception as e:
                logger.error(f"Failed to update progress for job {j_id} in DB: {e}")

        return pb2.JobCallbackResponse(status="ok")

    def CompleteJob(self, request, context):
        j_id = str(request.job_id)
        try:
            if request.results:
                results = [
                    pipeline_proto_to_result(r, include_internal=True) for r in request.results
                ]
            else:
                results = json.loads(request.results_json) if request.results_json else []
            record_job_completion(
                job_id=j_id,
                domain=request.domain or "market",
                task=request.task or "pipeline",
                results=results,
                rule_version=request.rule_version or "",
                duration_minutes=float(request.duration_minutes),
                high_conf=int(request.high_conf),
                med_conf=int(request.med_conf),
                low_conf=int(request.low_conf),
                match_rate=float(request.match_rate),
            )
        except Exception as e:
            logger.error(
                f"Failed to record job completion in DB for job {j_id}: {e}", exc_info=True
            )
            context.abort(grpc.StatusCode.INTERNAL, str(e))

        clear_job_progress(j_id)

        if request.webhook_log_json:
            try:
                webhook_log = json.loads(request.webhook_log_json)
                if isinstance(webhook_log, dict):
                    log_outbound_request(**webhook_log)
            except Exception as e:
                logger.error(f"Failed to log outbound webhook for job {j_id}: {e}")

        return pb2.JobCallbackResponse(status="ok")

    def FailJob(self, request, context):
        j_id = str(request.job_id)
        try:
            record_job_failure(
                job_id=j_id,
                error_message=request.error_message or "Job execution failed",
                duration_minutes=float(request.duration_minutes),
                is_cancelled=bool(request.is_cancelled),
            )
        except Exception as e:
            logger.error(f"Failed to record job failure in DB for job {j_id}: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

        clear_job_progress(j_id)
        return pb2.JobCallbackResponse(status="ok")


def start_backend_grpc_server(port: int | None = None, host: str = "0.0.0.0") -> grpc.Server:
    """Starts a BackendCallbackService gRPC server on `host:port` (pass port=0 for ephemeral)."""
    target_port = config.BACKEND_GRPC_PORT if port is None else port
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        interceptors=[GrpcContextInterceptor()],
        options=list(GRPC_CHANNEL_OPTIONS),
    )
    pb2_grpc.add_BackendCallbackServiceServicer_to_server(BackendCallbackServicer(), server)
    bind_addr = f"{host}:{target_port}"
    bound_port = server.add_insecure_port(bind_addr)
    server._bound_port = bound_port  # type: ignore[attr-defined]
    server.start()
    logger.info(f"[gRPC] Backend Callback gRPC Server listening on {host}:{bound_port}")
    return server


def ensure_backend_grpc_server() -> grpc.Server | None:
    """Starts the singleton BackendCallbackService gRPC server if not already running."""
    global _active_server
    with _server_lock:
        if _active_server is not None:
            return _active_server
        try:
            _active_server = start_backend_grpc_server()
            return _active_server
        except Exception as e:
            logger.warning(f"[gRPC] Could not bind Backend Callback gRPC Server: {e}")
            return None


def stop_backend_grpc_server(grace: float = 2.0) -> None:
    """Stops the singleton BackendCallbackService gRPC server if running."""
    global _active_server
    with _server_lock:
        if _active_server is not None:
            _active_server.stop(grace=grace)
            _active_server = None
