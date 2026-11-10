"""
gRPC Server implementation for SKU MatchOps Backend Gateway.
Receives progress, completion, and failure callbacks from ML Engine workers over gRPC
(port 50052 by default) and persists terminal job outcomes to PostgreSQL.
"""

import json
import logging
import threading
import time
from concurrent import futures
from typing import Any

import grpc
from sqlalchemy import text

from backend.app.core import config
from backend.app.core.grpc_utils import (
    GRPC_CHANNEL_OPTIONS,
    GrpcContextInterceptor,
    pipeline_proto_to_result,
)
from backend.app.db.session import get_engine, log_outbound_request
from backend.app.protos import engine_service_pb2 as pb2
from backend.app.protos import engine_service_pb2_grpc as pb2_grpc
from backend.app.services.persistence import record_job_completion, record_job_failure

logger = logging.getLogger("matchops.backend.grpc")

JOB_PROGRESS_PREFIX = "matchops:job:progress:"
JOB_PROGRESS_TTL = 3600

# In-memory mirror for degraded single-node fallback when Redis is unreachable
_job_progress: dict[str, float] = {}
_job_eta: dict[str, int | None] = {}
_job_stage: dict[str, str] = {}
_job_tenant: dict[str, str] = {}

_redis_lock = threading.Lock()
_redis_client = None
_redis_target: str | None = None
_redis_down_until: float = 0.0

_server_lock = threading.Lock()
_active_server: grpc.Server | None = None


def job_events_channel(tenant_id: str) -> str:
    return f"matchops:jobs:events:{tenant_id or 'default'}"


def _mark_redis_down() -> None:
    global _redis_down_until
    _redis_down_until = time.monotonic() + 5.0


def _progress_redis():
    """Shared Redis client with short timeouts and a 5s circuit breaker on outage."""
    global _redis_client, _redis_target
    if time.monotonic() < _redis_down_until:
        return None
    url = config.REDIS_URL
    with _redis_lock:
        if _redis_client is None or _redis_target != url:
            try:
                import redis

                _redis_client = redis.Redis.from_url(
                    url, socket_connect_timeout=0.5, socket_timeout=0.5, decode_responses=True
                )
                _redis_target = url
            except Exception:
                _mark_redis_down()
                return None
        return _redis_client


def set_job_progress(
    job_id: str,
    pct: float,
    eta: int | None = None,
    stage: str | None = None,
    tenant_id: str | None = None,
) -> None:
    jid = str(job_id)
    _job_progress[jid] = float(pct)
    _job_eta[jid] = eta
    if stage is not None:
        _job_stage[jid] = stage
    if tenant_id:
        _job_tenant[jid] = tenant_id

    r = _progress_redis()
    if r is None:
        return
    fields: dict[str, str] = {"pct": str(float(pct)), "eta": "" if eta is None else str(int(eta))}
    if stage is not None:
        fields["stage"] = stage
    if tenant_id:
        fields["tenant_id"] = tenant_id
    try:
        key = f"{JOB_PROGRESS_PREFIX}{jid}"
        pipe = r.pipeline()
        pipe.hset(key, mapping=fields)
        pipe.expire(key, JOB_PROGRESS_TTL)
        pipe.execute()
    except Exception as e:
        logger.debug(f"Redis progress write skipped for job {jid}: {e}")
        _mark_redis_down()


def get_job_progress_state(job_id: str) -> tuple[float, int | None]:
    """Reads live progress & ETA from Redis (multi-replica safe), falling back to local memory."""
    jid = str(job_id)
    r = _progress_redis()
    if r is not None:
        try:
            pct_raw, eta_raw = r.hmget(f"{JOB_PROGRESS_PREFIX}{jid}", "pct", "eta")
            if pct_raw is not None:
                eta_val = int(eta_raw) if eta_raw not in (None, "") else None
                return float(pct_raw), eta_val
        except Exception:
            _mark_redis_down()
    return _job_progress.get(jid, 0.0), _job_eta.get(jid)


def get_bulk_job_progress_states(job_ids: list[str]) -> dict[str, tuple[float, int | None]]:
    """Batch-fetches live progress & ETA for multiple running jobs in a single Redis pipeline."""
    if not job_ids:
        return {}
    jids = [str(j) for j in job_ids]
    out: dict[str, tuple[float, int | None]] = {}
    r = _progress_redis()
    if r is not None:
        try:
            pipe = r.pipeline()
            for jid in jids:
                pipe.hmget(f"{JOB_PROGRESS_PREFIX}{jid}", "pct", "eta")
            raw_list = pipe.execute()
            for jid, res in zip(jids, raw_list):
                if res and res[0] is not None:
                    eta_val = int(res[1]) if res[1] not in (None, "") else None
                    out[jid] = (float(res[0]), eta_val)
        except Exception:
            _mark_redis_down()
    for jid in jids:
        if jid not in out:
            out[jid] = (_job_progress.get(jid, 0.0), _job_eta.get(jid))
    return out


def publish_job_event(tenant_id: str | None, event: dict[str, Any]) -> None:
    """Broadcasts a live job state update on the tenant's Redis Pub/Sub channel."""
    r = _progress_redis()
    if r is None:
        return
    try:
        r.publish(job_events_channel(tenant_id or "default"), json.dumps(event, default=str))
    except Exception as e:
        logger.debug(f"Redis event publish skipped for job {event.get('id')}: {e}")
        _mark_redis_down()


def _resolve_job_tenant(job_id: str) -> str:
    jid = str(job_id)
    cached = _job_tenant.get(jid)
    if cached:
        return cached
    r = _progress_redis()
    if r is not None:
        try:
            val = r.hget(f"{JOB_PROGRESS_PREFIX}{jid}", "tenant_id")
            if val:
                _job_tenant[jid] = str(val)
                return _job_tenant[jid]
        except Exception:
            _mark_redis_down()
    try:
        with get_engine().connect() as conn:
            row = conn.execute(
                text("SELECT tenant_id FROM jobs WHERE id = :id"), {"id": jid}
            ).fetchone()
            if row and row[0]:
                _job_tenant[jid] = str(row[0])
                return _job_tenant[jid]
    except Exception:
        pass
    return "default"


def clear_job_progress(job_id: str) -> None:
    """Evicts terminal job progress/ETA/stage entries from Redis and local memory."""
    jid = str(job_id)
    _job_progress.pop(jid, None)
    _job_eta.pop(jid, None)
    _job_stage.pop(jid, None)
    _job_tenant.pop(jid, None)
    r = _progress_redis()
    if r is not None:
        try:
            r.delete(f"{JOB_PROGRESS_PREFIX}{jid}")
        except Exception:
            _mark_redis_down()


class BackendCallbackServicer(pb2_grpc.BackendCallbackServiceServicer):
    def ReportJobProgress(self, request, context):
        j_id = str(request.job_id)
        pct = float(request.progress_pct)
        eta = int(request.eta_seconds) if request.has_eta else None

        last_stage = _job_stage.get(j_id)
        if last_stage is None:
            r = _progress_redis()
            if r is not None:
                try:
                    last_stage = r.hget(f"{JOB_PROGRESS_PREFIX}{j_id}", "stage")
                except Exception:
                    _mark_redis_down()

        tenant_id = _job_tenant.get(j_id)
        # Only perform a DB write when the stage changes or when first observed
        if last_stage != request.current_stage or _job_stage.get(j_id) is None:
            try:
                with get_engine().begin() as conn:
                    # Never resurrect a job that already reached a terminal state
                    row = conn.execute(
                        text(
                            "UPDATE jobs SET status = 'running', current_stage = :stage, updated_at = now() "
                            "WHERE id = :id AND status IN ('queued', 'running') RETURNING tenant_id"
                        ),
                        {"stage": request.current_stage, "id": j_id},
                    ).fetchone()
                    if row is None:
                        clear_job_progress(j_id)
                        return pb2.JobCallbackResponse(status="ok")
                    tenant_id = str(row[0] or "default")
            except Exception as e:
                logger.error(f"Failed to update progress for job {j_id} in DB: {e}")

        if not tenant_id:
            tenant_id = _resolve_job_tenant(j_id)

        set_job_progress(j_id, pct=pct, eta=eta, stage=request.current_stage, tenant_id=tenant_id)
        publish_job_event(
            tenant_id,
            {
                "id": j_id,
                "status": "running",
                "current_stage": request.current_stage,
                "progress_pct": pct,
                "eta_seconds": eta,
            },
        )
        return pb2.JobCallbackResponse(status="ok")

    def CompleteJob(self, request, context):
        j_id = str(request.job_id)
        tenant_id = _resolve_job_tenant(j_id)
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
        publish_job_event(
            tenant_id,
            {
                "id": j_id,
                "status": "completed",
                "current_stage": "done",
                "progress_pct": 100.0,
                "eta_seconds": 0,
                "high_conf": int(request.high_conf),
                "med_conf": int(request.med_conf),
                "low_conf": int(request.low_conf),
                "match_rate": float(request.match_rate),
                "duration_minutes": float(request.duration_minutes),
            },
        )

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
        tenant_id = _resolve_job_tenant(j_id)
        is_cancelled = bool(request.is_cancelled)
        err_msg = request.error_message or "Job execution failed"
        try:
            record_job_failure(
                job_id=j_id,
                error_message=err_msg,
                duration_minutes=float(request.duration_minutes),
                is_cancelled=is_cancelled,
            )
        except Exception as e:
            logger.error(f"Failed to record job failure in DB for job {j_id}: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

        clear_job_progress(j_id)
        publish_job_event(
            tenant_id,
            {
                "id": j_id,
                "status": "cancelled" if is_cancelled else "failed",
                "current_stage": "cancelled" if is_cancelled else "failed",
                "error_message": err_msg,
                "duration_minutes": float(request.duration_minutes),
                "eta_seconds": None,
            },
        )
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
