"""Outbound gRPC client sending job progress, completion, and failure callbacks to the Gateway."""

import logging
import threading
from typing import Any

import grpc

from engine.core import config
from engine.grpc.grpc_utils import GRPC_CHANNEL_OPTIONS, grpc_metadata, result_to_pipeline_proto
from engine.pipelines.worker_pipeline import register_job_callbacks
from engine.protos import engine_service_pb2 as pb2
from engine.protos import engine_service_pb2_grpc as pb2_grpc

logger = logging.getLogger("matchops.engine.grpc_callback")

_stub_lock = threading.Lock()
_callback_channels: dict[str, grpc.Channel] = {}
_callback_stubs: dict[str, pb2_grpc.BackendCallbackServiceStub] = {}


def _get_backend_stub(target: str | None = None) -> pb2_grpc.BackendCallbackServiceStub:
    """Returns a cached gRPC stub for the Gateway's BackendCallbackService."""
    addr = target or config.BACKEND_GRPC_URL
    with _stub_lock:
        stub = _callback_stubs.get(addr)
        if stub is None:
            channel = grpc.insecure_channel(addr, options=list(GRPC_CHANNEL_OPTIONS))
            stub = pb2_grpc.BackendCallbackServiceStub(channel)
            _callback_channels[addr] = channel
            _callback_stubs[addr] = stub
        return stub


def report_job_progress(
    backend_grpc_url: str,
    job_id: str,
    stage: str,
    pct: float,
    eta: int | None,
) -> None:
    """Emits an intermediate job progress event to the Backend Gateway via gRPC."""
    try:
        stub = _get_backend_stub(backend_grpc_url)
        stub.ReportJobProgress(
            pb2.JobProgressRequest(
                job_id=job_id,
                current_stage=stage,
                progress_pct=float(pct),
                eta_seconds=int(eta or 0),
                has_eta=eta is not None,
            ),
            metadata=grpc_metadata(),
            timeout=2.0,
        )
    except Exception as e:
        logger.debug(f"Failed to report progress via gRPC: {e}")


def complete_job(
    backend_grpc_url: str,
    job_id: str,
    domain: str,
    task: str,
    rule_version: str,
    duration_minutes: float,
    high_conf: int,
    med_conf: int,
    low_conf: int,
    match_rate: float,
    webhook_log_json: str,
    results: list[dict[str, Any]],
) -> None:
    """Delivers final typed job results and metrics to the Backend Gateway via gRPC."""
    stub = _get_backend_stub(backend_grpc_url)
    stub.CompleteJob(
        pb2.JobCompleteRequest(
            job_id=job_id,
            status="completed",
            domain=domain,
            task=task,
            rule_version=rule_version,
            duration_minutes=float(duration_minutes),
            high_conf=high_conf,
            med_conf=med_conf,
            low_conf=low_conf,
            match_rate=float(match_rate),
            webhook_log_json=webhook_log_json,
            results=[result_to_pipeline_proto(r) for r in results],
        ),
        metadata=grpc_metadata(),
        timeout=30.0,
    )


def fail_job(backend_grpc_url: str, job_id: str, error_message: str) -> None:
    """Reports terminal job failure to the Backend Gateway via gRPC."""
    try:
        stub = _get_backend_stub(backend_grpc_url)
        stub.FailJob(
            pb2.JobFailRequest(
                job_id=job_id,
                status="failed",
                error_message=error_message,
                duration_minutes=0.0,
                is_cancelled=False,
            ),
            metadata=grpc_metadata(),
            timeout=15.0,
        )
    except Exception as be_err:
        logger.error(f"[ENGINE WORKER] Failed to notify Gateway of job failure: {be_err}")


def register_engine_callbacks() -> None:
    """Binds gRPC callback handlers into worker_pipeline without creating an upward dependency."""
    register_job_callbacks(
        on_progress=report_job_progress,
        on_complete=complete_job,
        on_fail=fail_job,
    )


register_engine_callbacks()
