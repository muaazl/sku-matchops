"""Job lifecycle orchestration, Redis-first dispatch, enrichment, SSE snapshots, and dashboard stats."""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from backend.core import config
from backend.core.exceptions import (
    ResourceConflictError,
    ResourceNotFoundError,
    ServiceUnavailableError,
    ServiceValidationError,
)
from backend.core.observability import get_trace_id
from backend.repositories import (
    audit_repository,
    job_progress_repository,
    job_repository,
    sku_repository,
    task_queue_repository,
)
from backend.schemas.job_schema import JobResponse
from backend.schemas.sku_schema import BaseRequest, SKUItem
from backend.services import engine_gateway_service

logger = logging.getLogger("matchops.services.job")

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def try_enqueue_redis_task(
    job_id: str, payload: dict[str, Any], label: str
) -> dict[str, Any] | None:
    """Attempts to enqueue a task payload onto the Redis queue. Returns None on fallback."""
    try:
        queue = task_queue_repository.get_task_queue()
        if queue is not None:
            task_id = queue.enqueue_task(payload)
            return {"status": "queued", "job_id": str(job_id), "task_id": task_id}
    except Exception as e:
        logger.warning(
            f"Redis queue unavailable ({e}); falling back to gRPC dispatch for {label} {job_id}."
        )
    return None


def dispatch_batch_payload(job_id: str, request: BaseRequest, task: str) -> dict[str, Any]:
    """Dispatches a batch job via Redis queue, falling back to direct Engine gRPC."""
    skus_payload = [
        {
            "name": sku.name,
            "price": sku.price or 0.0,
            "description": sku.description or "",
            "category": sku.category or "",
        }
        for sku in request.skus
    ]
    payload = {
        "job_id": str(job_id),
        "task": task,
        "domain": request.domain or "market",
        "skus": skus_payload,
        "backend_grpc_url": config.BACKEND_GRPC_URL,
        "callback_url": request.callback_url,
        "sheet_name": request.sheet_name,
        "spreadsheet_id": request.spreadsheet_id,
        "trace_id": get_trace_id(),
    }
    queued = try_enqueue_redis_task(job_id, payload, "job")
    if queued is not None:
        return queued
    return engine_gateway_service.dispatch_batch_job(job_id=job_id, request=request, task=task)


def register_and_dispatch_job(
    *,
    job_type: str,
    domain: str,
    total_items: int,
    input_skus_json: str,
    sheet_name: str | None,
    target_sheet: str | None,
    dispatch_fn: Callable[[str], object],
) -> dict[str, Any]:
    """Allocates a sequential job row, tracks initial progress, and dispatches work."""
    job_id = job_repository.insert_queued_job_with_retry(
        job_type=job_type,
        domain=domain,
        total_items=total_items,
        input_skus_json=input_skus_json,
        sheet_name=sheet_name,
        target_sheet=target_sheet,
    )
    initial_eta = max(3, int(total_items * 0.04))
    job_progress_repository.set_job_progress(job_id, pct=0.0, eta=initial_eta, stage="queued")
    job_progress_repository.publish_job_event(
        {
            "id": job_id,
            "status": "queued",
            "current_stage": "queued",
            "progress_pct": 0.0,
            "eta_seconds": initial_eta,
        }
    )
    logger.info(f"[JOB {job_id}] Queued {total_items} SKUs (task={job_type}, domain={domain})")

    try:
        dispatch_fn(job_id)
    except Exception as dispatch_err:
        job_progress_repository.clear_job_progress(job_id)
        logger.error(f"[JOB {job_id}] Dispatch to ML Engine failed: {dispatch_err}")
        try:
            job_repository.mark_job_dispatch_failed(job_id, str(dispatch_err))
        except Exception as db_err:
            logger.error(f"[JOB {job_id}] Failed to record failure state in database: {db_err}")
        job_progress_repository.publish_job_event(
            {
                "id": job_id,
                "status": "failed",
                "current_stage": "failed",
                "error_message": str(dispatch_err),
                "eta_seconds": None,
            }
        )
        raise RuntimeError(f"Failed to dispatch job to ML Engine: {dispatch_err}")

    return {"job_id": job_id, "status": "queued", "total_skus": total_items}


def enqueue_job(request: BaseRequest, task: str) -> dict[str, Any]:
    """Validates and registers a matcher/classifier/pipeline job and dispatches it."""
    if not request.skus:
        raise ValueError("'skus' list is empty.")

    return register_and_dispatch_job(
        job_type=task,
        domain=request.domain or "market",
        total_items=len(request.skus),
        input_skus_json=json.dumps([sku.model_dump() for sku in request.skus]),
        sheet_name=request.sheet_name,
        target_sheet=request.sheet_name or "N/A",
        dispatch_fn=lambda job_id: dispatch_batch_payload(
            job_id=job_id, request=request, task=task
        ),
    )


def enrich_job_record(
    job_dict: dict[str, Any],
    prefetched_progress: tuple[float, int | None] | None = None,
) -> dict[str, Any]:
    """Enriches a job dictionary with live Redis/memory progress_pct and rolling dynamic ETA."""
    j_id = str(job_dict.get("id", ""))
    status = job_dict.get("status")

    if status == "completed":
        job_dict["progress_pct"] = 100.0
        job_dict["eta_seconds"] = 0
        return job_dict

    if status in ("failed", "cancelled"):
        job_dict["progress_pct"] = (
            prefetched_progress[0]
            if prefetched_progress is not None
            else job_progress_repository.get_in_memory_progress(j_id)
        )
        job_dict["eta_seconds"] = None
        return job_dict

    if status == "queued":
        job_dict["progress_pct"] = 0.0
        total = job_dict.get("total_items", 0)
        job_dict["eta_seconds"] = max(3, int(total * 0.035)) if total > 0 else 5
        return job_dict

    progress_pct, cached_eta = (
        prefetched_progress
        if prefetched_progress is not None
        else job_progress_repository.fetch_job_progress_state(j_id)
    )
    job_dict["progress_pct"] = progress_pct

    if status == "running" and cached_eta is not None:
        job_dict["eta_seconds"] = cached_eta
    elif status == "running" and job_dict.get("started_at"):
        try:
            started_str = str(job_dict["started_at"]).replace(" ", "T")
            started_at = datetime.fromisoformat(started_str)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            elapsed = max(0.1, (datetime.now(UTC) - started_at).total_seconds())

            if 0.5 < progress_pct < 99.9:
                total_est_seconds = elapsed / (progress_pct / 100.0)
                job_dict["eta_seconds"] = max(0, int(total_est_seconds - elapsed))
            elif job_dict.get("total_items", 0) > 0:
                total = job_dict.get("total_items", 0)
                est_total = max(3, int(total * 0.035))
                job_dict["eta_seconds"] = max(3, int(est_total - elapsed))
        except Exception:
            job_dict["eta_seconds"] = None

    return job_dict


def enrich_job_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch-enriches job rows using at most one pipelined Redis call for running jobs."""
    dicts = [dict(r) for r in rows]
    running_ids = [str(d.get("id", "")) for d in dicts if d.get("status") == "running"]
    progress_map = (
        job_progress_repository.fetch_bulk_job_progress_states(running_ids) if running_ids else {}
    )
    return [enrich_job_record(d, progress_map.get(str(d.get("id", "")))) for d in dicts]


def list_jobs(
    *,
    status: str | None = None,
    job_type: str | None = None,
    created_by: str | None = None,
    domain: str | None = None,
    sheet_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    limit: int | None = 10000,
) -> tuple[int, list[JobResponse]]:
    """Lists filtered jobs enriched with live progress and returns `(total_count, items)`."""
    total_count, rows = job_repository.count_and_fetch_jobs(
        status=status,
        job_type=job_type,
        created_by=created_by,
        domain=domain,
        sheet_name=sheet_name,
        date_from=date_from,
        date_to=date_to,
        page=page,
        limit=limit,
    )
    return total_count, [JobResponse(**d) for d in enrich_job_rows(rows)]


def fetch_job_detail(job_id: str) -> JobResponse:
    """Fetches a single enriched job or raises ResourceNotFoundError."""
    row = job_repository.fetch_job_by_id(job_id)
    if not row:
        raise ResourceNotFoundError("Job not found")
    return JobResponse(**enrich_job_record(row))


def fetch_recent_job_snapshots(limit: int = 25) -> list[dict[str, Any]]:
    """Returns JSON-serializable recent job snapshots for SSE stream initialization."""
    rows = job_repository.fetch_recent_jobs(limit=limit)
    return [JobResponse(**d).model_dump(mode="json") for d in enrich_job_rows(rows)]


def fetch_single_job_snapshot(job_id: str) -> dict[str, Any] | None:
    """Returns a single JSON-serializable job snapshot or None if missing."""
    row = job_repository.fetch_job_by_id(job_id)
    if not row:
        return None
    return JobResponse(**enrich_job_record(row)).model_dump(mode="json")


async def open_job_event_stream():
    """Opens the async Redis Pub/Sub client for job SSE streaming."""
    try:
        return await job_progress_repository.open_job_pubsub()
    except Exception as e:
        raise ServiceUnavailableError(str(e)) from e


def cancel_job(job_id: str) -> dict[str, str]:
    """Atomically cancels a queued/running job and signals Redis + Engine cancellation."""
    cancelled, existing = job_repository.cancel_active_job(job_id)
    if not cancelled:
        if not existing:
            raise ResourceNotFoundError("Job not found")
        if existing["status"] == "cancelled":
            return {"message": "Job is already cancelled"}
        raise ResourceConflictError(
            f"Cannot cancel job with status '{existing['status']}'. Only 'queued' or 'running' jobs can be cancelled."
        )

    job_progress_repository.clear_job_progress(job_id)
    job_progress_repository.publish_job_event(
        {
            "id": str(job_id),
            "status": "cancelled",
            "current_stage": "cancelled",
            "eta_seconds": None,
        }
    )

    try:
        queue = task_queue_repository.get_task_queue()
        if queue is not None:
            queue.request_cancel(str(job_id))
    except Exception as e:
        logger.warning(f"Could not set Redis cancellation flag for job {job_id}: {e}")

    engine_gateway_service.cancel_engine_job(job_id)
    return {"message": "Cancellation completed"}


def retry_job(job_id: str) -> dict[str, Any]:
    """Re-enqueues a stored job using its saved `input_skus_json` payload."""
    row = job_repository.fetch_job_by_id(job_id)
    if not row:
        raise ResourceNotFoundError("Job not found")

    input_skus_json = row.get("input_skus_json")
    if not input_skus_json:
        raise ServiceValidationError("Cannot retry a job with no stored inputs")

    try:
        skus_data = json.loads(input_skus_json)
        skus = [SKUItem(**item) for item in skus_data]
    except Exception as e:
        raise ServiceValidationError(f"Failed to parse stored SKUs: {e}") from e

    request = BaseRequest(
        skus=skus,
        domain=row.get("domain") or "market",
        callback_url="",
        sheet_name=row.get("sheet_name"),
    )
    result = enqueue_job(request, task=row.get("type") or "pipeline")
    return {"message": "Retry initiated", "new_job_id": result["job_id"]}


def compute_dashboard_stats(
    domain: str | None = "all", timeframe: str | None = "30d"
) -> dict[str, Any]:
    """Aggregates KPI cards, confidence buckets, match-source breakdown, and volume trends."""
    agg_rows = sku_repository.aggregate_sku_source_metrics(domain, timeframe)

    total_skus = sum(int(r[1] or 0) for r in agg_rows)
    total_sum_conf = sum(float(r[2] or 0.0) for r in agg_rows)
    total_cnt_conf = sum(int(r[3] or 0) for r in agg_rows)
    raw_avg = (total_sum_conf / total_cnt_conf) if total_cnt_conf > 0 else 0.0
    avg_conf_pct = round(raw_avg * 100, 1) if raw_avg <= 1.0 else round(raw_avg, 1)
    high_count = sum(int(r[4] or 0) for r in agg_rows)
    med_count = sum(int(r[5] or 0) for r in agg_rows)
    low_count = sum(int(r[6] or 0) for r in agg_rows)

    high_pct = round((high_count / max(1, total_skus)) * 100, 1)
    med_pct = round((med_count / max(1, total_skus)) * 100, 1)
    low_pct = round((low_count / max(1, total_skus)) * 100, 1)

    match_sources = [
        {
            "source": r[0],
            "count": r[1],
            "pct": round((r[1] / max(1, total_skus)) * 100, 1),
        }
        for r in agg_rows
    ]

    dom_rows = sku_repository.aggregate_sku_domain_counts()
    domain_breakdown = [{"domain": r[0] or "Unknown", "count": r[1]} for r in dom_rows]

    volume_trend: list[dict[str, Any]] = []
    is_24h = timeframe == "24h"
    trend_rows = sku_repository.aggregate_sku_volume_trend(domain, timeframe, hourly=is_24h)
    t_map = {r[0]: (r[1], r[2] or 0.0) for r in trend_rows}
    now = datetime.now(UTC)

    if is_24h:
        for i in range(23, -1, -1):
            h_dt = now - timedelta(hours=i)
            key = h_dt.strftime("%Y-%m-%d %H:00")
            label = h_dt.astimezone(ZoneInfo("Asia/Colombo")).strftime("%H:00")
            cnt, c_val = t_map.get(key, (0, 0.0))
            c_pct = round(c_val * 100, 1) if c_val <= 1.0 else round(c_val, 1)
            volume_trend.append({"label": label, "skus": cnt, "avgConfidence": c_pct})
    else:
        days = 7 if timeframe == "7d" else 30
        for i in range(days - 1, -1, -1):
            d_dt = now - timedelta(days=i)
            key = d_dt.strftime("%Y-%m-%d")
            label = d_dt.astimezone(ZoneInfo("Asia/Colombo")).strftime("%b %d")
            cnt, c_val = t_map.get(key, (0, 0.0))
            c_pct = round(c_val * 100, 1) if c_val <= 1.0 else round(c_val, 1)
            volume_trend.append({"label": label, "skus": cnt, "avgConfidence": c_pct})

    active_jobs, completed_7d, failed_7d = job_repository.aggregate_job_status_counts()
    try:
        requests_24h = audit_repository.count_api_requests_last_24h()
    except Exception:
        requests_24h = 0

    return {
        "stats": {
            "totalProcessedSkus": total_skus,
            "avgConfidencePct": avg_conf_pct,
            "highConfidenceCount": high_count,
            "highConfidencePct": high_pct,
            "mediumConfidenceCount": med_count,
            "mediumConfidencePct": med_pct,
            "lowConfidenceCount": low_count,
            "lowConfidencePct": low_pct,
            "activeJobs": active_jobs,
            "jobsCompleted7d": completed_7d,
            "jobsFailed7d": failed_7d,
            "totalRequests24h": requests_24h,
        },
        "confidenceDistribution": [
            {
                "tier": "High Confidence (≥85%)",
                "count": high_count,
                "pct": high_pct,
                "color": "#10b981",
            },
            {
                "tier": "Medium Confidence (60-84%)",
                "count": med_count,
                "pct": med_pct,
                "color": "#f59e0b",
            },
            {
                "tier": "Escalated / Low (<60%)",
                "count": low_count,
                "pct": low_pct,
                "color": "#ef4444",
            },
        ],
        "matchSourceDistribution": match_sources,
        "domainBreakdown": domain_breakdown,
        "volumeTrend": volume_trend,
    }


def compute_tag_stats(domain: str | None = "all", timeframe: str | None = "30d") -> dict[str, Any]:
    """Computes BasicType, Region/Category, and confidence-tier tag statistics."""
    bt_rows = sku_repository.aggregate_top_basic_types(domain, timeframe, limit=10)
    basic_type_distribution = [{"basicType": r[0], "count": r[1]} for r in bt_rows]

    tag_rows = sku_repository.aggregate_top_region_or_category(domain, timeframe, limit=10)
    region_distribution = [{"tag": r[0], "count": r[1]} for r in tag_rows]

    auto_t = config.AUTO_THRESHOLD
    review_t = config.REVIEW_THRESHOLD
    vals = sku_repository.aggregate_tag_confidence_tiers(domain, timeframe, auto_t, review_t)
    tag_status_breakdown = [
        {"tagType": "Basic Type", "auto": vals[0], "review": vals[1], "low": vals[2]},
        {"tagType": "Generic Keyword", "auto": vals[3], "review": vals[4], "low": vals[5]},
        {"tagType": "Region / Category", "auto": vals[6], "review": vals[7], "low": vals[8]},
    ]
    return {
        "basicTypeDistribution": basic_type_distribution,
        "regionDistribution": region_distribution,
        "tagStatusBreakdown": tag_status_breakdown,
        "thresholds": {"auto": auto_t, "review": review_t},
    }
