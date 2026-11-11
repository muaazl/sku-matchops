import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import Connection, text
from starlette.concurrency import run_in_threadpool

from backend.app.core import config as backend_config
from backend.app.core.security import Principal, get_principal
from backend.app.db.schema import jobs as jobs_table
from backend.app.db.session import get_db_connection, get_engine
from backend.app.grpc_server import (
    _job_progress,
    clear_job_progress,
    get_bulk_job_progress_states,
    get_job_progress_state,
    job_events_channel,
    publish_job_event,
)
from backend.app.schemas.models import BaseRequest, JobResponse, SKUItem
from backend.app.services.engine_client import cancel_engine_job
from backend.app.services.worker import enqueue_job

# The list never shows the uploaded SKUs (~15 KB a job); /jobs/{id} and retry still read them
_JOB_LIST_COLUMNS = ", ".join(c.name for c in jobs_table.columns if c.name != "input_skus_json")
_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

router = APIRouter()


def _enrich_job_dict(
    job_dict: dict, prefetched_progress: tuple[float, int | None] | None = None
) -> dict:
    """Enriches job dictionary with live Redis/in-memory progress_pct and rolling dynamic ETA."""
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
            else _job_progress.get(j_id, 0.0)
        )
        job_dict["eta_seconds"] = None
        return job_dict

    if status == "queued":
        job_dict["progress_pct"] = 0.0
        total = job_dict.get("total_items", 0)
        job_dict["eta_seconds"] = max(3, int(total * 0.035)) if total > 0 else 5
        return job_dict

    progress_pct, cached_eta = (
        prefetched_progress if prefetched_progress is not None else get_job_progress_state(j_id)
    )
    job_dict["progress_pct"] = progress_pct

    # Calculate dynamic rolling ETA for running jobs
    if status == "running" and cached_eta is not None:
        job_dict["eta_seconds"] = cached_eta
    elif status == "running" and job_dict.get("started_at"):
        try:
            started_str = str(job_dict["started_at"]).replace(" ", "T")
            started_at = datetime.fromisoformat(started_str)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            elapsed = max(0.1, (datetime.now(UTC) - started_at).total_seconds())

            if progress_pct > 0.5 and progress_pct < 99.9:
                total_est_seconds = elapsed / (progress_pct / 100.0)
                remaining_seconds = max(0, int(total_est_seconds - elapsed))
                job_dict["eta_seconds"] = remaining_seconds
            elif job_dict.get("total_items", 0) > 0:
                total = job_dict.get("total_items", 0)
                est_total = max(3, int(total * 0.035))
                job_dict["eta_seconds"] = max(3, int(est_total - elapsed))
        except Exception:
            job_dict["eta_seconds"] = None

    return job_dict


def _enrich_job_rows(rows) -> list[dict]:
    """Batch-enriches a sequence of job rows using at most 1 pipelined Redis call for running jobs."""
    dicts = [dict(r) for r in rows]
    running_ids = [str(d.get("id", "")) for d in dicts if d.get("status") == "running"]
    progress_map = get_bulk_job_progress_states(running_ids) if running_ids else {}
    return [_enrich_job_dict(d, progress_map.get(str(d.get("id", "")))) for d in dicts]


_TIMEFRAME_INTERVALS = {"24h": "24 hours", "7d": "7 days", "30d": "30 days"}


def _stats_filter(domain: str | None, timeframe: str | None) -> tuple[list[str], dict]:
    """WHERE clauses + params shared by the processed_skus dashboard aggregates."""
    clauses: list[str] = []
    params: dict = {}

    if domain and domain.lower() != "all":
        clauses.append("domain = :domain")
        params["domain"] = domain.lower()

    if timeframe in _TIMEFRAME_INTERVALS:
        clauses.append("created_at >= now() - CAST(:window AS interval)")
        params["window"] = _TIMEFRAME_INTERVALS[timeframe]

    return clauses, params


def _get_job(db: Connection, job_id: str):
    """The job row."""
    return (
        db.execute(
            text("SELECT * FROM jobs WHERE id = :id"),
            {"id": job_id},
        )
        .mappings()
        .fetchone()
    )


@router.get("/jobs", response_model=list[JobResponse])
def get_jobs(
    response: Response,
    status: str | None = None,
    type: str | None = None,
    created_by: str | None = None,
    domain: str | None = None,
    sheet_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    # Generous default backstop against loading the entire table on a very long-lived
    # install — well above any realistic job count for an admin tool, so it shouldn't
    # change behavior in practice. Pass limit=0 (as some callers already do for
    # /processed-skus) to explicitly request every row with no cap.
    limit: int | None = 10000,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    clauses: list[str] = []
    params: dict = {}

    if status:
        clauses.append("status = :status")
        params["status"] = status
    if type:
        clauses.append("type = :type")
        params["type"] = type
    if created_by:
        clauses.append("created_by = :created_by")
        params["created_by"] = created_by
    if date_from:
        clauses.append("started_at >= CAST(:date_from AS timestamptz)")
        params["date_from"] = date_from
    if date_to:
        clauses.append("started_at <= CAST(:date_to AS timestamptz)")
        params["date_to"] = date_to
    if domain:
        clauses.append("domain = :domain")
        params["domain"] = domain
    if sheet_name:
        clauses.append("(sheet_name ILIKE :sheet_name OR target_sheet ILIKE :sheet_name)")
        params["sheet_name"] = f"%{sheet_name}%"

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    total_count = db.execute(text(f"SELECT COUNT(*) FROM jobs {where}"), params).scalar()
    response.headers["X-Total-Count"] = str(total_count)

    query = f"SELECT {_JOB_LIST_COLUMNS} FROM jobs {where} ORDER BY started_at DESC"
    if limit is not None and limit > 0:
        query += " LIMIT :limit OFFSET :offset"
        params.update(limit=limit, offset=(page - 1) * limit)

    rows = db.execute(text(query), params).mappings().fetchall()
    return [JobResponse(**job_dict) for job_dict in _enrich_job_rows(rows)]


@router.get("/jobs/dashboard-stats")
def get_dashboard_stats(
    domain: str | None = "all",
    timeframe: str | None = "30d",
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    """
    Computes aggregate metrics from processed_skus for dashboard cards and charts.
    Fast indexed queries (< 10ms).
    """
    clauses, params = _stats_filter(domain, timeframe)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    # 1 & 2. Single-pass processed_skus summary + match-source breakdown (answered via covering index)
    summary_source_query = f"""
        SELECT
            CASE
                WHEN LOWER(TRIM(COALESCE(match_source, ''))) IN ('matcher', 'catalogue') THEN 'Matcher'
                WHEN LOWER(TRIM(COALESCE(match_source, ''))) = 'classifier' THEN 'Classifier'
                WHEN matched_catalog_name IS NOT NULL AND TRIM(matched_catalog_name) != '' THEN 'Matcher'
                WHEN (bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0) THEN 'Classifier'
                ELSE 'Matcher'
            END as normalized_source,
            COUNT(*) as cnt,
            COALESCE(SUM(confidence), 0.0) as sum_conf,
            COUNT(confidence) as cnt_conf,
            COALESCE(SUM(CASE WHEN confidence >= 0.85 THEN 1 ELSE 0 END), 0) as high_count,
            COALESCE(SUM(CASE WHEN confidence >= 0.60 AND confidence < 0.85 THEN 1 ELSE 0 END), 0) as med_count,
            COALESCE(SUM(CASE WHEN confidence < 0.60 THEN 1 ELSE 0 END), 0) as low_count
        FROM processed_skus
        {where_sql}
        GROUP BY normalized_source
        ORDER BY cnt DESC
    """
    agg_rows = db.execute(text(summary_source_query), params).fetchall()

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

    # 3. Domain Breakdown
    dom_query = """
        SELECT domain, COUNT(*) as count
        FROM processed_skus
        GROUP BY domain
    """
    dom_rows = db.execute(text(dom_query)).fetchall()
    domain_breakdown = [{"domain": r[0] or "Unknown", "count": r[1]} for r in dom_rows]

    # 4. Volume Trend (by hour or date)
    volume_trend = []
    is_24h = timeframe == "24h"
    if is_24h:
        trend_query = f"""
            SELECT to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:00') as hr, COUNT(*) as cnt, AVG(confidence) as avg_c
            FROM processed_skus
            {where_sql}
            GROUP BY hr
            ORDER BY hr ASC
        """
        trend_rows = db.execute(text(trend_query), params).fetchall()
        t_map = {r[0]: (r[1], r[2] or 0.0) for r in trend_rows}

        now = datetime.now(UTC)
        for i in range(23, -1, -1):
            h_dt = now - timedelta(hours=i)
            key = h_dt.strftime("%Y-%m-%d %H:00")
            label = h_dt.astimezone(ZoneInfo("Asia/Colombo")).strftime("%H:00")
            cnt, c_val = t_map.get(key, (0, 0.0))
            c_pct = round(c_val * 100, 1) if c_val <= 1.0 else round(c_val, 1)
            volume_trend.append({"label": label, "skus": cnt, "avgConfidence": c_pct})
    else:
        trend_query = f"""
            SELECT to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') as dt, COUNT(*) as cnt, AVG(confidence) as avg_c
            FROM processed_skus
            {where_sql}
            GROUP BY dt
            ORDER BY dt ASC
        """
        trend_rows = db.execute(text(trend_query), params).fetchall()
        t_map = {r[0]: (r[1], r[2] or 0.0) for r in trend_rows}

        days = 7 if timeframe == "7d" else 30
        now = datetime.now(UTC)
        for i in range(days - 1, -1, -1):
            d_dt = now - timedelta(days=i)
            key = d_dt.strftime("%Y-%m-%d")
            label = d_dt.astimezone(ZoneInfo("Asia/Colombo")).strftime("%b %d")
            cnt, c_val = t_map.get(key, (0, 0.0))
            c_pct = round(c_val * 100, 1) if c_val <= 1.0 else round(c_val, 1)
            volume_trend.append({"label": label, "skus": cnt, "avgConfidence": c_pct})

    # 5. Consolidated job counts in a single scan over jobs
    job_counts_row = db.execute(
        text(
            """
            SELECT
                COUNT(*) FILTER (WHERE status IN ('queued', 'running')) AS active_jobs,
                COUNT(*) FILTER (WHERE status = 'completed' AND started_at >= now() - interval '7 days') AS completed_7d,
                COUNT(*) FILTER (WHERE status = 'failed' AND started_at >= now() - interval '7 days') AS failed_7d
            FROM jobs
            """
        )
    ).fetchone()
    active_jobs = int(job_counts_row[0] or 0) if job_counts_row else 0
    completed_7d = int(job_counts_row[1] or 0) if job_counts_row else 0
    failed_7d = int(job_counts_row[2] or 0) if job_counts_row else 0
    requests_24h = 0
    try:
        requests_24h = int(
            db.execute(
                text(
                    "SELECT COUNT(*) FROM api_requests WHERE created_at >= now() - interval '24 hours'"
                )
            ).scalar()
            or 0
        )
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


@router.get("/jobs/tag-stats")
def get_tag_stats(
    domain: str | None = "all",
    timeframe: str | None = "30d",
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    """
    Computes Basic Type / Region-Category / tag-status breakdowns from processed_skus,
    for the tagging-strategy charts on the Dashboard and Guide pages.
    """
    clauses, params = _stats_filter(domain, timeframe)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    bt_clauses = clauses + ["bt IS NOT NULL", "bt != ''"]
    where_bt = f"WHERE {' AND '.join(bt_clauses)}"
    tag_clauses = clauses + ["COALESCE(NULLIF(region, ''), NULLIF(category, '')) IS NOT NULL"]
    where_tag = f"WHERE {' AND '.join(tag_clauses)}"

    # 1. Basic Type distribution (top 10)
    bt_query = f"""
        SELECT bt, COUNT(*) as count
        FROM processed_skus
        {where_bt}
        GROUP BY bt
        ORDER BY count DESC
        LIMIT 10
    """
    bt_rows = db.execute(text(bt_query), params).fetchall()
    basic_type_distribution = [{"basicType": r[0], "count": r[1]} for r in bt_rows]

    # 2. Region (food) / Category (market) distribution, top 10
    tag_query = f"""
        SELECT COALESCE(NULLIF(region, ''), NULLIF(category, '')) as tag, COUNT(*) as count
        FROM processed_skus
        {where_tag}
        GROUP BY tag
        ORDER BY count DESC
        LIMIT 10
    """
    tag_rows = db.execute(text(tag_query), params).fetchall()
    region_distribution = [{"tag": r[0], "count": r[1]} for r in tag_rows]

    # 3. Tag status breakdown (AUTO/REVIEW/LOW), derived from the stored per-tag
    # confidence columns using the same thresholds the live tagger applies
    # (engine/classification/tagger.py get_status()), so this can never drift
    # out of sync with the actual pipeline behavior.
    auto_t = backend_config.AUTO_THRESHOLD
    review_t = backend_config.REVIEW_THRESHOLD

    status_q = f"""
        SELECT
            COALESCE(SUM(CASE WHEN bt_confidence >= :auto_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN bt_confidence >= :review_t AND bt_confidence < :auto_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN bt_confidence < :review_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN gk_confidence >= :auto_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN gk_confidence >= :review_t AND gk_confidence < :auto_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN gk_confidence < :review_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN region_confidence >= :auto_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN region_confidence >= :review_t AND region_confidence < :auto_t THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN region_confidence < :review_t THEN 1 ELSE 0 END), 0)
        FROM processed_skus
        {where_sql}
    """
    s_row = db.execute(
        text(status_q), {"auto_t": auto_t, "review_t": review_t, **params}
    ).fetchone()
    vals = [int(v or 0) for v in (s_row or (0,) * 9)]

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


_async_redis_pool = None


def get_async_redis_pool():
    global _async_redis_pool
    if _async_redis_pool is None:
        import redis.asyncio as aioredis

        _async_redis_pool = aioredis.ConnectionPool.from_url(
            backend_config.REDIS_URL,
            max_connections=50,
            socket_connect_timeout=2.0,
            decode_responses=True,
        )
    return _async_redis_pool


async def close_async_redis_pool():
    global _async_redis_pool
    if _async_redis_pool is not None:
        try:
            await _async_redis_pool.disconnect()
        except Exception:
            pass
        _async_redis_pool = None


async def _open_job_pubsub():
    import redis.asyncio as aioredis

    pool = get_async_redis_pool()
    client = aioredis.Redis(connection_pool=pool)
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    try:
        await pubsub.subscribe(job_events_channel())
    except Exception as e:
        await pubsub.aclose()
        await client.aclose()
        raise HTTPException(status_code=503, detail=f"Job event stream unavailable: {e}")
    return client, pubsub


def _recent_jobs() -> list[dict]:
    with get_engine().connect() as conn:
        rows = (
            conn.execute(
                text(f"SELECT {_JOB_LIST_COLUMNS} FROM jobs ORDER BY started_at DESC LIMIT 25")
            )
            .mappings()
            .fetchall()
        )
    return [JobResponse(**d).model_dump(mode="json") for d in _enrich_job_rows(rows)]


def _single_job(job_id: str) -> dict | None:
    with get_engine().connect() as conn:
        row = _get_job(conn, job_id)
    if not row:
        return None
    return JobResponse(**_enrich_job_dict(dict(row))).model_dump(mode="json")


@router.get("/jobs/stream")
async def stream_jobs(request: Request, principal: Principal = Depends(get_principal)):
    """SSE stream of live job state changes."""
    client, pubsub = await _open_job_pubsub()

    async def _gen():
        try:
            initial = await run_in_threadpool(_recent_jobs)
            yield f"data: {json.dumps({'type': 'snapshot', 'jobs': initial})}\n\n"
            while not await request.is_disconnected():
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if msg and msg.get("data"):
                    yield f"data: {msg['data']}\n\n"
                else:
                    yield ": keepalive\n\n"
        finally:
            try:
                await pubsub.aclose()
            finally:
                await client.aclose()

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get("/jobs/{id}", response_model=JobResponse)
def get_job(
    id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    row = _get_job(db, id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = _enrich_job_dict(dict(row))
    return JobResponse(**job_dict)


@router.get("/jobs/{id}/stream")
async def stream_job(id: str, request: Request, principal: Principal = Depends(get_principal)):
    """SSE stream of live progress, stage, and terminal outcome for a single job."""
    if await run_in_threadpool(_single_job, id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    client, pubsub = await _open_job_pubsub()

    async def _gen():
        try:
            initial = await run_in_threadpool(_single_job, id)
            if initial is None:
                return
            yield f"data: {json.dumps(initial)}\n\n"
            if initial.get("status") in _TERMINAL_STATUSES:
                return
            while not await request.is_disconnected():
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if msg and msg.get("data"):
                    payload = json.loads(msg["data"])
                    if str(payload.get("id")) == str(id):
                        yield f"data: {msg['data']}\n\n"
                        if payload.get("status") in _TERMINAL_STATUSES:
                            return
                else:
                    yield ": keepalive\n\n"
        finally:
            try:
                await pubsub.aclose()
            finally:
                await client.aclose()

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/jobs/{id}/cancel")
def cancel_job(
    id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    res = db.execute(
        text(
            "UPDATE jobs SET status = 'cancelled', cancel_requested = 1, completed_at = now() "
            "WHERE id = :id AND status IN ('queued', 'running')"
        ),
        {"id": id},
    )
    if res.rowcount == 0:
        row = _get_job(db, id)
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] == "cancelled":
            return {"message": "Job is already cancelled"}
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job with status '{row['status']}'. Only 'queued' or 'running' jobs can be cancelled.",
        )

    db.execute(
        text("UPDATE batches SET status = 'cancelled', completed_at = now() WHERE id = :id"),
        {"id": id},
    )
    db.commit()
    clear_job_progress(id)
    publish_job_event(
        {
            "id": str(id),
            "status": "cancelled",
            "current_stage": "cancelled",
            "eta_seconds": None,
        },
    )

    # Notify ML Engine microservice
    cancel_engine_job(id)
    return {"message": "Cancellation completed"}


@router.post("/jobs/{id}/retry")
def retry_job(
    id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    row = _get_job(db, id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = dict(row)
    input_skus_json = job_dict.get("input_skus_json")
    if not input_skus_json:
        raise HTTPException(status_code=400, detail="Cannot retry a job with no stored inputs")

    try:
        skus_data = json.loads(input_skus_json)
        skus = [SKUItem(**item) for item in skus_data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse stored SKUs: {e}")

    request = BaseRequest(
        skus=skus,
        domain=job_dict.get("domain") or "market",
        callback_url="",
        sheet_name=job_dict.get("sheet_name"),
    )

    result = enqueue_job(request, task=job_dict.get("type") or "pipeline")
    return {"message": "Retry initiated", "new_job_id": result["job_id"]}
