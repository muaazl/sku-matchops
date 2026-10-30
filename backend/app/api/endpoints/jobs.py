import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import Connection, text

from backend.app.api.endpoints.engine_callbacks import _job_progress
from backend.app.core.db import get_db_connection
from backend.app.core.security import Principal, get_principal
from backend.app.schemas.models import BaseRequest, JobResponse, SKUItem
from backend.app.services.engine_client import cancel_engine_job
from backend.app.services.worker import enqueue_job
from engine import config as engine_config
from engine.core.models import jobs as jobs_table

# The list never shows the uploaded SKUs (~15 KB a job); /jobs/{id} and retry still read them
_JOB_LIST_COLUMNS = ", ".join(c.name for c in jobs_table.columns if c.name != "input_skus_json")

router = APIRouter()


def _enrich_job_dict(job_dict: dict) -> dict:
    """Enriches job dictionary with live in-memory progress_pct and rolling dynamic ETA."""
    j_id = str(job_dict.get("id", ""))
    status = job_dict.get("status")

    if status == "completed":
        job_dict["progress_pct"] = 100.0
        job_dict["eta_seconds"] = 0
        return job_dict
    elif status in ("failed", "cancelled"):
        job_dict["progress_pct"] = _job_progress.get(j_id, 0.0)
        job_dict["eta_seconds"] = None
        return job_dict

    progress_pct = _job_progress.get(j_id, 0.0)
    job_dict["progress_pct"] = progress_pct

    # Calculate dynamic rolling ETA for running/queued jobs
    if status == "running" and job_dict.get("started_at"):
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
    elif status == "queued":
        job_dict["progress_pct"] = 0.0
        total = job_dict.get("total_items", 0)
        job_dict["eta_seconds"] = max(3, int(total * 0.035)) if total > 0 else 5

    return job_dict


_TIMEFRAME_INTERVALS = {"24h": "24 hours", "7d": "7 days", "30d": "30 days"}


def _stats_filter(domain: str | None, timeframe: str | None, tenant_id: str) -> tuple[str, dict]:
    """WHERE clause + params shared by the processed_skus dashboard aggregates."""
    where_clauses = ["tenant_id = :tenant_id"]
    params = {"tenant_id": tenant_id}

    if domain and domain.lower() != "all":
        where_clauses.append("domain = :domain")
        params["domain"] = domain.lower()

    if timeframe in _TIMEFRAME_INTERVALS:
        where_clauses.append("created_at >= now() - CAST(:window AS interval)")
        params["window"] = _TIMEFRAME_INTERVALS[timeframe]

    return " AND ".join(where_clauses), params


def _tenant_job(db: Connection, job_id: str, principal: Principal):
    """The job row if it belongs to the caller's tenant (other tenants' jobs look like 404s)."""
    return (
        db.execute(
            text("SELECT * FROM jobs WHERE id = :id AND tenant_id = :tenant_id"),
            {"id": job_id, "tenant_id": principal.tenant_id},
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
    where = "WHERE tenant_id = :tenant_id"
    params = {"tenant_id": principal.tenant_id}

    if status:
        where += " AND status = :status"
        params["status"] = status
    if type:
        where += " AND type = :type"
        params["type"] = type
    if created_by:
        where += " AND created_by = :created_by"
        params["created_by"] = created_by
    if date_from:
        where += " AND started_at >= CAST(:date_from AS timestamptz)"
        params["date_from"] = date_from
    if date_to:
        where += " AND started_at <= CAST(:date_to AS timestamptz)"
        params["date_to"] = date_to
    if domain:
        where += " AND domain = :domain"
        params["domain"] = domain
    if sheet_name:
        where += " AND (sheet_name LIKE :sheet_name OR target_sheet LIKE :sheet_name)"
        params["sheet_name"] = f"%{sheet_name}%"

    total_count = db.execute(text(f"SELECT COUNT(*) FROM jobs {where}"), params).scalar()
    response.headers["X-Total-Count"] = str(total_count)

    query = f"SELECT {_JOB_LIST_COLUMNS} FROM jobs {where} ORDER BY started_at DESC"
    if limit is not None and limit > 0:
        query += " LIMIT :limit OFFSET :offset"
        params.update(limit=limit, offset=(page - 1) * limit)

    rows = db.execute(text(query), params).mappings().fetchall()

    jobs = []
    for row in rows:
        job_dict = _enrich_job_dict(dict(row))
        jobs.append(JobResponse(**job_dict))
    return jobs


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
    where_str, params = _stats_filter(domain, timeframe, principal.tenant_id)
    tenant = {"tenant_id": principal.tenant_id}

    # 1. Total processed and confidence breakdown
    stat_query = f"""
        SELECT
            COUNT(*) as total_skus,
            AVG(confidence) as avg_conf,
            SUM(CASE WHEN confidence >= 0.85 THEN 1 ELSE 0 END) as high_count,
            SUM(CASE WHEN confidence >= 0.60 AND confidence < 0.85 THEN 1 ELSE 0 END) as med_count,
            SUM(CASE WHEN confidence < 0.60 THEN 1 ELSE 0 END) as low_count
        FROM processed_skus
        WHERE {where_str}
    """
    stat_row = db.execute(text(stat_query), params).fetchone()

    total_skus = stat_row[0] or 0
    raw_avg = stat_row[1] or 0.0
    avg_conf_pct = round(raw_avg * 100, 1) if raw_avg <= 1.0 else round(raw_avg, 1)
    high_count = stat_row[2] or 0
    med_count = stat_row[3] or 0
    low_count = stat_row[4] or 0

    high_pct = round((high_count / max(1, total_skus)) * 100, 1)
    med_pct = round((med_count / max(1, total_skus)) * 100, 1)
    low_pct = round((low_count / max(1, total_skus)) * 100, 1)

    # 2. Match Source Breakdown
    source_query = f"""
        SELECT
            CASE
                WHEN LOWER(TRIM(COALESCE(match_source, ''))) IN ('matcher', 'catalogue') THEN 'Matcher'
                WHEN LOWER(TRIM(COALESCE(match_source, ''))) = 'classifier' THEN 'Classifier'
                WHEN matched_catalog_name IS NOT NULL AND TRIM(matched_catalog_name) != '' THEN 'Matcher'
                WHEN (bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0) THEN 'Classifier'
                ELSE 'Matcher'
            END as normalized_source,
            COUNT(*) as count
        FROM processed_skus
        WHERE {where_str}
        GROUP BY normalized_source
        ORDER BY count DESC
    """
    source_rows = db.execute(text(source_query), params).fetchall()
    match_sources = [
        {
            "source": r[0],
            "count": r[1],
            "pct": round((r[1] / max(1, total_skus)) * 100, 1),
        }
        for r in source_rows
    ]

    # 3. Domain Breakdown
    dom_query = """
        SELECT domain, COUNT(*) as count
        FROM processed_skus
        WHERE tenant_id = :tenant_id
        GROUP BY domain
    """
    dom_rows = db.execute(text(dom_query), tenant).fetchall()
    domain_breakdown = [{"domain": r[0] or "Unknown", "count": r[1]} for r in dom_rows]

    # 4. Volume Trend (by hour or date)
    volume_trend = []
    is_24h = timeframe == "24h"
    if is_24h:
        trend_query = f"""
            SELECT to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:00') as hr, COUNT(*) as cnt, AVG(confidence) as avg_c
            FROM processed_skus
            WHERE {where_str}
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
            WHERE {where_str}
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

    # 5. Legacy/Job counts for backward compatibility
    active_jobs = db.execute(
        text(
            "SELECT COUNT(*) FROM jobs WHERE tenant_id = :tenant_id AND status IN ('queued', 'running')"
        ),
        tenant,
    ).scalar()
    completed_7d = db.execute(
        text(
            "SELECT COUNT(*) FROM jobs WHERE tenant_id = :tenant_id AND status = 'completed' AND started_at >= now() - interval '7 days'"
        ),
        tenant,
    ).scalar()
    failed_7d = db.execute(
        text(
            "SELECT COUNT(*) FROM jobs WHERE tenant_id = :tenant_id AND status = 'failed' AND started_at >= now() - interval '7 days'"
        ),
        tenant,
    ).scalar()
    requests_24h = db.execute(
        text("SELECT COUNT(*) FROM api_requests WHERE created_at >= now() - interval '24 hours'")
    ).scalar()

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
    where_str, params = _stats_filter(domain, timeframe, principal.tenant_id)

    # 1. Basic Type distribution (top 10)
    bt_query = f"""
        SELECT bt, COUNT(*) as count
        FROM processed_skus
        WHERE {where_str} AND bt IS NOT NULL AND bt != ''
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
        WHERE {where_str} AND COALESCE(NULLIF(region, ''), NULLIF(category, '')) IS NOT NULL
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
    auto_t = engine_config.AUTO_THRESHOLD
    review_t = engine_config.REVIEW_THRESHOLD

    def status_breakdown(column: str) -> dict:
        q = f"""
            SELECT
                SUM(CASE WHEN {column} >= :auto_t THEN 1 ELSE 0 END) as auto_count,
                SUM(CASE WHEN {column} >= :review_t AND {column} < :auto_t THEN 1 ELSE 0 END) as review_count,
                SUM(CASE WHEN {column} < :review_t THEN 1 ELSE 0 END) as low_count
            FROM processed_skus
            WHERE {where_str} AND {column} IS NOT NULL
        """
        row = db.execute(text(q), {"auto_t": auto_t, "review_t": review_t, **params}).fetchone()
        return {"auto": row[0] or 0, "review": row[1] or 0, "low": row[2] or 0}

    tag_status_breakdown = [
        {"tagType": "Basic Type", **status_breakdown("bt_confidence")},
        {"tagType": "Generic Keyword", **status_breakdown("gk_confidence")},
        {"tagType": "Region / Category", **status_breakdown("region_confidence")},
    ]

    return {
        "basicTypeDistribution": basic_type_distribution,
        "regionDistribution": region_distribution,
        "tagStatusBreakdown": tag_status_breakdown,
        "thresholds": {"auto": auto_t, "review": review_t},
    }


@router.get("/jobs/{id}", response_model=JobResponse)
def get_job(
    id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    row = _tenant_job(db, id, principal)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = _enrich_job_dict(dict(row))
    return JobResponse(**job_dict)


@router.post("/jobs/{id}/cancel")
def cancel_job(
    id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    if not _tenant_job(db, id, principal):
        raise HTTPException(status_code=404, detail="Job not found")
    db.execute(
        text(
            "UPDATE jobs SET status = 'cancelled', cancel_requested = 1, completed_at = now() WHERE id = :id AND status IN ('queued', 'running')"
        ),
        {"id": id},
    )
    db.execute(
        text("UPDATE batches SET status = 'cancelled', completed_at = now() WHERE id = :id"),
        {"id": id},
    )
    db.commit()

    # Notify ML Engine microservice
    cancel_engine_job(id)
    return {"message": "Cancellation completed"}


@router.post("/jobs/{id}/retry")
def retry_job(
    id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    row = _tenant_job(db, id, principal)
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

    result = enqueue_job(
        request, task=job_dict.get("type") or "pipeline", tenant_id=principal.tenant_id
    )
    return {"message": "Retry initiated", "new_job_id": result["job_id"]}
