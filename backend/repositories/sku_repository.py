"""SQLAlchemy repository for `processed_skus` persistence, history search, streaming, and stats."""

from collections.abc import Iterator
from typing import Any

from sqlalchemy import Connection, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.core import config
from backend.core.database import get_engine
from backend.models.sku_model import processed_skus

_PROCESSED_SKU_KEY = ("batch_id", "sku_index", "rule_version")
_PG_MAX_BIND_PARAMS = 65535
_TIMEFRAME_INTERVALS = {"24h": "24 hours", "7d": "7 days", "30d": "30 days"}


def upsert_processed_skus(
    conn: Connection,
    rows: list[dict[str, Any]],
    chunk_size: int | None = None,
) -> int:
    """Bulk-upserts per-SKU results keyed by (batch_id, sku_index, rule_version)."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    effective_chunk = min(chunk_size or config.DB_BULK_CHUNK_SIZE, _PG_MAX_BIND_PARAMS // len(cols))
    update_cols = [c for c in cols if c != "id" and c not in _PROCESSED_SKU_KEY]

    for start in range(0, len(rows), effective_chunk):
        stmt = pg_insert(processed_skus).values(rows[start : start + effective_chunk])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_processed_skus_job_sku_rule",
            set_={c: stmt.excluded[c] for c in update_cols},
        )
        conn.execute(stmt)
    return len(rows)


def _build_processed_sku_filters(
    *,
    batch_id: str | None = None,
    domain: str | None = None,
    min_confidence: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sku_name: str | None = None,
    bt: str | None = None,
    gk: str | None = None,
    match_source: str | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if batch_id:
        clauses.append("batch_id = :batch_id")
        params["batch_id"] = batch_id
    if domain:
        clauses.append("domain = :domain")
        params["domain"] = domain
    if min_confidence is not None:
        clauses.append("confidence >= :min_confidence")
        params["min_confidence"] = min_confidence
    if date_from:
        clauses.append("created_at >= CAST(:date_from AS timestamptz)")
        params["date_from"] = date_from
    if date_to:
        clauses.append("created_at <= CAST(:date_to AS timestamptz)")
        params["date_to"] = date_to
    if sku_name:
        clauses.append("sku_name ILIKE :sku_name")
        params["sku_name"] = f"%{sku_name}%"
    if bt:
        clauses.append("bt ILIKE :bt")
        params["bt"] = f"%{bt}%"
    if gk:
        clauses.append("gk_json ILIKE :gk")
        params["gk"] = f"%{gk}%"
    if match_source:
        source_val = match_source.strip().lower()
        if source_val == "matcher":
            clauses.append(
                """(
                LOWER(TRIM(COALESCE(match_source, ''))) IN ('matcher', 'catalogue')
                OR (
                    (match_source IS NULL OR TRIM(match_source) = '' OR LOWER(TRIM(match_source)) = 'unknown')
                    AND NOT (
                        (matched_catalog_name IS NULL OR TRIM(matched_catalog_name) = '')
                        AND ((bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0))
                    )
                )
            )"""
            )
        elif source_val == "classifier":
            clauses.append(
                """(
                LOWER(TRIM(COALESCE(match_source, ''))) = 'classifier'
                OR (
                    (match_source IS NULL OR TRIM(match_source) = '' OR LOWER(TRIM(match_source)) = 'unknown')
                    AND (matched_catalog_name IS NULL OR TRIM(matched_catalog_name) = '')
                    AND ((bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0))
                )
            )"""
            )
        else:
            clauses.append("match_source ILIKE :match_source")
            params["match_source"] = f"%{match_source}%"

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def count_and_fetch_processed_skus(
    *,
    batch_id: str | None = None,
    domain: str | None = None,
    min_confidence: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sku_name: str | None = None,
    bt: str | None = None,
    gk: str | None = None,
    match_source: str | None = None,
    page: int = 1,
    limit: int | None = 50,
) -> tuple[int, list[dict[str, Any]]]:
    """Counts and fetches paginated `processed_skus` rows matching optional filters."""
    where, params = _build_processed_sku_filters(
        batch_id=batch_id,
        domain=domain,
        min_confidence=min_confidence,
        date_from=date_from,
        date_to=date_to,
        sku_name=sku_name,
        bt=bt,
        gk=gk,
        match_source=match_source,
    )
    with get_engine().connect() as conn:
        total_count = int(
            conn.execute(text(f"SELECT COUNT(*) FROM processed_skus {where}"), params).scalar() or 0
        )
        query = f"SELECT * FROM processed_skus {where} ORDER BY created_at DESC"
        if limit is not None and limit > 0:
            query += " LIMIT :limit OFFSET :offset"
            params.update(limit=limit, offset=(page - 1) * limit)
        rows = conn.execute(text(query), params).mappings().fetchall()
    return total_count, [dict(row) for row in rows]


def count_all_processed_skus() -> int:
    """Returns total number of rows in `processed_skus`."""
    with get_engine().connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM processed_skus")).scalar() or 0)


def fetch_processed_sku_by_id(sku_id: str) -> dict[str, Any] | None:
    """Fetches a single `processed_skus` row by primary key."""
    with get_engine().connect() as conn:
        row = (
            conn.execute(text("SELECT * FROM processed_skus WHERE id = :id"), {"id": sku_id})
            .mappings()
            .fetchone()
        )
    return dict(row) if row else None


def stream_processed_skus_partitions(
    ids: list[str] | None = None,
    batch_size: int = 2000,
    db: Connection | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Streams partitions of `processed_skus` rows for CSV export while bounding memory."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if ids:
        clauses.append("id = ANY(:ids)")
        params["ids"] = ids

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT * FROM processed_skus {where} ORDER BY created_at DESC LIMIT 50000"

    def _run_stream(conn: Connection) -> Iterator[list[dict[str, Any]]]:
        exec_conn = (
            conn.execution_options(stream_results=True, max_row_buffer=batch_size)
            if hasattr(conn, "execution_options")
            else conn
        )
        result = exec_conn.execute(text(query), params)
        try:
            mappings = result.mappings()
            partitions = (
                mappings.partitions(batch_size)
                if hasattr(mappings, "partitions")
                else [mappings.fetchall()]
            )
            for partition in partitions:
                yield [dict(row) for row in partition]
        finally:
            if hasattr(result, "close"):
                result.close()

    if db is not None:
        yield from _run_stream(db)
    else:
        with get_engine().connect() as conn:
            yield from _run_stream(conn)


def _build_stats_filter(
    domain: str | None, timeframe: str | None
) -> tuple[list[str], dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if domain and domain.lower() != "all":
        clauses.append("domain = :domain")
        params["domain"] = domain.lower()
    if timeframe in _TIMEFRAME_INTERVALS:
        clauses.append("created_at >= now() - CAST(:window AS interval)")
        params["window"] = _TIMEFRAME_INTERVALS[timeframe]
    return clauses, params


def aggregate_sku_source_metrics(domain: str | None, timeframe: str | None) -> list[tuple]:
    """Aggregates match-source counts and confidence buckets via the covering stats index."""
    clauses, params = _build_stats_filter(domain, timeframe)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
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
    with get_engine().connect() as conn:
        return conn.execute(text(query), params).fetchall()


def aggregate_sku_domain_counts() -> list[tuple]:
    """Counts processed SKUs grouped by domain."""
    with get_engine().connect() as conn:
        return conn.execute(
            text("SELECT domain, COUNT(*) as count FROM processed_skus GROUP BY domain")
        ).fetchall()


def aggregate_sku_volume_trend(
    domain: str | None, timeframe: str | None, *, hourly: bool
) -> list[tuple]:
    """Aggregates processed SKU counts and average confidence by hour or calendar date."""
    clauses, params = _build_stats_filter(domain, timeframe)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    fmt = "YYYY-MM-DD HH24:00" if hourly else "YYYY-MM-DD"
    query = f"""
        SELECT to_char(created_at AT TIME ZONE 'UTC', '{fmt}') as bucket, COUNT(*) as cnt, AVG(confidence) as avg_c
        FROM processed_skus
        {where_sql}
        GROUP BY bucket
        ORDER BY bucket ASC
    """
    with get_engine().connect() as conn:
        return conn.execute(text(query), params).fetchall()


def aggregate_top_basic_types(
    domain: str | None, timeframe: str | None, limit: int = 10
) -> list[tuple]:
    """Fetches top BasicType frequencies across processed SKUs."""
    clauses, params = _build_stats_filter(domain, timeframe)
    bt_clauses = clauses + ["bt IS NOT NULL", "bt != ''"]
    where_bt = f"WHERE {' AND '.join(bt_clauses)}"
    query = f"""
        SELECT bt, COUNT(*) as count
        FROM processed_skus
        {where_bt}
        GROUP BY bt
        ORDER BY count DESC
        LIMIT :limit
    """
    with get_engine().connect() as conn:
        return conn.execute(text(query), {**params, "limit": limit}).fetchall()


def aggregate_top_region_or_category(
    domain: str | None, timeframe: str | None, limit: int = 10
) -> list[tuple]:
    """Fetches top Region/Category tag frequencies across processed SKUs."""
    clauses, params = _build_stats_filter(domain, timeframe)
    tag_clauses = clauses + ["COALESCE(NULLIF(region, ''), NULLIF(category, '')) IS NOT NULL"]
    where_tag = f"WHERE {' AND '.join(tag_clauses)}"
    query = f"""
        SELECT COALESCE(NULLIF(region, ''), NULLIF(category, '')) as tag, COUNT(*) as count
        FROM processed_skus
        {where_tag}
        GROUP BY tag
        ORDER BY count DESC
        LIMIT :limit
    """
    with get_engine().connect() as conn:
        return conn.execute(text(query), {**params, "limit": limit}).fetchall()


def aggregate_tag_confidence_tiers(
    domain: str | None,
    timeframe: str | None,
    auto_threshold: float,
    review_threshold: float,
) -> tuple[int, ...]:
    """Computes AUTO/REVIEW/LOW bucket counts across BT, GK, and Region/Category."""
    clauses, params = _build_stats_filter(domain, timeframe)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
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
    with get_engine().connect() as conn:
        row = conn.execute(
            text(query),
            {"auto_t": auto_threshold, "review_t": review_threshold, **params},
        ).fetchone()
    return tuple(int(v or 0) for v in (row or (0,) * 9))
