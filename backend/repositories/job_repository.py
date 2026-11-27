"""SQLAlchemy repository for job lifecycle persistence, querying, and restart reconciliation."""

import json
import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from backend.core.database import get_engine
from backend.models.job_model import jobs as jobs_table
from backend.repositories.batch_repository import reconcile_interrupted_batches, update_batch_status
from backend.repositories.sequence_repository import allocate_next_job_id
from backend.repositories.sku_repository import upsert_processed_skus

logger = logging.getLogger("matchops.repositories.job")

# The job list endpoint never displays uploaded SKUs (~15 KB/job); detail and retry still read them.
JOB_LIST_COLUMNS = ", ".join(c.name for c in jobs_table.columns if c.name != "input_skus_json")


def insert_queued_job_with_retry(
    *,
    job_type: str,
    domain: str,
    total_items: int,
    input_skus_json: str,
    sheet_name: str | None,
    target_sheet: str | None,
    max_attempts: int = 5,
) -> str:
    """Allocates a sequential job ID and inserts a queued `jobs` row with collision retry."""
    for attempt in range(1, max_attempts + 1):
        conn = get_engine().connect()
        try:
            job_id = allocate_next_job_id(conn)
            conn.execute(
                text(
                    """
                    INSERT INTO jobs (
                        id, batch_id, type, status, current_stage,
                        total_items, completed_items, created_by,
                        started_at, domain, sheet_name, target_sheet, input_skus_json
                    ) VALUES (
                        :id, :id, :type, 'queued', 'queued', :total_items, 0, 'system',
                        now(), :domain, :sheet_name, :target_sheet, :input_skus_json
                    )
                    """
                ),
                {
                    "id": job_id,
                    "type": job_type,
                    "total_items": total_items,
                    "domain": domain,
                    "sheet_name": sheet_name,
                    "target_sheet": target_sheet,
                    "input_skus_json": input_skus_json,
                },
            )
            conn.commit()
            return job_id
        except IntegrityError as e:
            conn.rollback()
            if attempt == max_attempts:
                logger.error(
                    f"Failed to allocate a unique job id after {max_attempts} attempts: {e}"
                )
                raise RuntimeError(
                    f"Could not allocate a unique job id after {max_attempts} attempts: {e}"
                )
            logger.warning(
                f"Job id collided with an existing row (attempt {attempt}/{max_attempts}); retrying with a new id."
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to insert job into DB: {e}")
            raise RuntimeError(f"Database error registering job: {e}")
        finally:
            conn.close()
    raise RuntimeError("Unreachable")


def mark_job_dispatch_failed(job_id: str, error_message: str) -> None:
    """Marks a queued job failed when engine/queue dispatch throws an error."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET status = 'failed', current_stage = 'failed', "
                "error_message = :err WHERE id = :id"
            ),
            {"err": error_message, "id": job_id},
        )


def fetch_job_by_id(job_id: str) -> dict[str, Any] | None:
    """Fetches a full `jobs` row by ID."""
    with get_engine().connect() as conn:
        row = (
            conn.execute(text("SELECT * FROM jobs WHERE id = :id"), {"id": job_id})
            .mappings()
            .fetchone()
        )
    return dict(row) if row else None


def _build_job_filter_clauses(
    *,
    status: str | None = None,
    job_type: str | None = None,
    created_by: str | None = None,
    domain: str | None = None,
    sheet_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if status:
        clauses.append("status = :status")
        params["status"] = status
    if job_type:
        clauses.append("type = :type")
        params["type"] = job_type
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
    return where, params


def count_and_fetch_jobs(
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
) -> tuple[int, list[dict[str, Any]]]:
    """Counts and fetches filtered `jobs` rows (omitting `input_skus_json`)."""
    where, params = _build_job_filter_clauses(
        status=status,
        job_type=job_type,
        created_by=created_by,
        domain=domain,
        sheet_name=sheet_name,
        date_from=date_from,
        date_to=date_to,
    )
    with get_engine().connect() as conn:
        total_count = int(
            conn.execute(text(f"SELECT COUNT(*) FROM jobs {where}"), params).scalar() or 0
        )
        query = f"SELECT {JOB_LIST_COLUMNS} FROM jobs {where} ORDER BY started_at DESC"
        if limit is not None and limit > 0:
            query += " LIMIT :limit OFFSET :offset"
            params.update(limit=limit, offset=(page - 1) * limit)
        rows = conn.execute(text(query), params).mappings().fetchall()
    return total_count, [dict(r) for r in rows]


def fetch_recent_jobs(limit: int = 25) -> list[dict[str, Any]]:
    """Fetches the most recent `jobs` rows for SSE initial snapshots."""
    with get_engine().connect() as conn:
        rows = (
            conn.execute(
                text(f"SELECT {JOB_LIST_COLUMNS} FROM jobs ORDER BY started_at DESC LIMIT :limit"),
                {"limit": limit},
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows]


def cancel_active_job(job_id: str) -> tuple[bool, dict[str, Any] | None]:
    """Atomically transitions a queued/running job and its batch to cancelled."""
    with get_engine().begin() as conn:
        res = conn.execute(
            text(
                "UPDATE jobs SET status = 'cancelled', cancel_requested = 1, completed_at = now() "
                "WHERE id = :id AND status IN ('queued', 'running')"
            ),
            {"id": job_id},
        )
        if res.rowcount == 0:
            existing = (
                conn.execute(text("SELECT * FROM jobs WHERE id = :id"), {"id": job_id})
                .mappings()
                .fetchone()
            )
            return False, dict(existing) if existing else None

        update_batch_status(job_id, "cancelled", conn=conn)
    return True, None


def requeue_job_for_replay(job_id: str, batch_id: str) -> bool:
    """Resets job and batch rows back to queued for DLQ replay. Returns False if job missing."""
    with get_engine().begin() as conn:
        res = conn.execute(
            text(
                "UPDATE jobs SET status = 'queued', current_stage = 'queued', error_message = NULL, "
                "completed_at = NULL WHERE id = :id"
            ),
            {"id": job_id},
        )
        if res.rowcount == 0:
            return False
        conn.execute(
            text("UPDATE batches SET status = 'queued', completed_at = NULL WHERE id = :id"),
            {"id": batch_id},
        )
    return True


def mark_job_failed_after_replay_error(
    job_id: str,
    batch_id: str | None,
    error_message: str,
    *,
    include_batch: bool = True,
) -> None:
    """Compensating SQL rollback marking job (and optionally batch) failed after replay error."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET status = 'failed', current_stage = 'failed', "
                "error_message = :err WHERE id = :id"
            ),
            {"err": error_message, "id": job_id},
        )
        if include_batch and batch_id:
            conn.execute(
                text("UPDATE batches SET status = 'failed' WHERE id = :id"),
                {"id": batch_id},
            )


def fetch_job_status_and_inputs(job_id: str) -> dict[str, Any] | None:
    """Fetches status, cancel_requested, domain, and input_skus_json for host upload updates."""
    with get_engine().connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT id, status, cancel_requested, domain, input_skus_json "
                    "FROM jobs WHERE id = :id"
                ),
                {"id": job_id},
            )
            .mappings()
            .fetchone()
        )
    return dict(row) if row else None


def activate_job_if_queued(job_id: str, stage: str) -> dict[str, Any] | None:
    """Checks job state and transitions queued->running atomically for host uploads."""
    with get_engine().begin() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT id, status, cancel_requested, domain, input_skus_json "
                    "FROM jobs WHERE id = :id"
                ),
                {"id": job_id},
            )
            .mappings()
            .fetchone()
        )
        if not row:
            return None
        row_dict = dict(row)
        cancel_requested = bool(row_dict["cancel_requested"]) or row_dict["status"] == "cancelled"
        if (
            not cancel_requested
            and row_dict["status"] not in ("completed", "failed", "cancelled")
            and row_dict["status"] != "running"
        ):
            conn.execute(
                text(
                    "UPDATE jobs SET status = 'running', current_stage = :stage, updated_at = now() "
                    "WHERE id = :id AND status IN ('queued', 'running')"
                ),
                {"stage": stage, "id": job_id},
            )
        return row_dict


def update_job_stage_if_active(job_id: str, stage: str) -> tuple[bool, tuple[str, Any] | None]:
    """Updates current_stage on active jobs; returns (True, None) or (False, (status, cancel_req))."""
    with get_engine().begin() as conn:
        rowcount = conn.execute(
            text(
                "UPDATE jobs SET status = 'running', current_stage = :stage, updated_at = now() "
                "WHERE id = :id AND status IN ('queued', 'running') AND COALESCE(cancel_requested, 0) = 0"
            ),
            {"stage": stage, "id": job_id},
        ).rowcount
        if rowcount == 0:
            job_row = conn.execute(
                text("SELECT status, cancel_requested FROM jobs WHERE id = :id"),
                {"id": job_id},
            ).fetchone()
            return False, (job_row[0], job_row[1]) if job_row else None
    return True, None


def complete_job_atomically(
    *,
    job_id: str,
    row_builder: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    completed_items: int,
    duration_minutes: float,
    high_conf: int,
    med_conf: int,
    low_conf: int,
    match_rate: float,
) -> tuple[int, list[dict[str, Any]]]:
    """Atomically verifies non-cancelled state, upserts SKUs, and marks job + batch completed."""
    with get_engine().begin() as conn:
        input_row = conn.execute(
            text("SELECT input_skus_json, status, cancel_requested FROM jobs WHERE id = :id"),
            {"id": job_id},
        ).fetchone()
        if input_row and (input_row[1] == "cancelled" or bool(input_row[2])):
            return -1, []

        input_skus: list[dict[str, Any]] = []
        if input_row and input_row[0]:
            try:
                input_skus = json.loads(input_row[0])
            except Exception:
                pass

        rows = row_builder(input_skus)
        upsert_processed_skus(conn, rows)

        conn.execute(
            text(
                """
                UPDATE jobs SET
                    status = 'completed',
                    current_stage = 'done',
                    completed_items = :completed_items,
                    completed_at = now(),
                    duration_minutes = :duration_minutes,
                    high_conf = :high_conf,
                    med_conf = :med_conf,
                    low_conf = :low_conf,
                    match_rate = :match_rate
                WHERE id = :id AND status != 'cancelled' AND COALESCE(cancel_requested, 0) = 0
                """
            ),
            {
                "completed_items": completed_items,
                "duration_minutes": duration_minutes,
                "high_conf": high_conf,
                "med_conf": med_conf,
                "low_conf": low_conf,
                "match_rate": match_rate,
                "id": job_id,
            },
        )
        update_batch_status(job_id, "completed", conn=conn)
    return len(rows), rows


def mark_job_failed_or_cancelled(
    *,
    job_id: str,
    error_message: str,
    duration_minutes: float = 0.0,
    is_cancelled: bool = False,
) -> None:
    """Marks a job (and its originating batch, if any) failed or cancelled."""
    status = "cancelled" if is_cancelled else "failed"
    current_stage = "cancelled" if is_cancelled else "failed"
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                UPDATE jobs SET
                    status = :status,
                    current_stage = :current_stage,
                    error_message = :error_message,
                    duration_minutes = :duration_minutes,
                    completed_at = now()
                WHERE id = :id
                """
            ),
            {
                "status": status,
                "current_stage": current_stage,
                "error_message": error_message,
                "duration_minutes": duration_minutes,
                "id": job_id,
            },
        )
        update_batch_status(job_id, status, conn=conn)


def fetch_stale_live_job_ids(live_job_ids: list[str]) -> set[str]:
    """Finds live Redis job IDs whose DB records are already terminal or cancel-requested."""
    if not live_job_ids:
        return set()
    with get_engine().connect() as conn:
        stale_rows = conn.execute(
            text(
                """
                SELECT id FROM jobs
                WHERE id = ANY(:live)
                  AND (status IN ('cancelled', 'completed', 'failed') OR COALESCE(cancel_requested, 0) = 1)
                """
            ),
            {"live": live_job_ids},
        ).fetchall()
    return {str(r[0]) for r in stale_rows}


def fail_orphaned_jobs_and_batches(live_job_ids: list[str]) -> int:
    """Marks queued/running jobs missing from Redis as failed and reconciles orphaned batches."""
    with get_engine().begin() as conn:
        cur = conn.execute(
            text(
                """
                UPDATE jobs SET
                    status = 'failed',
                    current_stage = 'failed',
                    error_message = 'Interrupted by a service restart. Use Retry to resubmit.',
                    completed_at = now()
                WHERE status IN ('queued', 'running') AND NOT (id = ANY(:live))
                """
            ),
            {"live": live_job_ids},
        )
        reconcile_interrupted_batches(conn=conn)
        return int(cur.rowcount or 0)


def aggregate_job_status_counts() -> tuple[int, int, int]:
    """Returns (active_jobs, completed_7d, failed_7d) in a single table scan."""
    with get_engine().connect() as conn:
        row = conn.execute(
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
    if not row:
        return 0, 0, 0
    return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
