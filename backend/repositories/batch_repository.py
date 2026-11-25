"""SQLAlchemy repository for upload and merchant-fetch batch records."""

from typing import Any

from sqlalchemy import Connection, text

from backend.core.database import get_engine


def insert_batch(
    *,
    batch_id: str,
    source: str,
    filename: str,
    domain: str,
    status: str,
    created_by: str,
) -> None:
    """Inserts a new row into the `batches` table."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO batches (id, source, filename, domain, status, created_by)
                VALUES (:id, :source, :filename, :domain, :status, :created_by)
                """
            ),
            {
                "id": batch_id,
                "source": source,
                "filename": filename,
                "domain": domain,
                "status": status,
                "created_by": created_by,
            },
        )


def fetch_batch_by_id(batch_id: str) -> dict[str, Any] | None:
    """Fetches a single `batches` record by primary key."""
    with get_engine().connect() as conn:
        row = (
            conn.execute(text("SELECT * FROM batches WHERE id = :id"), {"id": batch_id})
            .mappings()
            .fetchone()
        )
    return dict(row) if row else None


def update_batch_status(batch_id: str, status: str, conn: Connection | None = None) -> None:
    """Updates terminal status and completion timestamp on a batch record."""
    stmt = text("UPDATE batches SET status = :status, completed_at = now() WHERE id = :id")
    params = {"status": status, "id": batch_id}
    if conn is not None:
        conn.execute(stmt, params)
        return
    with get_engine().begin() as tx_conn:
        tx_conn.execute(stmt, params)


def requeue_batch_for_replay(batch_id: str, conn: Connection | None = None) -> None:
    """Resets a batch row back to 'queued' during DLQ replay."""
    stmt = text("UPDATE batches SET status = 'queued', completed_at = NULL WHERE id = :id")
    params = {"id": batch_id}
    if conn is not None:
        conn.execute(stmt, params)
        return
    with get_engine().begin() as tx_conn:
        tx_conn.execute(stmt, params)


def mark_batch_failed_on_replay_error(batch_id: str, conn: Connection | None = None) -> None:
    """Marks a batch failed as compensation when DLQ replay fails in Redis."""
    stmt = text("UPDATE batches SET status = 'failed' WHERE id = :id")
    params = {"id": batch_id}
    if conn is not None:
        conn.execute(stmt, params)
        return
    with get_engine().begin() as tx_conn:
        tx_conn.execute(stmt, params)


def reconcile_interrupted_batches(conn: Connection | None = None) -> None:
    """Marks queued/running batches failed when all associated jobs are no longer active."""
    stmt = text(
        """
        UPDATE batches SET status = 'failed', completed_at = now()
        WHERE status IN ('queued', 'running')
          AND id NOT IN (
              SELECT DISTINCT batch_id FROM jobs
              WHERE status IN ('queued', 'running')
                AND batch_id IS NOT NULL
          )
        """
    )
    if conn is not None:
        conn.execute(stmt)
        return
    with get_engine().begin() as tx_conn:
        tx_conn.execute(stmt)
