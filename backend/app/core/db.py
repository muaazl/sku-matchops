from sqlalchemy import Connection, text

from engine.core.db import get_connection, get_engine

__all__ = ["get_db_connection", "get_engine", "get_next_job_id"]


def get_db_connection():
    """FastAPI dependency yielding a pooled PostgreSQL connection."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def get_next_job_id(conn: Connection) -> str:
    """Gets the next sequential job ID starting from 1."""
    row = conn.execute(
        text("SELECT COALESCE(MAX(CAST(id AS BIGINT)), 0) + 1 FROM jobs WHERE id ~ '^[0-9]+$'")
    ).fetchone()
    return str(row[0] if row and row[0] else 1)
