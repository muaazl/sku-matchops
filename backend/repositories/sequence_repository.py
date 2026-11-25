"""PostgreSQL sequence allocation queries for jobs and rules."""

from sqlalchemy import Connection, text


def allocate_next_sequence_id(conn: Connection, seq_name: str, table: str, col: str) -> str:
    """Allocates the next sequential numeric ID in O(1) via PostgreSQL sequence (MAX+1 fallback)."""
    val = conn.execute(
        text("SELECT CASE WHEN to_regclass(:seq) IS NOT NULL THEN nextval(:seq) END"),
        {"seq": seq_name},
    ).scalar()
    if val is not None:
        return str(val)
    row = conn.execute(
        text(
            f"SELECT COALESCE(MAX(CAST({col} AS BIGINT)), 0) + 1 FROM {table} WHERE {col} ~ '^[0-9]+$'"
        )
    ).fetchone()
    return str(row[0] if row and row[0] else 1)


def allocate_next_job_id(conn: Connection) -> str:
    """Allocates the next sequential job ID starting from 1."""
    return allocate_next_sequence_id(conn, "jobs_id_seq", "jobs", "id")


def allocate_next_rule_id(conn: Connection) -> str:
    """Allocates the next sequential rule ID starting from 1."""
    return allocate_next_sequence_id(conn, "rules_id_seq", "rules", "rule_id")
