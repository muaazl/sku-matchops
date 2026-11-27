"""Database schema initialization, table verification, and health check queries."""

import logging

from sqlalchemy import inspect, text

from backend.core.database import get_engine
from backend.models import metadata

logger = logging.getLogger("matchops.repositories.db_init")


def initialize_database_schema() -> None:
    """Initializes PostgreSQL extensions, ORM tables, and sequential ID sequences."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

        # Idempotently create any missing tables and indexes
        metadata.create_all(engine)

        with engine.begin() as conn:
            for seq, table, col in (
                ("jobs_id_seq", "jobs", "id"),
                ("rules_id_seq", "rules", "rule_id"),
            ):
                conn.execute(text(f"CREATE SEQUENCE IF NOT EXISTS {seq}"))
                conn.execute(
                    text(
                        f"SELECT setval('{seq}', "
                        f"GREATEST(COALESCE((SELECT MAX(CAST({col} AS BIGINT)) FROM {table} WHERE {col} ~ '^[0-9]+$'), 0), 1), "
                        f"EXISTS(SELECT 1 FROM {table} WHERE {col} ~ '^[0-9]+$'))"
                    )
                )
        logger.info("Database schema, tables, and sequences are fully initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise


def verify_required_tables() -> None:
    """Verifies that all declared model tables exist in PostgreSQL."""
    existing_tables = set(inspect(get_engine()).get_table_names())
    missing_tables = sorted(set(metadata.tables.keys()) - existing_tables)
    if missing_tables:
        raise RuntimeError(f"Missing required PostgreSQL tables: {', '.join(missing_tables)}")


def check_database_health() -> bool:
    """Executes a lightweight connectivity probe against PostgreSQL."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
