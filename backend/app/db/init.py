import logging

from sqlalchemy import text

from backend.app.db.schema import metadata
from backend.app.db.session import get_engine

logger = logging.getLogger("matchops.db.init")


def init_db() -> None:
    """Synchronizes PostgreSQL database schema and ID sequences directly using SQLAlchemy metadata."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
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
        logger.info("Database schema is fully synchronized.")
    except Exception as e:
        logger.error(f"Failed to synchronize database schema: {e}")
        raise
