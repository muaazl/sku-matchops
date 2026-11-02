import logging

from sqlalchemy import text

from backend.app.db.schema import metadata
from backend.app.db.session import get_engine

logger = logging.getLogger("matchops.db.init")


def init_db() -> None:
    """Synchronizes PostgreSQL database schema directly using SQLAlchemy metadata."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        metadata.create_all(engine)
        logger.info("Database schema is fully synchronized.")
    except Exception as e:
        logger.error(f"Failed to synchronize database schema: {e}")
        raise
