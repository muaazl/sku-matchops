import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from backend.app.core import config as app_config
from backend.app.db.schema import metadata
from backend.app.db.session import get_engine

logger = logging.getLogger("matchops.db.init")


def _run_alembic_upgrade() -> bool:
    """Runs Alembic migrations programmatically to head."""
    # Search for alembic.ini in project root or current working directory
    possible_paths = [
        Path(os.getcwd()) / "alembic.ini",
        Path(__file__).resolve().parent.parent.parent.parent / "alembic.ini",
        Path(__file__).resolve().parent.parent.parent / "alembic.ini",
    ]
    ini_path = next((p for p in possible_paths if p.is_file()), None)
    if not ini_path:
        logger.warning("alembic.ini not found. Falling back to metadata.create_all.")
        return False

    alembic_cfg = Config(str(ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", app_config.DATABASE_URL)
    alembic_cfg.attributes["configure_logger"] = False
    try:
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic schema migrations upgraded to head successfully.")
        return True
    except Exception as e:
        err_msg = str(e)
        if "Can't locate revision identified by" in err_msg:
            try:
                logger.info(
                    f"Legacy or orphaned revision detected in database ({err_msg}). Stamping current head revision..."
                )
                command.stamp(alembic_cfg, "head", purge=True)
                logger.info("Alembic schema version stamped to head successfully.")
                return True
            except Exception as stamp_err:
                logger.warning(f"Could not auto-stamp Alembic head: {stamp_err}")
        logger.warning(f"Alembic upgrade encountered an issue ({e}). Checking table status...")
        return False


def init_db() -> None:
    """Synchronizes PostgreSQL database schema and ID sequences using Alembic migrations."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

        migrated = _run_alembic_upgrade()
        if not migrated:
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
        logger.info("Database schema and sequences are fully synchronized.")
    except Exception as e:
        logger.error(f"Failed to synchronize database schema: {e}")
        raise
