import logging
import os

from engine import config

logger = logging.getLogger("matchops.db.init")


def init_db() -> None:
    """Applies all pending Alembic schema migrations (idempotent)."""
    from alembic import command
    from alembic.config import Config

    # Locate alembic.ini (either in root or /app)
    ini_path = os.path.join(config.ROOT_DIR, "alembic.ini")
    if not os.path.exists(ini_path) and os.path.exists("/app/alembic.ini"):
        ini_path = "/app/alembic.ini"

    migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")

    cfg = Config(ini_path)
    cfg.set_main_option("script_location", migrations_dir)
    command.upgrade(cfg, "head")
    logger.info("Database schema is fully synchronized with head.")
