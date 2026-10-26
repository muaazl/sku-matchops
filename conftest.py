"""
Points every test at a dedicated PostgreSQL database (never the app's DATABASE_URL).

    TEST_DATABASE_URL=postgresql+psycopg://matchops:matchops@localhost:5432/matchops_test
    TEST_REDIS_URL=redis://localhost:6379/15

The database is created if missing and migrated with Alembic once per session. Tests that
need it use the `pg` fixture and are skipped when no PostgreSQL server is reachable.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from engine import config

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://matchops:matchops@localhost:5432/matchops_test"
)
config.DATABASE_URL = TEST_DATABASE_URL
# Isolated Redis DB so tests never enqueue into (or read) the app's real queues
config.REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")

_pg_error: str | None = None


def _create_database_if_missing(url_str: str) -> None:
    url = make_url(url_str)
    admin = create_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 5},
    )
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def pg():
    """Migrated test database engine; skips the test when PostgreSQL is unavailable."""
    global _pg_error
    if _pg_error is None:
        try:
            from engine.core.db import init_db

            _create_database_if_missing(TEST_DATABASE_URL)
            init_db()
            _pg_error = ""
        except Exception as e:
            _pg_error = f"PostgreSQL not available at {TEST_DATABASE_URL}: {e}"
    if _pg_error:
        pytest.skip(_pg_error)

    from engine.core.db import get_engine

    return get_engine()
