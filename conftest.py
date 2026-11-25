"""
Points every test at a dedicated PostgreSQL database (never the app's DATABASE_URL).

    TEST_DATABASE_URL=postgresql+psycopg://matchops:matchops@localhost:5432/matchops_test
    TEST_REDIS_URL=redis://localhost:6379/15

The database is created if missing and schema-synced once per session. Tests that
need it use the `pg` fixture and are skipped when no PostgreSQL server is reachable.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from backend.core import config as backend_config
from engine.core import config

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://matchops:matchops@localhost:5432/matchops_test"
)
backend_config.DATABASE_URL = TEST_DATABASE_URL
# Isolated Redis DB so tests never enqueue into (or read) the app's real queues
config.REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
backend_config.REDIS_URL = config.REDIS_URL
# Tests expect dev-mode auth by default unless opted in via monkeypatch (e.g. `auth_on`)
os.environ["AUTH_ENABLED"] = "false"
backend_config.AUTH_ENABLED = False

_pg_error: str | None = None
_loopback_error: str | None = None


def _check_loopback() -> str | None:
    import socket

    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        c = socket.socket()
        c.settimeout(0.5)
        try:
            c.connect(("127.0.0.1", port))
            return None
        except Exception as e:
            return f"TCP loopback connect forbidden on this environment (e.g. WinError 10013): {e}"
        finally:
            c.close()
    finally:
        s.close()


@pytest.fixture(scope="session")
def loopback():
    """Skips tests requiring live loopback TCP sockets when loopback connections are forbidden."""
    global _loopback_error
    if _loopback_error is None:
        _loopback_error = _check_loopback() or ""
    if _loopback_error:
        pytest.skip(_loopback_error)


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
    """Synchronized test database engine; skips the test when PostgreSQL is unavailable."""
    global _pg_error
    if _pg_error is None:
        try:
            from backend.repositories.database_init_repository import initialize_database_schema

            _create_database_if_missing(TEST_DATABASE_URL)
            initialize_database_schema()
            _pg_error = ""
        except Exception as e:
            _pg_error = f"PostgreSQL not available at {TEST_DATABASE_URL}: {e}"
    if _pg_error:
        pytest.skip(_pg_error)

    from backend.core.database import get_engine

    return get_engine()
