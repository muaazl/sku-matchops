"""SQLAlchemy pooled PostgreSQL engine and connection lifecycle management."""

from collections.abc import Iterator

from sqlalchemy import Connection, Engine, create_engine

from backend.core import config

_engine: Engine | None = None


def get_engine() -> Engine:
    """Returns the process-wide pooled PostgreSQL engine for the Backend API Gateway."""
    global _engine
    if _engine is None:
        connect_args = {"options": "-c timezone=utc", "connect_timeout": 5}
        _engine = create_engine(
            config.DATABASE_URL,
            pool_size=config.DB_POOL_SIZE,
            max_overflow=config.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=getattr(config, "DB_POOL_RECYCLE", 1800),
            # Keep now()/CURRENT_TIMESTAMP and naive timestamp params in UTC.
            connect_args=connect_args,
        )
    return _engine


def dispose_engine() -> None:
    """Closes all pooled connections and clears the cached SQLAlchemy engine."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


def get_connection() -> Connection:
    """Checks out a connection from the shared pool. Callers must close() it."""
    return get_engine().connect()


def get_db_connection() -> Iterator[Connection]:
    """Yields a pooled PostgreSQL connection and guarantees closure."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
