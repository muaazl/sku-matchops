import json
import logging
import os
import uuid

from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

from engine import config
from engine.core.models import api_requests, processed_skus

logger = logging.getLogger("matchops.db")

_engine: Engine | None = None

_PROCESSED_SKU_KEY = ("batch_id", "sku_index", "rule_version")
_PG_MAX_BIND_PARAMS = 65535


def get_engine() -> Engine:
    """Process-wide pooled PostgreSQL engine shared by the Gateway and the Engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            config.DATABASE_URL,
            pool_size=config.DB_POOL_SIZE,
            max_overflow=config.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            # Keep now()/CURRENT_TIMESTAMP and naive timestamp params in UTC.
            connect_args={"options": "-c timezone=utc", "connect_timeout": 5},
        )
    return _engine


def dispose_engine() -> None:
    """Closes all pooled connections and forgets the engine (e.g. after DATABASE_URL changes)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


def get_connection() -> Connection:
    """Checks a connection out of the shared pool. Callers must close() it."""
    return get_engine().connect()


def init_db() -> None:
    """Applies all pending Alembic migrations (idempotent)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(os.path.join(config.ROOT_DIR, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(config.ROOT_DIR, "migrations"))
    command.upgrade(cfg, "head")


def upsert_processed_skus(conn: Connection, rows: list[dict], chunk_size: int | None = None) -> int:
    """
    Bulk-writes per-SKU results in multi-row INSERT chunks. Rows sharing
    (batch_id, sku_index, rule_version) with an existing row update it in place
    (ON CONFLICT DO UPDATE), so re-delivering a job batch never duplicates records.
    Runs inside the caller's transaction.
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    chunk_size = min(chunk_size or config.DB_BULK_CHUNK_SIZE, _PG_MAX_BIND_PARAMS // len(cols))
    update_cols = [c for c in cols if c != "id" and c not in _PROCESSED_SKU_KEY]

    for start in range(0, len(rows), chunk_size):
        stmt = pg_insert(processed_skus).values(rows[start : start + chunk_size])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_processed_skus_job_sku_rule",
            set_={c: stmt.excluded[c] for c in update_cols},
        )
        conn.execute(stmt)
    return len(rows)


def log_outbound_request(
    url: str,
    method: str,
    payload: object,
    response_status: int,
    response_text: str,
    duration_ms: int,
    path: str = "/doPost",
) -> None:
    """Logs outbound HTTP requests (e.g. Google Sheets webhooks) to api_requests."""
    try:
        payload_str = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
        resp_str = str(response_text) if response_text is not None else ""

        with get_engine().begin() as conn:
            conn.execute(
                api_requests.insert().values(
                    id=str(uuid.uuid4()),
                    method=method,
                    path=path,
                    payload_json_redacted=payload_str,
                    response_json=resp_str,
                    status_code=response_status,
                    duration_ms=duration_ms,
                    ip_address="outbound",
                    headers_json=json.dumps({"Content-Type": "application/json"}),
                    query_params_json=json.dumps({"callback_url": url}),
                )
            )
        logger.info(
            f"[DB] Logged outbound {method} request to {path} (status {response_status}) in api_requests."
        )
    except Exception as e:
        logger.error(f"[DB] Failed to log outbound request to DB: {e}")
