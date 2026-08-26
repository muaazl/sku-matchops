import sqlite3
from engine.rules_engine.db.seed_rules import DB_PATH
from engine.db import ensure_db_initialized

def get_db_connection():
    """FastAPI dependency for SQLite database connection."""
    conn = ensure_db_initialized(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_next_job_id(conn_or_path=None) -> str:
    """Gets the next sequential job ID starting from 1."""
    is_conn = False
    if conn_or_path is None:
        conn = sqlite3.connect(DB_PATH)
    elif hasattr(conn_or_path, 'cursor'):
        conn = conn_or_path
        is_conn = True
    else:
        conn = sqlite3.connect(conn_or_path)

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(CAST(id AS INTEGER)), 0) + 1 FROM jobs WHERE id GLOB '[0-9]*'")
        row = cursor.fetchone()
        next_id = row[0] if row and row[0] else 1
    except Exception:
        next_id = 1
    finally:
        if not is_conn:
            conn.close()
            
    return str(next_id)
