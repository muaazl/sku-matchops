"""
One-off copy of a legacy SQLite database (data/sku-matchops.db) into PostgreSQL.

    python -m scripts.database.migrate [--sqlite data/sku-matchops.db]

Applies Alembic migrations first, then copies every table. Re-runnable: rows whose
primary key already exists in PostgreSQL are skipped.
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, Text, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Add project root to python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine import config
from engine.core.db import get_engine, init_db
from engine.core.models import metadata

DEFAULT_SQLITE_PATH = os.path.join(config.DATA_DIR, "sku-matchops.db")
CHUNK_SIZE = 1000


def _convert(col, val):
    if val is None:
        return None
    if isinstance(col.type, DateTime):
        dt = datetime.fromisoformat(str(val).replace(" ", "T"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    if isinstance(col.type, Text) and not isinstance(val, str):
        # SQLite is dynamically typed: e.g. weight_val held bare floats
        return json.dumps(val) if isinstance(val, (int, float)) else str(val)
    if col.name in ("confidence", "match_score") and val > 1.0:
        return val / 100.0  # legacy 0-100 scale -> 0-1
    return val


def migrate(sqlite_path: str = DEFAULT_SQLITE_PATH) -> dict[str, int]:
    """Copies all rows from the SQLite file into PostgreSQL. Returns rows copied per table."""
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found at {os.path.abspath(sqlite_path)}")

    init_db()
    src = sqlite3.connect(sqlite_path)
    copied = {}
    try:
        with get_engine().begin() as conn:
            for table in metadata.sorted_tables:
                exists = src.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table.name,)
                ).fetchone()
                if not exists:
                    continue
                src_cols = {r[1] for r in src.execute(f"PRAGMA table_info({table.name})")}
                cols = [c for c in table.columns if c.name in src_cols]
                cur = src.execute(f"SELECT {', '.join(c.name for c in cols)} FROM {table.name}")

                copied[table.name] = 0
                while chunk := cur.fetchmany(CHUNK_SIZE):
                    rows = [{c.name: _convert(c, v) for c, v in zip(cols, r)} for r in chunk]
                    conn.execute(pg_insert(table).values(rows).on_conflict_do_nothing())
                    copied[table.name] += len(rows)

                # Advance SERIAL sequences past the copied ids
                if isinstance(table.c.id.type, Integer):
                    conn.execute(
                        text(
                            f"SELECT setval(pg_get_serial_sequence('{table.name}', 'id'), "
                            f"COALESCE((SELECT MAX(id) FROM {table.name}), 0) + 1, false)"
                        )
                    )
                print(f"Copied {copied[table.name]} rows into {table.name}.")
    finally:
        src.close()

    print("Migration completed successfully.")
    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a legacy SQLite database into PostgreSQL.")
    parser.add_argument("--sqlite", default=DEFAULT_SQLITE_PATH, help="Path to the SQLite file")
    migrate(parser.parse_args().sqlite)
