import os
import sqlite3
import sys

from engine.config import DB_PATH
from engine.core.db import ensure_db_initialized

# Add project root to python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def migrate():
    print(f"Connecting to database at {os.path.abspath(DB_PATH)}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    try:
        # 1. Set WAL mode
        conn.execute("PRAGMA journal_mode=WAL;")
        print("Set PRAGMA journal_mode=WAL;")

        # 2. Drop deprecated tables if they exist
        conn.execute("DROP TABLE IF EXISTS staged_training_rows;")
        conn.execute("DROP TABLE IF EXISTS merchant_credentials;")
        print("Dropped deprecated tables (staged_training_rows, merchant_credentials).")

        # 3. Create all tables and indexes if they do not exist
        ensure_db_initialized(conn)
        print("Ensured all 11 tables and indexes exist.")

        # 4. Handle duration_minutes migration if both columns exist
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(jobs);")
        updated_jobs_columns = [row[1] for row in cursor.fetchall()]
        if (
            "duration_minutes" in updated_jobs_columns
            and "duration_seconds" in updated_jobs_columns
        ):
            conn.execute(
                "UPDATE jobs SET duration_minutes = duration_seconds / 60.0 WHERE duration_minutes IS NULL AND duration_seconds IS NOT NULL;"
            )
            conn.commit()
            print("Migrated existing duration_seconds values to duration_minutes.")

        # 5. Handle dictionary tables columns migration (catalog_count, metadata_json, gk_count)
        cursor.execute("PRAGMA table_info(classifier_dictionaries);")
        dict_cols = [row[1] for row in cursor.fetchall()]
        if "catalog_count" not in dict_cols:
            conn.execute(
                "ALTER TABLE classifier_dictionaries ADD COLUMN catalog_count INTEGER DEFAULT 0;"
            )
            print("Added catalog_count to classifier_dictionaries.")
        if "metadata_json" not in dict_cols:
            conn.execute("ALTER TABLE classifier_dictionaries ADD COLUMN metadata_json TEXT;")
            print("Added metadata_json to classifier_dictionaries.")

        cursor.execute("PRAGMA table_info(brand_flavors);")
        brand_cols = [row[1] for row in cursor.fetchall()]
        if "catalog_count" not in brand_cols:
            conn.execute("ALTER TABLE brand_flavors ADD COLUMN catalog_count INTEGER DEFAULT 0;")
            print("Added catalog_count to brand_flavors.")

        cursor.execute("PRAGMA table_info(bt_gk_map);")
        map_cols = [row[1] for row in cursor.fetchall()]
        if "gk_count" not in map_cols:
            conn.execute("ALTER TABLE bt_gk_map ADD COLUMN gk_count INTEGER DEFAULT 0;")
            print("Added gk_count to bt_gk_map.")
        if "catalog_count" not in map_cols:
            conn.execute("ALTER TABLE bt_gk_map ADD COLUMN catalog_count INTEGER DEFAULT 0;")
            print("Added catalog_count to bt_gk_map.")

        # Re-ensure indexes after adding columns
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_classifier_dict_lookup ON classifier_dictionaries(domain, tag_type, catalog_count DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_brand_flavors_count ON brand_flavors(domain, catalog_count DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bt_gk_map_count ON bt_gk_map(domain, catalog_count DESC);"
        )
        conn.commit()

        print("Migration completed successfully.")

    except Exception as e:
        print(f"Error during migration: {e}", file=sys.stderr)
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
