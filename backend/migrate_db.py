import sqlite3
import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine import config
from engine.db import ensure_db_initialized

DB_PATH = config.DB_PATH

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
        if "duration_minutes" in updated_jobs_columns and "duration_seconds" in updated_jobs_columns:
            conn.execute("UPDATE jobs SET duration_minutes = duration_seconds / 60.0 WHERE duration_minutes IS NULL AND duration_seconds IS NOT NULL;")
            conn.commit()
            print("Migrated existing duration_seconds values to duration_minutes.")
            
        print("Migration completed successfully.")
        
    except Exception as e:
        print(f"Error during migration: {e}", file=sys.stderr)
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
