import sqlite3
import os

from engine import config
from engine.db import ensure_db_initialized

DB_DIR = config.DB_DIR
DB_PATH = config.DB_PATH

def seed_db():
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Connect and initialize tables
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    # Clear rules tables if re-seeding
    cursor.executescript("""
        DROP TABLE IF EXISTS conditions; 
        DROP TABLE IF EXISTS actions; 
        DROP TABLE IF EXISTS rules;
        DROP TABLE IF EXISTS staged_training_rows;
    """)
    
    ensure_db_initialized(conn)
    
    conn.commit()
    conn.close()
    print(f"Database seeded successfully at {DB_PATH}")

if __name__ == "__main__":
    seed_db()
