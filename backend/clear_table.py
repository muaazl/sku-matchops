import argparse
import os
import sys
import sqlite3

# Add project root to python path to import seed_rules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.rules_engine.db.seed_rules import DB_PATH

def clear_table(table_name):
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file does not exist at {os.path.abspath(DB_PATH)}")
        sys.exit(1)

    print(f"Connecting to database at {os.path.abspath(DB_PATH)}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Validate that the table exists (prevents SQL injection and typos)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        if table_name not in tables:
            print(f"Error: Table '{table_name}' does not exist in the database.")
            print(f"Available tables: {', '.join(tables)}")
            sys.exit(1)

        # Get count before deletion
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]

        if row_count == 0:
            print(f"Table '{table_name}' is already empty. No action taken.")
            return

        # Delete rows
        cursor.execute(f"DELETE FROM {table_name};")
        conn.commit()

        print(f"Successfully deleted {row_count} rows from table '{table_name}'.")

    except Exception as e:
        print(f"Error clearing table '{table_name}': {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erase data from a database table.")
    parser.add_argument("--table", help="The name of the table to clear.")
    parser.add_argument("--arg", help="The name of the table to clear.")
    
    args = parser.parse_args()
    
    table_name = args.table or args.arg
    if not table_name:
        parser.error("You must specify a table using either --table or --arg")

    clear_table(table_name)
