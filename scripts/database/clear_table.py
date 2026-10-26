import argparse
import os
import sys

from sqlalchemy import inspect, text

# Add project root to python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.core.db import get_engine


def clear_table(table_name):
    engine = get_engine()
    print(f"Connecting to database at {engine.url.render_as_string(hide_password=True)}")

    # Validate that the table exists (prevents SQL injection and typos)
    tables = inspect(engine).get_table_names()
    if table_name not in tables:
        print(f"Error: Table '{table_name}' does not exist in the database.")
        print(f"Available tables: {', '.join(tables)}")
        sys.exit(1)

    try:
        with engine.begin() as conn:
            # Get count before deletion
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()

            if row_count == 0:
                print(f"Table '{table_name}' is already empty. No action taken.")
                return

            # Delete rows
            conn.execute(text(f"DELETE FROM {table_name}"))

        print(f"Successfully deleted {row_count} rows from table '{table_name}'.")

    except Exception as e:
        print(f"Error clearing table '{table_name}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erase data from a database table.")
    parser.add_argument("--table", help="The name of the table to clear.")
    parser.add_argument("--arg", help="The name of the table to clear.")

    args = parser.parse_args()

    table_name = args.table or args.arg
    if not table_name:
        parser.error("You must specify a table using either --table or --arg")

    clear_table(table_name)
