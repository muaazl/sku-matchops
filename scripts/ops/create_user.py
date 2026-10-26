"""
Creates a MatchOps user (e.g. the first platform admin before turning on AUTH_ENABLED).

    python -m scripts.ops.create_user --username ops --role admin [--tenant default]

The password is read from the MATCHOPS_NEW_USER_PASSWORD env var or prompted for.
"""

import argparse
import getpass
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.api.endpoints.auth import create_user
from backend.app.core.security import ROLES
from engine.core.db import get_engine, init_db

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a MatchOps user.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--tenant", default="default", help="'default' = platform operator")
    args = parser.parse_args()

    password = os.getenv("MATCHOPS_NEW_USER_PASSWORD") or getpass.getpass("Password: ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")

    init_db()
    with get_engine().begin() as conn:
        user_id = create_user(conn, args.username, password, args.role, args.tenant)
    print(f"Created user {args.username} (id={user_id}, role={args.role}, tenant={args.tenant}).")
