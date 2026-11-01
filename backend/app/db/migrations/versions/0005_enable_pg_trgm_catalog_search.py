"""enable pg_trgm extension and add GIN trigram index on catalog_items

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-01
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    op.create_index(
        "catalog_search_trgm_idx",
        "catalog_items",
        ["name", "description"],
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops", "description": "gin_trgm_ops"},
    )


def downgrade() -> None:
    op.drop_index("catalog_search_trgm_idx", table_name="catalog_items")
