"""dashboard covering index: tenant-scoped processed_skus aggregates without heap reads

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("idx_processed_skus_tenant_created", table_name="processed_skus")
    op.create_index(
        "idx_processed_skus_tenant_stats",
        "processed_skus",
        ["tenant_id", "created_at"],
        postgresql_include=["domain", "confidence", "match_source"],
    )
    # Bulk loads left the planner with stale row estimates (1.7k estimated vs 343k actual)
    op.execute("ANALYZE processed_skus")
    op.execute("ANALYZE jobs")


def downgrade() -> None:
    op.drop_index("idx_processed_skus_tenant_stats", table_name="processed_skus")
    op.create_index(
        "idx_processed_skus_tenant_created", "processed_skus", ["tenant_id", "created_at"]
    )
