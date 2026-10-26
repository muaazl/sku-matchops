"""tenancy and rbac: tenant_id on operational tables, users table

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# table -> index on (tenant_id, ...) used by tenant-scoped listings
TENANT_INDEXES = {
    "rules": ("idx_rules_tenant_domain", ["tenant_id", "domain"]),
    "jobs": ("idx_jobs_tenant_started", ["tenant_id", "started_at"]),
    "batches": ("idx_batches_tenant", ["tenant_id"]),
    "processed_skus": ("idx_processed_skus_tenant_created", ["tenant_id", "created_at"]),
}


def upgrade() -> None:
    # Existing rows all belong to the 'default' tenant (server default fills them in place)
    for table, (index_name, columns) in TENANT_INDEXES.items():
        op.add_column(
            table, sa.Column("tenant_id", sa.Text(), nullable=False, server_default="default")
        )
        op.create_index(index_name, table, columns)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("is_active", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.CheckConstraint("role IN ('admin', 'rule_engineer', 'reviewer')", name="ck_users_role"),
    )


def downgrade() -> None:
    op.drop_table("users")
    for table, (index_name, _) in TENANT_INDEXES.items():
        op.drop_index(index_name, table_name=table)
        op.drop_column(table, "tenant_id")
