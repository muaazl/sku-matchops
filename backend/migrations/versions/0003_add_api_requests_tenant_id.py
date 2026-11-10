"""Audit remediation: add composite indexes, scope unique constraints, add tenant_id to requests

Revision ID: 0003_add_api_requests_tenant_id
Revises: 0002_perf_indexes
Create Date: 2026-11-10 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_add_api_requests_tenant_id"
down_revision: str | None = "0002_perf_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add tenant_id to api_requests and composite index
    op.add_column(
        "api_requests",
        sa.Column("tenant_id", sa.Text(), nullable=True, server_default="default"),
    )
    op.create_index(
        "idx_api_requests_tenant_created",
        "api_requests",
        ["tenant_id", "created_at"],
    )

    # 2. Scope username uniqueness per tenant
    op.drop_constraint("users_username_key", "users", type_="unique")
    op.create_unique_constraint("uq_users_tenant_username", "users", ["tenant_id", "username"])

    # 3. Scope processed_skus idempotency constraint to tenant
    op.drop_constraint("uq_processed_skus_job_sku_rule", "processed_skus", type_="unique")
    op.create_unique_constraint(
        "uq_processed_skus_tenant_job_sku_rule",
        "processed_skus",
        ["tenant_id", "batch_id", "sku_index", "rule_version"],
    )

    # 4. Composite performance indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_tenant_status_started "
        "ON jobs (tenant_id, status, started_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_processed_skus_tenant_batch "
        "ON processed_skus (tenant_id, batch_id, created_at DESC);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_processed_skus_tenant_batch;")
    op.execute("DROP INDEX IF EXISTS idx_jobs_tenant_status_started;")
    op.drop_constraint("uq_processed_skus_tenant_job_sku_rule", "processed_skus", type_="unique")
    op.create_unique_constraint(
        "uq_processed_skus_job_sku_rule",
        "processed_skus",
        ["batch_id", "sku_index", "rule_version"],
    )
    op.drop_constraint("uq_users_tenant_username", "users", type_="unique")
    op.create_unique_constraint("users_username_key", "users", ["username"])
    op.drop_index("idx_api_requests_tenant_created", table_name="api_requests")
    op.drop_column("api_requests", "tenant_id")
