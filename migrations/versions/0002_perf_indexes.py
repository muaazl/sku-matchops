"""Add trigram search indexes and expand covering dashboard stats index

Revision ID: 0002_perf_indexes
Revises: 0001_initial_schema
Create Date: 2026-11-09 14:30:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_perf_indexes"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_sheet_trgm ON jobs "
        "USING gin (sheet_name gin_trgm_ops, target_sheet gin_trgm_ops);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_processed_skus_search_trgm ON processed_skus "
        "USING gin (sku_name gin_trgm_ops, bt gin_trgm_ops, gk_json gin_trgm_ops);"
    )
    op.execute("DROP INDEX IF EXISTS idx_processed_skus_tenant_stats;")
    op.execute(
        "CREATE INDEX idx_processed_skus_tenant_stats ON processed_skus (tenant_id, created_at) "
        "INCLUDE (domain, confidence, match_source, matched_catalog_name, bt, bt_confidence);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_processed_skus_search_trgm;")
    op.execute("DROP INDEX IF EXISTS idx_jobs_sheet_trgm;")
    op.execute("DROP INDEX IF EXISTS idx_processed_skus_tenant_stats;")
    op.execute(
        "CREATE INDEX idx_processed_skus_tenant_stats ON processed_skus (tenant_id, created_at) "
        "INCLUDE (domain, confidence, match_source);"
    )
