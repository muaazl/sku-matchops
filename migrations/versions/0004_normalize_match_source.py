"""normalize match_source to TitleCase Matcher and Classifier

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-30
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        UPDATE processed_skus
        SET match_source = CASE
            WHEN LOWER(TRIM(COALESCE(match_source, ''))) IN ('matcher', 'catalogue') THEN 'Matcher'
            WHEN LOWER(TRIM(COALESCE(match_source, ''))) = 'classifier' THEN 'Classifier'
            WHEN matched_catalog_name IS NOT NULL AND TRIM(matched_catalog_name) != '' THEN 'Matcher'
            WHEN (bt IS NOT NULL AND TRIM(bt) != '') OR (bt_confidence IS NOT NULL AND bt_confidence > 0) THEN 'Classifier'
            ELSE 'Matcher'
        END
        WHERE match_source IS NULL
           OR match_source NOT IN ('Matcher', 'Classifier');
    """)
    op.execute("ANALYZE processed_skus;")


def downgrade() -> None:
    pass

