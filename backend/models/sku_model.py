"""SQLAlchemy ORM model for processed per-SKU audit trail results."""

from sqlalchemy import Column, Float, Index, Integer, Text, UniqueConstraint

from backend.models.base_model import Base, build_timestamp_column


class ProcessedSku(Base):
    """ORM definition for the `processed_skus` table."""

    __tablename__ = "processed_skus"

    id = Column("id", Text, primary_key=True)
    batch_id = Column("batch_id", Text)
    sku_index = Column("sku_index", Integer)
    rule_version = Column("rule_version", Text)
    sku_name = Column("sku_name", Text)
    domain = Column("domain", Text)
    bt = Column("bt", Text)
    gk_json = Column("gk_json", Text)
    region = Column("region", Text)
    category = Column("category", Text)
    flavor_extraction = Column("flavor_extraction", Text)
    brand_extraction = Column("brand_extraction", Text)
    confidence = Column("confidence", Float)
    match_source = Column("match_source", Text)
    rules_applied_json = Column("rules_applied_json", Text)
    logic_notes = Column("logic_notes", Text)
    matched_catalog_name = Column("matched_catalog_name", Text)
    match_score = Column("match_score", Float)
    bt_confidence = Column("bt_confidence", Float)
    gk_confidence = Column("gk_confidence", Float)
    region_confidence = Column("region_confidence", Float)
    input_price = Column("input_price", Float)
    input_description = Column("input_description", Text)
    input_category = Column("input_category", Text)
    upload_job_id = Column("upload_job_id", Text)
    upload_status = Column("upload_status", Text)
    created_at = build_timestamp_column("created_at", default=True)

    __table_args__ = (
        # Idempotency key: re-delivering the same job batch updates rows instead of duplicating them
        UniqueConstraint(
            "batch_id",
            "sku_index",
            "rule_version",
            name="uq_processed_skus_job_sku_rule",
        ),
        Index("idx_processed_skus_created_at", "created_at"),
        Index("idx_processed_skus_batch_id", "batch_id"),
        Index("idx_processed_skus_batch_created", "batch_id", "created_at"),
        Index("idx_processed_skus_domain_created", "domain", "created_at"),
        Index(
            "idx_processed_skus_stats",
            "created_at",
            postgresql_include=[
                "domain",
                "confidence",
                "match_source",
                "matched_catalog_name",
                "bt",
                "bt_confidence",
            ],
        ),
        Index(
            "idx_processed_skus_search_trgm",
            "sku_name",
            "bt",
            "gk_json",
            postgresql_using="gin",
            postgresql_ops={
                "sku_name": "gin_trgm_ops",
                "bt": "gin_trgm_ops",
                "gk_json": "gin_trgm_ops",
            },
        ),
    )


processed_skus = ProcessedSku.__table__
