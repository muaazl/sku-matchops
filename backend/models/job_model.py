"""SQLAlchemy ORM model for asynchronous matching, classification, and upload jobs."""

from sqlalchemy import Column, Float, Index, Integer, Text

from backend.models.base_model import Base, build_timestamp_column


class Job(Base):
    """ORM definition for the `jobs` table."""

    __tablename__ = "jobs"

    id = Column("id", Text, primary_key=True)
    batch_id = Column("batch_id", Text)
    type = Column("type", Text)
    status = Column("status", Text)
    current_stage = Column("current_stage", Text)
    total_items = Column("total_items", Integer, server_default="0")
    completed_items = Column("completed_items", Integer, server_default="0")
    cancel_requested = Column("cancel_requested", Integer, server_default="0")
    retry_of_job_id = Column("retry_of_job_id", Text)
    error_message = Column("error_message", Text)
    created_by = Column("created_by", Text)
    started_at = build_timestamp_column("started_at")
    updated_at = build_timestamp_column("updated_at")
    completed_at = build_timestamp_column("completed_at")
    domain = Column("domain", Text)
    sheet_name = Column("sheet_name", Text)
    target_sheet = Column("target_sheet", Text)
    duration_minutes = Column("duration_minutes", Float)
    high_conf = Column("high_conf", Integer, server_default="0")
    med_conf = Column("med_conf", Integer, server_default="0")
    low_conf = Column("low_conf", Integer, server_default="0")
    match_rate = Column("match_rate", Float, server_default="0")
    input_skus_json = Column("input_skus_json", Text)

    __table_args__ = (
        Index("idx_jobs_started_at", "started_at"),
        Index("idx_jobs_status_started", "status", "started_at"),
        Index(
            "idx_jobs_sheet_trgm",
            "sheet_name",
            "target_sheet",
            postgresql_using="gin",
            postgresql_ops={"sheet_name": "gin_trgm_ops", "target_sheet": "gin_trgm_ops"},
        ),
    )


jobs = Job.__table__
