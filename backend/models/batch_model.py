"""SQLAlchemy ORM model for upload and portal fetch batches."""

from sqlalchemy import Column, Integer, Text

from backend.models.base_model import Base, build_timestamp_column


class Batch(Base):
    """ORM definition for the `batches` table."""

    __tablename__ = "batches"

    id = Column("id", Text, primary_key=True)
    source = Column("source", Text)
    merchant_id = Column("merchant_id", Text)
    filename = Column("filename", Text)
    domain = Column("domain", Text)
    status = Column("status", Text)
    total_skus = Column("total_skus", Integer)
    created_by = Column("created_by", Text)
    created_at = build_timestamp_column("created_at", default=True)
    completed_at = build_timestamp_column("completed_at")


batches = Batch.__table__
