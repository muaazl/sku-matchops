"""SQLAlchemy ORM model for HTTP and webhook request audit logs."""

from sqlalchemy import Column, Index, Integer, Text

from backend.models.base_model import Base, build_timestamp_column


class ApiRequest(Base):
    """ORM definition for the `api_requests` audit table."""

    __tablename__ = "api_requests"

    id = Column("id", Text, primary_key=True)
    method = Column("method", Text)
    path = Column("path", Text)
    payload_json_redacted = Column("payload_json_redacted", Text)
    response_json = Column("response_json", Text)
    status_code = Column("status_code", Integer)
    duration_ms = Column("duration_ms", Integer)
    ip_address = Column("ip_address", Text)
    headers_json = Column("headers_json", Text)
    query_params_json = Column("query_params_json", Text)
    created_at = build_timestamp_column("created_at", default=True)

    __table_args__ = (Index("idx_api_requests_created_at", "created_at"),)


api_requests = ApiRequest.__table__
