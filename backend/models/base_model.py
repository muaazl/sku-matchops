"""SQLAlchemy DeclarativeBase and shared metadata for PostgreSQL models."""

from sqlalchemy import Column, DateTime, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all Backend Gateway ORM models."""


metadata = Base.metadata


def build_timestamp_column(name: str, default: bool = False) -> Column:
    """Builds a timezone-aware timestamp column with optional server default."""
    return Column(name, DateTime(timezone=True), server_default=func.now() if default else None)
