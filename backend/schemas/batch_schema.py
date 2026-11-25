"""Pydantic schemas for CSV upload batches and portal merchant fetches."""

from datetime import datetime

from pydantic import BaseModel


class MerchantFetchRequest(BaseModel):
    merchant_id: str
    bearer_token: str
    portal_url: str
    domain: str
    task: str = "pipeline"


class BatchResponse(BaseModel):
    id: str
    source: str | None = None
    filename: str | None = None
    merchant_id: str | None = None
    domain: str | None = None
    status: str | None = None
    created_by: str | None = None
    created_at: datetime | None = None
