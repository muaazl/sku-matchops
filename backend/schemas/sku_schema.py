"""Pydantic schemas for SKU matching, classification, pipeline requests, and processed SKUs."""

from datetime import datetime

from pydantic import BaseModel

from backend.core.constants import DOMAIN_MARKET


class SKUItem(BaseModel):
    name: str
    price: float | None = 0.0
    description: str | None = ""
    category: str | None = ""


class BaseRequest(BaseModel):
    skus: list[SKUItem]
    domain: str = DOMAIN_MARKET
    callback_url: str
    spreadsheet_id: str | None = None
    sheet_name: str | None = None


class MatchRequest(BaseRequest):
    pass


class ClassifyRequest(BaseRequest):
    pass


class PipelineRequest(BaseRequest):
    pass


class ProcessedSkuResponse(BaseModel):
    id: str
    batch_id: str | None = None
    sku_name: str
    domain: str
    bt: str | None = None
    gk_json: str | None = None
    region: str | None = None
    confidence: float | None = None
    match_source: str | None = None
    rules_applied_json: str | None = None
    logic_notes: str | None = None
    matched_catalog_name: str | None = None
    match_score: float | None = None
    bt_confidence: float | None = None
    gk_confidence: float | None = None
    region_confidence: float | None = None
    input_price: float | None = None
    input_description: str | None = None
    input_category: str | None = None
    created_at: datetime | None = None
