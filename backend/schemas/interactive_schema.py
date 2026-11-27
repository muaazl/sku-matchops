"""Pydantic schemas for interactive single-SKU inference, suggestion, audit, and rule rerun."""

from pydantic import BaseModel


class RunSingleRequest(BaseModel):
    sku_name: str
    domain: str
    task: str
    price: float | None = 0.0
    description: str | None = ""
    category: str | None = ""


class RerunRulesRequest(BaseModel):
    sku_name: str
    domain: str
    bt: str
    gk: str
    region: str | None = ""
    category: str | None = ""
    price: float | None = 0.0
    confidence: float | None = 1.0
    match_source: str | None = "Classifier"
    matched_sku: str | None = ""
    reasoning: str | None = ""


class SuggestRequest(BaseModel):
    sku_name: str
    domain: str
    current_bt: str | None = ""
    exclude_bt: str | None = ""
    exclude_gk: str | None = ""
