"""Pydantic schemas for deterministic rule definitions, testing, and reordering."""

from typing import Literal

from pydantic import BaseModel, Field


class RuleConditionModel(BaseModel):
    condition_group: int = Field(..., ge=1)
    condition_type: Literal[
        "sku_contains",
        "bt_is",
        "gk_contains",
        "category_contains",
        "region_is",
        "price_below",
        "price_above",
        "flavor_contains",
        "flavor_is",
    ]
    value: str = Field(..., max_length=200)
    negate: int = Field(default=0, ge=0, le=1)
    operator: Literal["AND", "OR"] = "AND"


class RuleActionModel(BaseModel):
    action_type: Literal[
        "set_bt",
        "add_gk",
        "remove_gk",
        "set_region",
        "set_category",
    ]
    value: str = Field(..., max_length=200)


class RuleModel(BaseModel):
    rule_id: str = Field(..., max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    domain: Literal["market", "food", "shared"]
    priority: int = Field(..., ge=1, le=10000)
    description: str = Field(..., max_length=250)
    reasoning: str = Field(..., max_length=500)
    condition_logic: Literal["AND", "OR"] = "AND"
    is_active: int = Field(default=1, ge=0, le=1)
    conditions: list[RuleConditionModel] = []
    actions: list[RuleActionModel] = []


class RuleTestRequest(BaseModel):
    sample_record: dict


class RuleDraftTestRequest(BaseModel):
    rule: RuleModel
    sample_record: dict


class RuleReorderRequest(BaseModel):
    ordered_rule_ids: list[str] = []


class RuleOperationResponse(BaseModel):
    message: str
    rule_id: str | None = None
