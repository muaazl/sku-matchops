from typing import Literal

from pydantic import BaseModel, Field

from engine import config


class SKUItem(BaseModel):
    name: str
    price: float | None = 0.0
    description: str | None = ""
    category: str | None = ""


class BaseRequest(BaseModel):
    skus: list[SKUItem]
    domain: str = config.DOMAIN_MARKET
    callback_url: str
    spreadsheet_id: str | None = None
    sheet_name: str | None = None


class MatchRequest(BaseRequest):
    pass


class ClassifyRequest(BaseRequest):
    pass


class PipelineRequest(BaseRequest):
    pass


class UploadRequest(BaseModel):
    outlet_id_or_name: str
    skus: list[dict]
    domain: str = config.DOMAIN_MARKET


class MatchResult(BaseModel):
    matched_catalog_name: str
    score: float
    status: str
    logic_notes: str
    rules_applied: str = ""
    suggested_bt: str | None = ""
    suggested_gk: str | None = ""
    suggested_region: str | None = ""


class TagResponse(BaseModel):
    domain: str
    total: int
    results: list[MatchResult]


class ClassifyResult(BaseModel):
    suggested_bt: str
    bt_confidence: float
    bt_status: str
    bt_source: str
    suggested_gk: str
    gk_confidence: float
    gk_status: str
    suggested_region: str
    region_confidence: float
    region_status: str
    region_source: str
    model: str | None = None
    bt_model: str | None = None
    rules_applied: str = ""
    logic_notes: str = ""


class ClassifyResponse(BaseModel):
    domain: str
    total: int
    results: list[ClassifyResult]


class PipelineResult(BaseModel):
    matched_catalog_name: str
    score: float
    status: str
    logic_notes: str
    rules_applied: str = ""
    suggested_bt: str | None = None
    bt_confidence: float | None = None
    bt_status: str | None = None
    suggested_gk: str | None = None
    gk_confidence: float | None = None
    gk_status: str | None = None
    suggested_region: str | None = None
    region_confidence: float | None = None
    region_status: str | None = None
    model: str | None = None
    bt_model: str | None = None
    pipeline_source: str | None = None
    escalated: bool = False


class PipelineResponse(BaseModel):
    domain: str
    total: int
    escalated_count: int
    results: list[PipelineResult]


# -- Jobs Models --
class JobResponse(BaseModel):
    id: str
    batch_id: str | None = None
    type: str
    status: str
    current_stage: str
    total_items: int
    completed_items: int
    eta_seconds: int | None = None
    error_message: str | None = None
    created_by: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    domain: str | None = None
    sheet_name: str | None = None
    target_sheet: str | None = None
    duration_minutes: float | None = None
    high_conf: int | None = None
    med_conf: int | None = None
    low_conf: int | None = None
    match_rate: float | None = None
    input_skus_json: str | None = None
    progress_pct: float | None = 0.0


# -- Batches Models --
class BatchCreateRequest(BaseModel):
    source: str
    domain: str
    created_by: str


class MerchantFetchRequest(BaseModel):
    merchant_id: str
    bearer_token: str
    portal_url: str
    domain: str
    task: str = "pipeline"


# -- Qdrant Proxy Models --
class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict | None = None


# -- Rules API Models --
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


class RuleActionModel(BaseModel):
    action_type: Literal[
        "set_bt",
        "add_gk",
        "remove_gk",
        "set_region",
        "set_category",
        "set_visibility",
        "normalize_sku",
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


class EnqueueJobResponse(BaseModel):
    job_id: str
    status: str
    total_skus: int


class BatchResponse(BaseModel):
    id: str
    source: str | None = None
    filename: str | None = None
    merchant_id: str | None = None
    domain: str | None = None
    status: str | None = None
    created_by: str | None = None
    created_at: str | None = None


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
    created_at: str | None = None


class ApiRequestResponse(BaseModel):
    id: str
    method: str
    path: str
    status_code: int
    duration_ms: int | None = None
    ip_address: str | None = None
    created_at: str | None = None


class ApiRequestDetailResponse(ApiRequestResponse):
    headers_json: str | None = None
    query_params_json: str | None = None
    payload_json_redacted: str | None = None
    response_json: str | None = None
