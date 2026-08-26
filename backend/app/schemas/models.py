from typing import Optional, Literal
from pydantic import BaseModel, Field
from engine import config


class SKUItem(BaseModel):
    name: str
    price: Optional[float] = 0.0
    description: Optional[str] = ""
    category: Optional[str] = ""


class BaseRequest(BaseModel):
    skus: list[SKUItem]
    domain: str = config.DOMAIN_MARKET
    callback_url: str
    spreadsheet_id: Optional[str] = None
    sheet_name: Optional[str] = None


class MatchRequest(BaseRequest):
    pass

class ClassifyRequest(BaseRequest):
    pass

class PipelineRequest(BaseRequest):
    pass


class MatchResult(BaseModel):
    matched_catalog_name: str
    score: float
    status: str
    logic_notes: str
    rules_applied: str = ""
    suggested_bt: Optional[str] = ""
    suggested_gk: Optional[str] = ""
    suggested_region: Optional[str] = ""

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
    suggested_bt: Optional[str] = None
    bt_confidence: Optional[float] = None
    bt_status: Optional[str] = None
    suggested_gk: Optional[str] = None
    gk_confidence: Optional[float] = None
    gk_status: Optional[str] = None
    suggested_region: Optional[str] = None
    region_confidence: Optional[float] = None
    region_status: Optional[str] = None
    pipeline_source: Optional[str] = None
    escalated: bool = False

class PipelineResponse(BaseModel):
    domain: str
    total: int
    escalated_count: int
    results: list[PipelineResult]

# -- Jobs Models --
class JobResponse(BaseModel):
    id: str
    batch_id: Optional[str] = None
    type: str
    status: str
    current_stage: str
    total_items: int
    completed_items: int
    eta_seconds: Optional[int] = None
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    domain: Optional[str] = None
    sheet_name: Optional[str] = None
    target_sheet: Optional[str] = None
    duration_minutes: Optional[float] = None
    high_conf: Optional[int] = None
    med_conf: Optional[int] = None
    low_conf: Optional[int] = None
    match_rate: Optional[float] = None
    input_skus_json: Optional[str] = None
    progress_pct: Optional[float] = 0.0

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
    score_threshold: Optional[float] = None
    filters: Optional[dict] = None

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
        "flavor_is"
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
        "normalize_sku"
    ]
    value: str = Field(..., max_length=200)

class RuleModel(BaseModel):
    rule_id: str = Field(..., max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    domain: Literal["market", "food", "shared"]
    module: Literal["bt_override", "gk_injection", "formatter", "visibility"]
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
    rule_id: Optional[str] = None

class EnqueueJobResponse(BaseModel):
    job_id: str
    status: str
    total_skus: int

class BatchResponse(BaseModel):
    id: str
    source: Optional[str] = None
    filename: Optional[str] = None
    merchant_id: Optional[str] = None
    domain: Optional[str] = None
    status: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None

class ProcessedSkuResponse(BaseModel):
    id: str
    batch_id: Optional[str] = None
    sku_name: str
    domain: str
    bt: Optional[str] = None
    gk_json: Optional[str] = None
    region: Optional[str] = None
    confidence: Optional[float] = None
    match_source: Optional[str] = None
    rules_applied_json: Optional[str] = None
    logic_notes: Optional[str] = None
    matched_catalog_name: Optional[str] = None
    match_score: Optional[float] = None
    bt_confidence: Optional[float] = None
    gk_confidence: Optional[float] = None
    region_confidence: Optional[float] = None
    created_at: Optional[str] = None

class ApiRequestResponse(BaseModel):
    id: str
    method: str
    path: str
    status_code: int
    duration_ms: Optional[int] = None
    ip_address: Optional[str] = None
    created_at: Optional[str] = None

class ApiRequestDetailResponse(ApiRequestResponse):
    headers_json: Optional[str] = None
    query_params_json: Optional[str] = None
    payload_json_redacted: Optional[str] = None
    response_json: Optional[str] = None


