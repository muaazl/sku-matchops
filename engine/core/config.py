"""Engine configuration, runtime paths, model manifests, and matching/classification thresholds."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from engine.core.onnx_factory import check_cuda_provider_available

# --- Path Configuration (Root & Base) ---
BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

# --- Resource & Threading Limits ---
# Limit CPU thread pools to half of available cores to avoid resource starvation
MAX_CPU_CORES = max(1, (os.cpu_count() or 4) // 2)
MAX_CONCURRENT_TASKS = MAX_CPU_CORES
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = str(MAX_CPU_CORES)


class _MaxLengthLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Asking to truncate to max_length" not in record.getMessage()


logging.getLogger("transformers.tokenization_utils_base").addFilter(_MaxLengthLogFilter())

# --- Data & Cache Paths ---
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
STAGING_DIR = DATA_DIR / "staged_sheets"
QDRANT_DATA_DIR = DATA_DIR / "qdrant"

# "json" emits one JSON object per log line (with trace_id); anything else is plain text
LOG_FORMAT = "text"

# --- External Services & Integrations ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# Rule-cache distribution & invalidation (snapshot + version keys + pub/sub)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Engine & Callback Services ---
ENGINE_GRPC_PORT = int(os.getenv("ENGINE_GRPC_PORT", "50051"))
BACKEND_GRPC_URL = os.getenv("BACKEND_GRPC_URL", "127.0.0.1:50052")
BACKEND_HTTP_URL = os.getenv("BACKEND_HTTP_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("MATCHOPS_API_KEY", "bismillah")
INTERNAL_GRPC_SECRET = os.getenv("INTERNAL_GRPC_SECRET", "")

if not API_KEY:
    raise RuntimeError("A non-empty MATCHOPS_API_KEY must be configured in environment variables.")

# --- Distributed job queue (Redis) ---
# Off -> direct gRPC dispatch to a single engine. On but Redis unreachable -> same fallback.
TASK_QUEUE_ENABLED = True
# Batches with at most this many SKUs go to the realtime tier, larger ones to bulk
TASK_REALTIME_MAX_SKUS = 50
# Total attempts per task (first run included) before it moves to the dead-letter queue
TASK_MAX_RETRIES = 3
# Retry delay after attempt n is BACKOFF * 2**(n-1) seconds
TASK_RETRY_BACKOFF_SECONDS = 5.0
# A worker must renew its lease within this window or the task is reclaimed (heartbeat runs every lease/3)
TASK_LEASE_SECONDS = float(os.getenv("TASK_LEASE_SECONDS", "120.0"))
TASK_POLL_INTERVAL = 0.5

# --- Qdrant Vector DB Settings ---
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
QDRANT_HNSW_ON_DISK = os.getenv("QDRANT_HNSW_ON_DISK", "false").lower() == "true"
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 100
QDRANT_INDEXING_THRESHOLD = 10000
QDRANT_MEMMAP_THRESHOLD = 5000
QDRANT_TIMEOUT = 60.0
# Max requests per Qdrant query_batch_points call
QDRANT_BATCH_CHUNK_SIZE = 100

# --- Domain & Collection Identifiers ---
DOMAIN_MARKET = "market"
DOMAIN_FOOD = "food"
QDRANT_COLLECTION_MARKET = "market_catalog"
QDRANT_COLLECTION_FOOD = "food_catalog"
QDRANT_TAGS_PREFIX = ""
CACHE_SALT = "MatchOps_v3"

# --- Model Artifacts & Quantization (Strict ONNX + INT8) ---
ONNX_DIR = DATA_DIR / "models"
ARCFACE_DIR = ONNX_DIR / "arcface"
EMBEDDING_ROOT_DIR = ONNX_DIR / "embedding"
RERANKER_ROOT_DIR = ONNX_DIR / "reranker"
GLINER_MODEL = ONNX_DIR / "gliner"


def normalize_model_folder_name(model_id: str) -> str:
    """Normalizes a HuggingFace model ID into a lowercase folder name (model basename only)."""
    clean = (model_id or "").strip().rstrip("/")
    base = clean.split("/")[-1] if "/" in clean else clean
    return base.lower().replace(" ", "-")


def read_json_manifest(path: Path | str) -> dict:
    """Reads a JSON manifest file and returns its dict payload, or empty dict if missing/invalid."""
    p = Path(path)
    if p.exists():
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def get_embedding_manifest() -> dict:
    """Returns the JSON manifest dictionary for the active embedding model."""
    return read_json_manifest(EMBEDDING_MANIFEST)


def get_reranker_manifest() -> dict:
    """Returns the JSON manifest dictionary for the active reranker model."""
    return read_json_manifest(RERANKER_MANIFEST)


def check_learned_sparse_enabled() -> bool:
    """True only when the configured embedding model generates learned sparse weights (BGE-M3)."""
    manifest = get_embedding_manifest()
    if "has_learned_sparse" in manifest:
        return bool(manifest["has_learned_sparse"])
    return "bge-m3" in EMBEDDING_MODEL.lower()


def resolve_embedding_dim(default: int = 1024) -> int:
    """Resolves active embedding vector dimensionality from manifest or environment."""
    manifest = get_embedding_manifest()
    if "embedding_dim" in manifest:
        try:
            return int(manifest["embedding_dim"])
        except (TypeError, ValueError):
            pass
    return int(os.getenv("EMBEDDING_DIM", str(default)))


def resolve_reranker_arch() -> str:
    """Returns active reranker architecture ('listwise_lbnl' or 'cross_encoder')."""
    manifest = get_reranker_manifest()
    arch = str(manifest.get("arch_type") or "").strip().lower()
    if arch in ("listwise_lbnl", "cross_encoder"):
        return arch
    return "listwise_lbnl" if "jina-reranker-v3" in RERANKER_MODEL.lower() else "cross_encoder"


def resolve_reranker_thresholds(
    arch: str, model_id: str | None = None
) -> tuple[float, float, float, float]:
    """
    Returns (conf_high, conf_medium, reranker_threshold, reranker_margin)
    calibrated to the active reranker's score scale from Phase 4/5 evaluation.
    """
    rerank_id = (model_id if model_id is not None else globals().get("RERANKER_MODEL", "")).lower()
    if arch == "listwise_lbnl" or "jina-reranker-v3" in rerank_id:
        # Listwise LBNL outputs raw cosine similarities (~0.25-0.70 for strong matches).
        return 0.62, 0.35, 0.30, 0.15
    if "jina-reranker-v2" in rerank_id:
        # Jina v2 multilingual cross-encoder outputs compressed sigmoid scores (pos median ~0.38-0.42).
        return 0.65, 0.35, 0.35, 0.12
    # Standard cross-encoders (e.g. BAAI/bge-reranker-v2-m3) apply sigmoid to logits -> [0.0, 1.0].
    return 0.78, 0.45, 0.50, 0.12


def resolve_onnx_artifact_and_providers(
    fp32_path: Path | str,
    fp16_path: Path | str | None,
    int8_path: Path | str,
    precision: str | None = None,
) -> tuple[Path, list[str], str]:
    """
    Resolves (onnx_path, providers, effective_precision) based on ONNX_PRECISION ('auto'|'fp16'|'fp32'|'int8').
    In 'auto' mode, prefers GPU FP16 (or FP32) with CUDAExecutionProvider when a CUDA GPU and FP16/FP32
    artifact are available, and falls back to CPUExecutionProvider + INT8 ONNX otherwise.
    """
    p_fp32 = Path(fp32_path)
    p_fp16 = Path(fp16_path) if fp16_path else None
    p_int8 = Path(int8_path)
    mode = (precision or globals().get("ONNX_PRECISION") or "auto").strip().lower()
    has_cuda = check_cuda_provider_available()

    if mode == "int8":
        return p_int8, ["CPUExecutionProvider"], "int8"

    if mode == "fp16":
        if p_fp16 is not None and p_fp16.exists():
            provs = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if has_cuda
                else ["CPUExecutionProvider"]
            )
            return p_fp16, provs, "fp16"
        if has_cuda and p_fp32.exists():
            return p_fp32, ["CUDAExecutionProvider", "CPUExecutionProvider"], "fp32"
        return p_int8, ["CPUExecutionProvider"], "int8"

    if mode == "fp32":
        if p_fp32.exists():
            provs = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if has_cuda
                else ["CPUExecutionProvider"]
            )
            return p_fp32, provs, "fp32"
        if p_fp16 is not None and p_fp16.exists():
            provs = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if has_cuda
                else ["CPUExecutionProvider"]
            )
            return p_fp16, provs, "fp16"
        return p_int8, ["CPUExecutionProvider"], "int8"

    # Default 'auto': use GPU FP16 (or FP32) when CUDA + artifact exist, else CPU INT8
    if has_cuda:
        if p_fp16 is not None and p_fp16.exists():
            return p_fp16, ["CUDAExecutionProvider", "CPUExecutionProvider"], "fp16"
        if p_fp32.exists():
            return p_fp32, ["CUDAExecutionProvider", "CPUExecutionProvider"], "fp32"
    if p_int8.exists() or not p_fp32.exists():
        return p_int8, ["CPUExecutionProvider"], "int8"
    return p_fp32, ["CPUExecutionProvider"], "fp32"


def refresh_model_config() -> None:
    """Re-evaluates EMBEDDING_MODEL / RERANKER_MODEL paths and EMBEDDING_DIM after model export."""
    global EMBEDDING_MODEL, EMBEDDING_POOLING, RERANKER_MODEL, ONNX_PRECISION
    global \
        EMBEDDING_DIR, \
        EMBEDDING_MANIFEST, \
        EMBEDDING_ONNX_FP32, \
        EMBEDDING_ONNX_FP16, \
        EMBEDDING_ONNX_INT8, \
        EMBEDDING_ONNX, \
        EMBEDDING_ORT_PROVIDERS, \
        EMBEDDING_ACTIVE_PRECISION
    global \
        RERANKER_DIR, \
        RERANKER_MANIFEST, \
        RERANKER_ONNX_FP32, \
        RERANKER_ONNX_FP16, \
        RERANKER_ONNX_INT8, \
        RERANKER_ONNX, \
        RERANKER_ORT_PROVIDERS, \
        RERANKER_ACTIVE_PRECISION
    global EMBEDDING_DIM
    global CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM
    global RERANKER_THRESHOLD, RERANKER_MARGIN

    load_dotenv(ENV_PATH, override=False)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_POOLING = os.getenv("EMBEDDING_POOLING", "auto").strip().lower()
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    ONNX_PRECISION = os.getenv("ONNX_PRECISION", "auto").strip().lower()

    EMBEDDING_DIR = Path(EMBEDDING_ROOT_DIR) / normalize_model_folder_name(EMBEDDING_MODEL)
    EMBEDDING_MANIFEST = EMBEDDING_DIR / "model_manifest.json"
    EMBEDDING_ONNX_FP32 = EMBEDDING_DIR / "model.onnx"
    EMBEDDING_ONNX_FP16 = EMBEDDING_DIR / "model_fp16.onnx"
    EMBEDDING_ONNX_INT8 = EMBEDDING_DIR / "model_int8.onnx"
    (
        EMBEDDING_ONNX,
        EMBEDDING_ORT_PROVIDERS,
        EMBEDDING_ACTIVE_PRECISION,
    ) = resolve_onnx_artifact_and_providers(
        EMBEDDING_ONNX_FP32, EMBEDDING_ONNX_FP16, EMBEDDING_ONNX_INT8, ONNX_PRECISION
    )

    RERANKER_DIR = Path(RERANKER_ROOT_DIR) / normalize_model_folder_name(RERANKER_MODEL)
    RERANKER_MANIFEST = RERANKER_DIR / "model_manifest.json"
    RERANKER_ONNX_FP32 = RERANKER_DIR / "model.onnx"
    RERANKER_ONNX_FP16 = RERANKER_DIR / "model_fp16.onnx"
    RERANKER_ONNX_INT8 = RERANKER_DIR / "model_int8.onnx"
    (
        RERANKER_ONNX,
        RERANKER_ORT_PROVIDERS,
        RERANKER_ACTIVE_PRECISION,
    ) = resolve_onnx_artifact_and_providers(
        RERANKER_ONNX_FP32, RERANKER_ONNX_FP16, RERANKER_ONNX_INT8, ONNX_PRECISION
    )

    EMBEDDING_DIM = resolve_embedding_dim(1024)
    (
        CONFIDENCE_THRESHOLD_HIGH,
        CONFIDENCE_THRESHOLD_MEDIUM,
        RERANKER_THRESHOLD,
        RERANKER_MARGIN,
    ) = resolve_reranker_thresholds(resolve_reranker_arch(), RERANKER_MODEL)


# Module-level model configuration, populated by refresh_model_config()
EMBEDDING_MODEL: str
EMBEDDING_POOLING: str
RERANKER_MODEL: str
ONNX_PRECISION: str

EMBEDDING_DIR: Path
EMBEDDING_MANIFEST: Path
EMBEDDING_ONNX_FP32: Path
EMBEDDING_ONNX_FP16: Path
EMBEDDING_ONNX_INT8: Path
EMBEDDING_ONNX: Path
EMBEDDING_ORT_PROVIDERS: list[str]
EMBEDDING_ACTIVE_PRECISION: str

RERANKER_DIR: Path
RERANKER_MANIFEST: Path
RERANKER_ONNX_FP32: Path
RERANKER_ONNX_FP16: Path
RERANKER_ONNX_INT8: Path
RERANKER_ONNX: Path
RERANKER_ORT_PROVIDERS: list[str]
RERANKER_ACTIVE_PRECISION: str

EMBEDDING_DIM: int
CONFIDENCE_THRESHOLD_HIGH: float
CONFIDENCE_THRESHOLD_MEDIUM: float
RERANKER_THRESHOLD: float
RERANKER_MARGIN: float

refresh_model_config()

GLINER_ONNX_FP32 = GLINER_MODEL / "model.onnx"
GLINER_ONNX = GLINER_ONNX_FP32

# --- BasicType (BT) Classifier Head Selection ---
# Model head type: "arcface" or "logreg"
FOOD_BT_MODEL = os.getenv("FOOD_BT_MODEL", "arcface").lower()
MARKET_BT_MODEL = os.getenv("MARKET_BT_MODEL", "arcface").lower()

FOOD_BT_ARCFACE_INT8 = ARCFACE_DIR / "food_bt_arcface_int8.onnx"
FOOD_BT_ARCFACE_LABELS = ARCFACE_DIR / "food_bt_arcface_labels.json"

MARKET_BT_ARCFACE_INT8 = ARCFACE_DIR / "market_bt_arcface_int8.onnx"
MARKET_BT_ARCFACE_LABELS = ARCFACE_DIR / "market_bt_arcface_labels.json"


def get_bt_model(domain: str) -> str:
    """Returns the configured BT model identifier ('arcface' or 'logreg') for a domain."""
    return FOOD_BT_MODEL if domain == DOMAIN_FOOD else MARKET_BT_MODEL


def get_arcface_dir() -> Path:
    """Returns the effective ArcFace directory, preferring ONNX_DIR/arcface if present, else ONNX_DIR."""
    arcface_sub = Path(ONNX_DIR) / "arcface"
    if arcface_sub.exists():
        return arcface_sub
    return Path(ONNX_DIR)


def get_bt_arcface_onnx_path(domain: str) -> Path:
    """Returns the active ONNX artifact path for the domain's ArcFace BT model (GPU FP16/FP32 or CPU INT8)."""
    arc_dir = get_arcface_dir()
    prefix = "food_bt_arcface" if domain == DOMAIN_FOOD else "market_bt_arcface"
    path, _, _ = resolve_onnx_artifact_and_providers(
        arc_dir / f"{prefix}.onnx",
        arc_dir / f"{prefix}_fp16.onnx",
        arc_dir / f"{prefix}_int8.onnx",
        ONNX_PRECISION,
    )
    return path


def get_bt_arcface_labels_path(domain: str) -> Path:
    """Returns the path to the label mapping JSON for the domain's ArcFace BT model."""
    labels_name = (
        "food_bt_arcface_labels.json" if domain == DOMAIN_FOOD else "market_bt_arcface_labels.json"
    )
    return get_arcface_dir() / labels_name


# --- NER & Entity Extraction ---
MARKET_NER_LABELS = ["brand"]
FOOD_NER_LABELS = ["flavor"]
NER_LABELS = MARKET_NER_LABELS + FOOD_NER_LABELS
NER_CONFIDENCE_THRESHOLD = 0.25


# --- Matching Parameters & Gates ---
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "18"))
# Max rows per ONNX session.run() (embedding, reranker, NER, ArcFace) to bound RAM
ONNX_BATCH_SIZE = int(os.getenv("ONNX_BATCH_SIZE", "32"))
UPSERT_BATCH_SIZE = 256
MATCH_CHUNK_SIZE = 250
CLASSIFY_CHUNK_SIZE = 250

FUZZY_BYPASS_RATIO = 90.0  # Token-sort ratio for bypass
FUZZY_BYPASS_TYPO_RATIO = 80.0  # Character alignment ratio for typo bypass
MATCHER_LOGIC_GATE_CANDIDATES = 5
MATCHER_RERANK_CANDIDATES = int(os.getenv("MATCHER_RERANK_CANDIDATES", "18"))

# --- Template Tag Enrichment ---
ENABLE_TEMPLATE_TAG_ENRICHMENT = True
ENABLE_BRAND_STRIPPED_SEARCH = True
ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = False

# Pipeline statuses that escalate to the classification layer
PIPELINE_ESCALATE_STATUSES = {"Medium Confidence", "Low / Rejected", "Low Confidence", "Rejected"}

# Ice Cream volume-to-form heuristic buckets
IC_TRIGGER_KEYWORD = "ice cream"
IC_BUCKETS = [
    (0, 78, "Stick/Bar"),
    (78, 110, "Cup"),
    (111, 160, "Cone"),
    (450, 10000, "Tub"),
]

# --- Google Sheets Catalog Configuration ---
FOOD_CATALOG_SHEET = "Food Catalog"
MARKET_CATALOG_SHEET = "Market Catalog"

# --- Classifier Constants & Thresholds ---
COL_NAME = "Name"
COL_DESCRIPTION = "Description"
COL_INPUT_CATEGORY = "SellerCategory"
COL_GK = "Generic keywords"
COL_BT = "basictype"

AUTO_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.50
BT_ZERO_SHOT_CONFIDENCE_THRESHOLD = 0.40
BT_DEFAULT_CONFIDENCE_THRESHOLD = 0.50
BT_TRAINED_CONFIDENCE_THRESHOLD = 0.40
GK_TRAINED_CONFIDENCE_THRESHOLD = 0.60
MAX_RERANKED_GK_TAGS = int(os.getenv("MAX_RERANKED_GK_TAGS", "5"))


def get_bt_confidence_threshold(source: str) -> float:
    """Returns minimum confidence required to apply predicted basic-type filter."""
    return (
        BT_ZERO_SHOT_CONFIDENCE_THRESHOLD
        if source == "zero-shot"
        else BT_DEFAULT_CONFIDENCE_THRESHOLD
    )


# --- Hybrid Fusion & Cold-Start Router ---
FUSION_METHOD = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
ALPHA = 0.5  # Dense vs sparse weight when FUSION_METHOD is "weighted"
USE_RERANKER = True
RRF_K = 60
TAG_SEARCH_LIMIT = 50

LIFECYCLE_FEW_SHOT_THRESHOLD = 15  # Sample cutoff separating few-shot from centroid prototypes
COLD_START_TAU = 0.05  # Temperature scaling for centroid cosine similarities
FEW_SHOT_TOP_K = 15
FEW_SHOT_GK_WEIGHT_THRESHOLD = 0.35
ZERO_SHOT_MAX_CANDIDATES = 5

# --- Domain Dish Logic & Training Mining ---
PRIMARY_DISH_TYPES = [
    "fried rice",
    "chop suey rice",
    "chop suey noodles",
    "chop suey",
    "biriyani",
    "kottu",
    "rice and curry",
    "nasi goreng",
    "noodles",
    "fried noodles",
    "pasta",
    "burger",
    "pizza",
    "submarine",
    "wrap",
    "taco",
    "soup",
]
UMBRELLA_MINING_THRESHOLD = 0.80

CLASSIFIER_WEIGHTS = (1.0, 0.8, 0.5)  # (Name, Description, Category)
MATCHER_WEIGHTS = (1.0, 0.3, 0.2)

# --- Schema Mapping Helpers ---
CATALOG_COL_MAP_FOOD = {
    "GenericKeywords": "Generic keywords",
    "BasicType": "basictype",
    "Region": "region",
}
CATALOG_COL_MAP_MARKET = {
    "GenericKeywords": "Generic keywords",
    "BasicType": "basictype",
    "Category": "category",
}


def get_third_tag_col(domain: str) -> str:
    """Returns column name for domain's third classification tag."""
    return "region" if domain == DOMAIN_FOOD else "category"


def resolve_input_category_col(columns) -> str:
    """Returns the raw seller-input category column name if present, avoiding target-label leakage."""
    col_map = {str(c).strip(): str(c) for c in columns}
    for candidate in (COL_INPUT_CATEGORY, "SellerCategory", "seller_category", "Seller_Category"):
        if candidate in col_map:
            return col_map[candidate]
    return ""
