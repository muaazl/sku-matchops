import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# --- Path Configuration (Root & Base) ---
BASE_DIR = Path(__file__).resolve().parent
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

# Suppress harmless third-party library warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

for _warn_pattern in (
    ".*Asking to truncate to max_length.*",
    ".*resume_download is deprecated.*",
    ".*The sentencepiece tokenizer that you are converting.*",
    ".*incorrect regex pattern.*",
):
    warnings.filterwarnings("ignore", message=_warn_pattern)

# --- Data & Cache Paths ---
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = BASE_DIR / "data" / "cache"
STAGING_DIR = DATA_DIR / "staged_sheets"
QDRANT_DATA_DIR = BASE_DIR / "data" / "qdrant"

LOG_DIR = DATA_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
# "json" emits one JSON object per log line (with trace_id); anything else is plain text
LOG_FORMAT = "text"

# --- External Services & Integrations ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# Rule-cache distribution & invalidation (snapshot + version keys + pub/sub)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Engine & Callback Services ---
ENVIRONMENT = "production"
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
TASK_LEASE_SECONDS = float(os.getenv("TASK_LEASE_SECONDS", "30.0"))
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
CACHE_SALT = "MatchOps_v2"

# --- Model Artifacts & Quantization (Strict ONNX + INT8) ---
ONNX_DIR = BASE_DIR / "models"
ARCFACE_DIR = ONNX_DIR / "arcface"
EMBEDDING_ROOT_DIR = ONNX_DIR / "embedding"
RERANKER_ROOT_DIR = ONNX_DIR / "reranker"
GLINER_MODEL = ONNX_DIR / "gliner"


def model_id_to_folder_name(model_id: str) -> str:
    """Normalizes a HuggingFace model ID into a lowercase folder name (model basename only)."""
    clean = (model_id or "").strip().rstrip("/")
    base = clean.split("/")[-1] if "/" in clean else clean
    return base.lower().replace(" ", "-")


def _read_json_manifest(path: Path | str) -> dict:
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
    return _read_json_manifest(EMBEDDING_MANIFEST)


def get_reranker_manifest() -> dict:
    return _read_json_manifest(RERANKER_MANIFEST)


def has_learned_sparse() -> bool:
    """True only when the configured embedding model generates learned sparse weights (BGE-M3)."""
    manifest = get_embedding_manifest()
    if "has_learned_sparse" in manifest:
        return bool(manifest["has_learned_sparse"])
    return "bge-m3" in EMBEDDING_MODEL.lower()


def _resolve_embedding_dim(default: int = 1024) -> int:
    manifest = get_embedding_manifest()
    if "embedding_dim" in manifest:
        try:
            return int(manifest["embedding_dim"])
        except (TypeError, ValueError):
            pass
    return int(os.getenv("EMBEDDING_DIM", str(default)))


def _resolve_reranker_arch() -> str:
    """Returns active reranker architecture ('listwise_lbnl' or 'cross_encoder')."""
    manifest = get_reranker_manifest()
    arch = str(manifest.get("arch_type") or "").strip().lower()
    if arch in ("listwise_lbnl", "cross_encoder"):
        return arch
    return "listwise_lbnl" if "jina-reranker-v3" in RERANKER_MODEL.lower() else "cross_encoder"


def _thresholds_for_reranker_arch(arch: str) -> tuple[float, float, float, float]:
    """
    Returns (conf_high, conf_medium, reranker_threshold, reranker_margin)
    calibrated to the active reranker's score scale.
    """
    if arch == "listwise_lbnl":
        # Listwise LBNL outputs raw cosine similarities (~0.30-0.75 for strong matches).
        return 0.62, 0.35, 0.30, 0.40
    # Standard cross-encoders apply sigmoid to logits -> [0.0, 1.0] probabilities.
    return 0.75, 0.45, 0.45, 0.15


def refresh_model_config() -> None:
    """Re-evaluates EMBEDDING_MODEL / RERANKER_MODEL paths and EMBEDDING_DIM after model export."""
    global EMBEDDING_MODEL, EMBEDDING_POOLING, RERANKER_MODEL
    global \
        EMBEDDING_DIR, \
        EMBEDDING_MANIFEST, \
        EMBEDDING_ONNX_FP32, \
        EMBEDDING_ONNX_INT8, \
        EMBEDDING_ONNX
    global RERANKER_DIR, RERANKER_MANIFEST, RERANKER_ONNX_FP32, RERANKER_ONNX_INT8, RERANKER_ONNX
    global EMBEDDING_DIM
    global CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM
    global RERANKER_THRESHOLD, RERANKER_MARGIN

    load_dotenv(ENV_PATH, override=False)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_POOLING = os.getenv("EMBEDDING_POOLING", "auto").strip().lower()
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    EMBEDDING_DIR = Path(EMBEDDING_ROOT_DIR) / model_id_to_folder_name(EMBEDDING_MODEL)
    EMBEDDING_MANIFEST = EMBEDDING_DIR / "model_manifest.json"
    EMBEDDING_ONNX_FP32 = EMBEDDING_DIR / "model.onnx"
    EMBEDDING_ONNX_INT8 = EMBEDDING_DIR / "model_int8.onnx"
    EMBEDDING_ONNX = EMBEDDING_ONNX_INT8

    RERANKER_DIR = Path(RERANKER_ROOT_DIR) / model_id_to_folder_name(RERANKER_MODEL)
    RERANKER_MANIFEST = RERANKER_DIR / "model_manifest.json"
    RERANKER_ONNX_FP32 = RERANKER_DIR / "model.onnx"
    RERANKER_ONNX_INT8 = RERANKER_DIR / "model_int8.onnx"
    RERANKER_ONNX = RERANKER_ONNX_INT8

    EMBEDDING_DIM = _resolve_embedding_dim(1024)
    (
        CONFIDENCE_THRESHOLD_HIGH,
        CONFIDENCE_THRESHOLD_MEDIUM,
        RERANKER_THRESHOLD,
        RERANKER_MARGIN,
    ) = _thresholds_for_reranker_arch(_resolve_reranker_arch())


# Module-level model configuration, populated by refresh_model_config()
EMBEDDING_MODEL: str
EMBEDDING_POOLING: str
RERANKER_MODEL: str

EMBEDDING_DIR: Path
EMBEDDING_MANIFEST: Path
EMBEDDING_ONNX_FP32: Path
EMBEDDING_ONNX_INT8: Path
EMBEDDING_ONNX: Path

RERANKER_DIR: Path
RERANKER_MANIFEST: Path
RERANKER_ONNX_FP32: Path
RERANKER_ONNX_INT8: Path
RERANKER_ONNX: Path

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
    """Returns the expected INT8 ONNX artifact path for the domain's ArcFace BT model."""
    int8_name = (
        "food_bt_arcface_int8.onnx" if domain == DOMAIN_FOOD else "market_bt_arcface_int8.onnx"
    )
    return get_arcface_dir() / int8_name


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
TOP_K_RETRIEVAL = 30
# Max rows per ONNX session.run() (embedding, reranker, NER, ArcFace) to bound RAM
ONNX_BATCH_SIZE = int(os.getenv("ONNX_BATCH_SIZE", "32"))
UPSERT_BATCH_SIZE = 256
MATCH_CHUNK_SIZE = 250
CLASSIFY_CHUNK_SIZE = 250

FUZZY_BYPASS_RATIO = 90.0  # Token-sort ratio for bypass
FUZZY_BYPASS_TYPO_RATIO = 80.0  # Character alignment ratio for typo bypass
MATCHER_LOGIC_GATE_CANDIDATES = 5
MATCHER_RERANK_CANDIDATES = 30

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
COL_INPUT_CATEGORY = "Category"
COL_GK = "Generic keywords"
COL_BT = "basictype"

AUTO_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.50
BT_ZERO_SHOT_CONFIDENCE_THRESHOLD = 0.40
BT_DEFAULT_CONFIDENCE_THRESHOLD = 0.50
BT_TRAINED_CONFIDENCE_THRESHOLD = 0.40
GK_TRAINED_CONFIDENCE_THRESHOLD = 0.50


def get_bt_confidence_threshold(source: str) -> float:
    """Returns minimum confidence required to apply predicted basic-type filter."""
    return (
        BT_ZERO_SHOT_CONFIDENCE_THRESHOLD
        if source == "zero-shot"
        else BT_DEFAULT_CONFIDENCE_THRESHOLD
    )


# Reranker score calibration values above are derived per active reranker architecture.

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


def get_third_tag_name(domain: str) -> str:
    """Returns display name for domain's third classification tag."""
    return "Region" if domain == DOMAIN_FOOD else "Category"
