import os
import warnings

from dotenv import load_dotenv

load_dotenv()

# --- Resource & Threading Limits ---
# Limit CPU thread pools to half of available cores to avoid resource starvation
MAX_CPU_CORES = max(1, (os.cpu_count() or 4) // 2)
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

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT_DIR, "data"))
CACHE_DIR = os.path.join(DATA_DIR, "cache")
STAGING_DIR = os.path.join(CACHE_DIR, "staged_sheets")
QDRANT_DATA_DIR = os.path.join(DATA_DIR, "qdrant")

DB_DIR = DATA_DIR
DB_PATH = os.path.join(DB_DIR, "sku-matchops.db")
LOG_DIR = os.path.join(DATA_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# --- External Services & Integrations ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "meilimasterkey")

# --- Qdrant Vector DB Settings ---
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 100
QDRANT_INDEXING_THRESHOLD = 10000
QDRANT_MEMMAP_THRESHOLD = 5000
QDRANT_TIMEOUT = 60.0
QDRANT_BATCH_CHUNK_SIZE = 100

# --- Domain & Collection Identifiers ---
DOMAIN_MARKET = "market"
DOMAIN_FOOD = "food"
QDRANT_COLLECTION_MARKET = "market_catalog"
QDRANT_COLLECTION_FOOD = "food_catalog"
MEILI_INDEX_MARKET = "market_catalog"
MEILI_INDEX_FOOD = "food_catalog"
MEILI_INDEX_MARKET_DICTS = "market_dictionaries"
MEILI_INDEX_FOOD_DICTS = "food_dictionaries"
CACHE_SALT = "MatchOps_v2"

# --- Model Artifacts & Quantization ---
ONNX_DIR = os.getenv("MODELS_DIR", os.path.join(ROOT_DIR, "models"))
ARCFACE_DIR = os.path.join(ONNX_DIR, "arcface")

USE_INT8_MODELS = os.getenv("USE_INT8_MODELS", "true").lower() == "true"
CLEANUP_FP32_MODELS = True

BI_ENCODER_MODEL = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
GLINER_MODEL = os.path.join(ONNX_DIR, "gliner")

BI_ENCODER_ONNX_FP32 = os.path.join(ONNX_DIR, "bge_m3", "model.onnx")
BI_ENCODER_ONNX_INT8 = os.path.join(ONNX_DIR, "bge_m3", "model_int8.onnx")
BI_ENCODER_ONNX = (
    BI_ENCODER_ONNX_INT8
    if (USE_INT8_MODELS and os.path.exists(BI_ENCODER_ONNX_INT8))
    else BI_ENCODER_ONNX_FP32
)

CROSS_ENCODER_ONNX_FP32 = os.path.join(ONNX_DIR, "reranker", "model.onnx")
CROSS_ENCODER_ONNX_INT8 = os.path.join(ONNX_DIR, "reranker", "model_int8.onnx")
CROSS_ENCODER_ONNX = (
    CROSS_ENCODER_ONNX_INT8
    if (USE_INT8_MODELS and os.path.exists(CROSS_ENCODER_ONNX_INT8))
    else CROSS_ENCODER_ONNX_FP32
)

GLINER_ONNX = os.path.join(ONNX_DIR, "gliner", "model.onnx")

# --- BasicType (BT) Classifier Head Selection ---
# Model head type: "arcface" or "logreg"
FOOD_BT_MODEL = os.getenv("FOOD_BT_MODEL", "arcface").lower()
MARKET_BT_MODEL = os.getenv("MARKET_BT_MODEL", "arcface").lower()

FOOD_BT_ARCFACE_FP32 = os.path.join(ARCFACE_DIR, "food_bt_arcface.onnx")
FOOD_BT_ARCFACE_INT8 = os.path.join(ARCFACE_DIR, "food_bt_arcface_int8.onnx")
FOOD_BT_ARCFACE_LABELS = os.path.join(ARCFACE_DIR, "food_bt_arcface_labels.json")

MARKET_BT_ARCFACE_FP32 = os.path.join(ARCFACE_DIR, "market_bt_arcface.onnx")
MARKET_BT_ARCFACE_INT8 = os.path.join(ARCFACE_DIR, "market_bt_arcface_int8.onnx")
MARKET_BT_ARCFACE_LABELS = os.path.join(ARCFACE_DIR, "market_bt_arcface_labels.json")


def get_bt_model(domain: str) -> str:
    """Returns the configured BT model identifier ('arcface' or 'logreg') for a domain."""
    return FOOD_BT_MODEL if domain == DOMAIN_FOOD else MARKET_BT_MODEL


def get_arcface_dir() -> str:
    """Returns the effective ArcFace directory, preferring ONNX_DIR/arcface if present, else ONNX_DIR."""
    arcface_sub = os.path.join(ONNX_DIR, "arcface")
    if os.path.exists(arcface_sub):
        return arcface_sub
    return ONNX_DIR


def get_bt_arcface_onnx_path(domain: str) -> str:
    """Returns the expected ONNX artifact path for the domain's ArcFace BT model."""
    int8_name = (
        "food_bt_arcface_int8.onnx" if domain == DOMAIN_FOOD else "market_bt_arcface_int8.onnx"
    )
    fp32_name = "food_bt_arcface.onnx" if domain == DOMAIN_FOOD else "market_bt_arcface.onnx"
    target_dir = get_arcface_dir()
    int8_path = os.path.join(target_dir, int8_name)
    fp32_path = os.path.join(target_dir, fp32_name)

    if USE_INT8_MODELS and os.path.exists(int8_path):
        return int8_path
    if not USE_INT8_MODELS and os.path.exists(fp32_path):
        return fp32_path
    return int8_path if USE_INT8_MODELS else fp32_path


def get_bt_arcface_labels_path(domain: str) -> str:
    """Returns the path to the label mapping JSON for the domain's ArcFace BT model."""
    labels_name = (
        "food_bt_arcface_labels.json" if domain == DOMAIN_FOOD else "market_bt_arcface_labels.json"
    )
    return os.path.join(get_arcface_dir(), labels_name)


# --- NER & Entity Extraction ---
MARKET_NER_LABELS = ["brand"]
FOOD_NER_LABELS = ["flavor"]
NER_LABELS = MARKET_NER_LABELS + FOOD_NER_LABELS
NER_CONFIDENCE_THRESHOLD = 0.25

# --- Matching Parameters & Gates ---
TOP_K_RETRIEVAL = 30
BGE_M3_DENSE_DIM = 1024
EMBED_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 256
MATCH_CHUNK_SIZE = 250
CLASSIFY_CHUNK_SIZE = 250

CONFIDENCE_THRESHOLD_HIGH = 4.0
CONFIDENCE_THRESHOLD_MEDIUM = 0.0
LOGIC_GATE_SIGMOID_SCALE = 0.55  # Calibrated logistic scale: 1 / (1 + exp(-scale * score))
FUZZY_BYPASS_RATIO = 90.0  # Token-sort ratio for bypass
FUZZY_BYPASS_TYPO_RATIO = 80.0  # Character alignment ratio for typo bypass
MATCHER_LOGIC_GATE_CANDIDATES = 5

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


# Cross-encoder logit scoring (raw logits, >0 is considered relevant match)
RERANKER_THRESHOLD = 0.0
RERANKER_MARGIN = 2.5

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
