import os

# Dynamically allocate max half of the available CPU cores (minimum 1 core)
cpu_count = os.cpu_count() or 4
MAX_CPU_CORES = max(1, cpu_count // 2)

# Limit CPU thread usage to prevent lagging
os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_CPU_CORES)

import warnings
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

# Suppress sklearn InconsistentVersionWarning for pickled estimators (like TF-IDF)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Suppress known harmless third-party warnings (Transformers / HF Hub)
warnings.filterwarnings("ignore", message=".*Asking to truncate to max_length.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resume_download is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*The sentencepiece tokenizer that you are converting.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*", category=UserWarning)

# --- Path Configuration ---
# BASE_DIR points to the engine package directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT_DIR, "data"))
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(DATA_DIR, "cache"))
STAGING_DIR = os.path.join(CACHE_DIR, "staged_sheets")
QDRANT_DATA_DIR = os.getenv("QDRANT_DATA_DIR", os.path.join(DATA_DIR, "qdrant"))

DB_DIR = DATA_DIR
DB_PATH = os.getenv("DB_PATH", os.path.join(DB_DIR, "sku-matchops.db"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(DATA_DIR, "logs"))
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "16"))
QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "100"))
QDRANT_INDEXING_THRESHOLD = int(os.getenv("QDRANT_INDEXING_THRESHOLD", "10000"))
QDRANT_MEMMAP_THRESHOLD = int(os.getenv("QDRANT_MEMMAP_THRESHOLD", "5000"))

# --- Meilisearch Configuration ---
MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "meilimasterkey")

# --- Database & Domain Constants ---
DOMAIN_MARKET = "market"
DOMAIN_FOOD = "food"
QDRANT_COLLECTION_MARKET = "market_catalog"
QDRANT_COLLECTION_FOOD = "food_catalog"
MEILI_INDEX_MARKET = "market_catalog"
MEILI_INDEX_FOOD = "food_catalog"
MEILI_INDEX_MARKET_DICTS = "market_dictionaries"
MEILI_INDEX_FOOD_DICTS = "food_dictionaries"


# Hash salt for cache validation
CACHE_SALT = "MatchOps_v2"

# --- ONNX Optimized Models ---
ONNX_DIR = os.getenv("MODELS_DIR", os.path.join(ROOT_DIR, "models"))
ARCFACE_DIR = os.path.join(ONNX_DIR, "arcface")

# Toggle for INT8 Quantized models (75% smaller RAM/Disk, 3-4x faster loading/inference)
USE_INT8_MODELS = os.getenv("USE_INT8_MODELS", "true").lower() == "true"
# Toggle to clean up large FP32 ONNX weights after INT8 quantization to reclaim ~4.4 GB disk space
CLEANUP_FP32_MODELS = os.getenv("CLEANUP_FP32_MODELS", "true").lower() == "true"

# --- AI Models ---
BI_ENCODER_MODEL = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
GLINER_MODEL = os.path.join(ONNX_DIR, "gliner")

BI_ENCODER_ONNX_FP32 = os.path.join(ONNX_DIR, "bge_m3", "model.onnx")
BI_ENCODER_ONNX_INT8 = os.path.join(ONNX_DIR, "bge_m3", "model_int8.onnx")
BI_ENCODER_ONNX = BI_ENCODER_ONNX_INT8 if (USE_INT8_MODELS and os.path.exists(BI_ENCODER_ONNX_INT8)) else BI_ENCODER_ONNX_FP32

CROSS_ENCODER_ONNX_FP32 = os.path.join(ONNX_DIR, "reranker", "model.onnx")
CROSS_ENCODER_ONNX_INT8 = os.path.join(ONNX_DIR, "reranker", "model_int8.onnx")
CROSS_ENCODER_ONNX = CROSS_ENCODER_ONNX_INT8 if (USE_INT8_MODELS and os.path.exists(CROSS_ENCODER_ONNX_INT8)) else CROSS_ENCODER_ONNX_FP32

GLINER_ONNX = os.path.join(ONNX_DIR, "gliner", "model.onnx")

# --- BasicType (BT) Classifier Model Toggles & Paths ---
# Options: "arcface" or "logreg" (default: "arcface")
FOOD_BT_MODEL = os.getenv("FOOD_BT_MODEL", "arcface").lower()
MARKET_BT_MODEL = os.getenv("MARKET_BT_MODEL", "arcface").lower()

FOOD_BT_ARCFACE_FP32 = os.path.join(ARCFACE_DIR, "food_bt_arcface.onnx")
FOOD_BT_ARCFACE_INT8 = os.path.join(ARCFACE_DIR, "food_bt_arcface_int8.onnx")
FOOD_BT_ARCFACE_LABELS = os.path.join(ARCFACE_DIR, "food_bt_arcface_labels.json")

MARKET_BT_ARCFACE_FP32 = os.path.join(ARCFACE_DIR, "market_bt_arcface.onnx")
MARKET_BT_ARCFACE_INT8 = os.path.join(ARCFACE_DIR, "market_bt_arcface_int8.onnx")
MARKET_BT_ARCFACE_LABELS = os.path.join(ARCFACE_DIR, "market_bt_arcface_labels.json")

def get_bt_model(domain: str) -> str:
    """Returns the configured BasicType model identifier ('arcface' or 'logreg') for a domain."""
    return FOOD_BT_MODEL if domain == DOMAIN_FOOD else MARKET_BT_MODEL

def get_arcface_dir() -> str:
    """Returns the effective ArcFace directory, preferring ONNX_DIR/arcface if present, else ONNX_DIR."""
    custom = getattr(config, "ARCFACE_DIR", None) if "config" in globals() else None
    if custom and custom != os.path.join(ROOT_DIR, "models", "arcface") and os.path.exists(custom):
        return custom
    arcface_sub = os.path.join(ONNX_DIR, "arcface")
    if os.path.exists(arcface_sub):
        return arcface_sub
    return ONNX_DIR

def get_bt_arcface_onnx_path(domain: str) -> str:
    """Returns the expected ONNX artifact path for the domain's ArcFace BT model."""
    int8_name = "food_bt_arcface_int8.onnx" if domain == DOMAIN_FOOD else "market_bt_arcface_int8.onnx"
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
    """Returns the path to the label encoder mapping JSON for the domain's ArcFace BT model."""
    labels_name = "food_bt_arcface_labels.json" if domain == DOMAIN_FOOD else "market_bt_arcface_labels.json"
    target_dir = get_arcface_dir()
    return os.path.join(target_dir, labels_name)

# --- NER Configuration ---
MARKET_NER_LABELS = [
    "brand"
]
FOOD_NER_LABELS = [
    "flavor",   # All food attributes (chicken, lamb, chocolate, vanilla…) live here
]
NER_LABELS = MARKET_NER_LABELS + FOOD_NER_LABELS  # Combined for initialization

# --- Matching Parameters ---
TOP_K_RETRIEVAL = 30
BGE_M3_DENSE_DIM = 1024
EMBED_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 256
CONFIDENCE_THRESHOLD_HIGH = 4.0
MATCH_CHUNK_SIZE = int(os.getenv("MATCH_CHUNK_SIZE", "250"))
CLASSIFY_CHUNK_SIZE = int(os.getenv("CLASSIFY_CHUNK_SIZE", "250"))

# --- Template Tag Enrichment & Search Configuration ---
# Enable brand/flavor template tag enrichment for Matcher and Classifier pipelines
ENABLE_TEMPLATE_TAG_ENRICHMENT = True
# Enable brand-stripped candidate retrieval in matcher queries
ENABLE_BRAND_STRIPPED_SEARCH = True
# If True, allow new_unregistered tags generated by entity substitution. If False, only keep dictionary-matched tags.
ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS = False


# --- Pipeline Mode: Escalation Threshold ---
# Matcher status values that trigger classifier escalation in "pipeline" task.
# Any result whose status is in this set will also be run through the classifier,
# and the higher-confidence source will be flagged as the winner.
PIPELINE_ESCALATE_STATUSES = {"Medium Confidence", "Low / Rejected", "Low Confidence", "Rejected"}

# --- Ice Cream Logic Gates ---
IC_TRIGGER_KEYWORD = "ice cream"
IC_BUCKETS = [
    (0, 78, "Stick/Bar"),
    (78, 110, "Cup"),
    (111, 160, "Cone"),
    (450, 10000, "Tub")
]


# --- Google Sheets Configuration ---
FOOD_CATALOG_SHEET = "Food Catalog"
MARKET_CATALOG_SHEET = "Market Catalog"

# --- Classifier Constants ---
COL_NAME = "Name"
COL_DESCRIPTION = "Description"
COL_INPUT_CATEGORY = "Category"
COL_GK = "Generic keywords"
COL_BT = "basictype"

AUTO_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.50

BT_ZERO_SHOT_CONFIDENCE_THRESHOLD = 0.40
BT_DEFAULT_CONFIDENCE_THRESHOLD = 0.50

def get_bt_confidence_threshold(source: str) -> float:
    """Minimum confidence required to apply a predicted basic-type filter, by prediction source."""
    return BT_ZERO_SHOT_CONFIDENCE_THRESHOLD if source == "zero-shot" else BT_DEFAULT_CONFIDENCE_THRESHOLD

# Cross-encoder returns raw logits (not probabilities).
# Any positive logit is treated as a relevant match.
# Tune this here without touching tagger.py.
RERANKER_THRESHOLD = 0.0
RERANKER_MARGIN = 2.5  # Max logit drop from top candidate before subsequent candidates are pruned

# Weighted embedding defaults: (Name, Description, Category)
CLASSIFIER_WEIGHTS = (1.0, 0.8, 0.5)  # Classifier needs desc+cat context for disambiguation
MATCHER_WEIGHTS = (1.0, 0.3, 0.2)     # Matcher: embed category with 0.2 weight

# --- Data Normalization Helpers ---
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
    """Returns the internal column name for the domain's third classification tag."""
    return "region" if domain == DOMAIN_FOOD else "category"

def get_third_tag_name(domain: str) -> str:
    """Returns the display name for the domain's third classification tag."""
    return "Region" if domain == DOMAIN_FOOD else "Category"