#!/usr/bin/env python3
"""
generate_market_keywords.py

Generates search autocomplete generic keywords for Market categories using:
1. Low/medium confidence SKU extraction from SQLite (jobs 578-596).
2. Local BGE-M3 ONNX dense vector embedding & semantic deduplication to isolate distinct products.
3. Dynamic few-shot catalog example mining for targeted categories.
4. Gemini 2.5 Flash with structured Pydantic schema and exponential backoff retry.
5. Deduplication, quality constraint filtering, and master JSON export.
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer

# Configure UTF-8 on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("keyword_gen")

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "sku-matchops.db"
ONNX_BGE_DIR = REPO_ROOT / "models" / "bge_m3"
ONNX_BGE_PATH = ONNX_BGE_DIR / "model_int8.onnx"
EXPORTS_DIR = DATA_DIR / "exports"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic Schemas for Guaranteed Structured JSON Output
# ─────────────────────────────────────────────────────────────────────────────

class KeywordGroup(BaseModel):
    basic_type: str = Field(
        description="The canonical Basic Type / product category (e.g. Tomato, Seer Fish, Lamb Chop, Spring Onion)."
    )
    generic_keywords: List[str] = Field(
        description="Deduplicated list of search autocomplete suggestions combining the base item with attributes, category, and synonyms. Strictly alphanumeric and spaces only."
    )


class KeywordResponse(BaseModel):
    keyword_groups: List[KeywordGroup] = Field(
        description="List of keyword groups categorized by basic_type."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Database Queries: Jobs, SKUs, and Catalog Few-Shots
# ─────────────────────────────────────────────────────────────────────────────

def parse_job_id_range(job_arg: str) -> List[str]:
    """Parses single IDs, comma-separated lists, and ranges like '578-596'."""
    job_ids = []
    tokens = [t.strip() for t in job_arg.split(",") if t.strip()]
    for token in tokens:
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                for i in range(min(start, end), max(start, end) + 1):
                    job_ids.append(str(i))
                continue
        job_ids.append(token.lstrip("#"))
    return sorted(list(set(job_ids)), key=lambda x: int(x) if x.isdigit() else x)


def fetch_low_confidence_skus(db_path: Path, job_ids: List[str], max_conf: float = 0.75) -> List[str]:
    """Fetches non-high-confidence SKUs from processed_skus for the specified jobs."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in job_ids)

    query = f"""
    SELECT DISTINCT sku_name
    FROM processed_skus
    WHERE batch_id IN ({placeholders})
      AND (confidence < ? OR bt IS NULL OR bt = '' OR match_source != 'Matcher')
      AND sku_name IS NOT NULL
      AND length(trim(sku_name)) > 2
    ORDER BY ROWID ASC
    """
    cursor.execute(query, (*job_ids, max_conf))
    rows = cursor.fetchall()
    conn.close()

    raw_skus = [r[0].strip() for r in rows if r[0] and r[0].strip()]
    logger.info(f"Retrieved {len(raw_skus)} candidate SKU rows from jobs {job_ids[0]}..{job_ids[-1]} with confidence < {max_conf}.")
    return raw_skus


def fetch_catalog_few_shots(db_path: Path, categories: List[str], limit_per_cat: int = 7) -> List[Tuple[str, List[str]]]:
    """Fetches 10-20 high quality tagged SKUs from catalog_items across specified categories."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    examples = []

    for cat in categories:
        query = """
        SELECT name, generic_keywords
        FROM catalog_items
        WHERE domain = 'market'
          AND category = ?
          AND basictype IS NOT NULL
          AND length(basictype) > 2
          AND generic_keywords IS NOT NULL
          AND length(generic_keywords) > 5
        GROUP BY basictype
        ORDER BY RANDOM()
        LIMIT ?
        """
        cursor.execute(query, (cat, limit_per_cat))
        for name, gks_raw in cursor.fetchall():
            # Parse generic keywords (comma-separated or JSON)
            if gks_raw.startswith("["):
                try:
                    gk_list = json.loads(gks_raw)
                except Exception:
                    gk_list = [k.strip() for k in gks_raw.split(",") if k.strip()]
            else:
                gk_list = [k.strip() for k in gks_raw.split(",") if k.strip()]
            
            # Clean keywords to alphanumeric and spaces
            cleaned_gks = []
            for k in gk_list:
                clean_k = re.sub(r"[^a-zA-Z0-9\s]", " ", k).strip()
                clean_k = re.sub(r"\s+", " ", clean_k)
                if clean_k and clean_k not in cleaned_gks:
                    cleaned_gks.append(clean_k)
            
            if cleaned_gks:
                examples.append((name, cleaned_gks))

    conn.close()
    logger.info(f"Mined {len(examples)} high-quality catalog few-shot examples across categories: {categories}.")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# 3. Text Normalization & Semantic Vector Deduplication (BGE-M3 ONNX)
# ─────────────────────────────────────────────────────────────────────────────

def clean_sku_text(text: str) -> str:
    """Strips weights, package sizes, Sinhala/non-ASCII characters, and punctuation."""
    if not text:
        return ""

    # Remove Sinhala unicode range and non-ASCII characters
    cleaned = re.sub(r"[\u0D80-\u0DFF]+", " ", text)
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)

    # Standardize multipacks: '2 x 100g' or 'pack of 2'
    cleaned = re.sub(r"\b\d+\s*[xX]\s*\d+\w*\b", " ", cleaned)
    cleaned = re.sub(r"\bpack\s*of\s*\d+\b", " ", cleaned, flags=re.IGNORECASE)

    # Remove measurements and units
    unit_pattern = (
        r"\b\d+(\.\d+)?\s*"
        r"(mg|g|gm|grams?|kg|kilo|kilograms?|ml|l|ltr|liters?|litres?|cl|pcs|pieces?|units?|packet|pkt|bundle|can|tin|pet|pack|slice|steaks?|box)\b"
    )
    cleaned = re.sub(unit_pattern, " ", cleaned, flags=re.IGNORECASE)

    # Remove stand-alone numbers
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)

    # Keep only alphanumeric and spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def vector_deduplicate_skus(
    raw_skus: List[str],
    onnx_path: Path,
    sim_threshold: float = 0.85,
    batch_size: int = 64,
) -> List[str]:
    """
    Uses local BGE-M3 ONNX model to compute dense embeddings and performs
    greedy clustering to isolate distinct products (skipping variants/brand duplicates).
    """
    # 1. Clean SKUs and filter those with insufficient Latin text
    sku_pairs = []
    for raw in raw_skus:
        cleaned = clean_sku_text(raw)
        # Must have at least 3 letters
        if len(re.findall(r"[a-zA-Z]", cleaned)) >= 3:
            sku_pairs.append((raw, cleaned))

    if not sku_pairs:
        logger.warning("No valid SKUs after cleaning.")
        return []

    logger.info(f"Loaded {len(sku_pairs)} SKUs with valid English product text for vector deduplication.")

    # 2. Load BGE-M3 Tokenizer and ONNX Session
    logger.info(f"Loading local BGE-M3 ONNX model from {onnx_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(onnx_path.parent), local_files_only=True, fix_mistral_regex=False
    )
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = min(os.cpu_count() or 4, 8)
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])

    # 3. Compute Dense Embeddings in batches
    clean_texts = [p[1] for p in sku_pairs]
    all_dense = []

    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="np")
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        outputs = session.run(None, ort_inputs)
        dense = outputs[0]  # Shape: (batch_size, 1024)

        # L2 Normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense_norm = dense / np.where(norms == 0, 1e-12, norms)
        all_dense.append(dense_norm)

    dense_matrix = np.vstack(all_dense)  # Shape: (N, 1024)
    logger.info(f"Computed embeddings for {len(clean_texts)} SKUs. Running greedy clustering (threshold = {sim_threshold})...")

    # 4. Greedy Cosine Similarity Clustering
    accepted_indices = []
    accepted_vecs = []

    for idx in range(len(sku_pairs)):
        cand_vec = dense_matrix[idx]
        if not accepted_vecs:
            accepted_indices.append(idx)
            accepted_vecs.append(cand_vec)
            continue

        # Compute dot product against all accepted vectors
        acc_mat = np.array(accepted_vecs)
        sims = np.dot(acc_mat, cand_vec)
        max_sim = float(np.max(sims))

        if max_sim < sim_threshold:
            accepted_indices.append(idx)
            accepted_vecs.append(cand_vec)

    distinct_skus = [sku_pairs[i][0] for i in accepted_indices]
    logger.info(f"Vector deduplication complete: {len(raw_skus)} raw -> {len(distinct_skus)} distinct products.")
    return distinct_skus


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dynamic Prompt Building & Gemini Structured Output
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(few_shots: List[Tuple[str, List[str]]], subgroup: str) -> str:
    """Builds the comprehensive keyword generator system prompt with dynamic catalog examples."""
    examples_text = ""
    for idx, (name, gks) in enumerate(few_shots, 1):
        gks_formatted = json.dumps(gks[:8])
        examples_text += f'*Example {idx}:* "{name}" -> {gks_formatted}\n'

    prompt = f"""You are an E-commerce Search Ops Keyword Generator specializing in the "{subgroup}" market category.
Your task is to build a master list of high-quality search autocomplete suggestions for our catalog.

Combinatorial Rules:
Generic keywords are formed by combining the base item with its attributes (Brand, Cut/Form Factor, Scent/Flavor, Category, Synonyms, and Local Common Names).
*Examples from our tagged catalog:*
{examples_text}

Quality Constraints (CRITICAL):
1. Only output alphanumeric characters and spaces. NO special characters (no hyphens, commas, periods, or slashes inside the keyword).
2. The input SKUs may contain typos or merchant abbreviations. You MUST fix these typos and expand abbreviations in your generated keywords.
3. Include both standard English terms and common localized Sri Lankan names where applicable (e.g., Mukunuwenna, Gotukola, Katta Karawala, Linna, Seer, Ambarella).
4. If you cannot confidently determine the basic type or keywords for an SKU, ignore it. Do not guess.
5. Maximize keyword diversity: provide thorough combinations for each distinct basic_type.

Task:
Below is a raw list of untagged SKUs. Generate a comprehensive, deduplicated list of search keywords. Output strictly in JSON format, grouped by basic_type."""

    return prompt


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
)
def generate_keywords_batch(client, system_prompt: str, sku_batch: List[str]) -> KeywordResponse:
    """Calls Gemini with exponential backoff and guaranteed JSON response schema."""
    batch_text = "\n".join(f"- {sku}" for sku in sku_batch)
    full_content = f"{system_prompt}\n\nRaw SKUs to process:\n{batch_text}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_content,
        config={
            "response_mime_type": "application/json",
            "response_schema": KeywordResponse,
            "temperature": 0.15,
        },
    )
    # Parse into KeywordResponse
    data = json.loads(response.text)
    return KeywordResponse(**data)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Master Aggregation & Post-Processing
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_keyword(keyword: str) -> str:
    """Strips illegal characters, normalizes whitespace, and applies Title Case."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(keyword))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def merge_and_finalize_keywords(
    all_responses: List[KeywordResponse],
) -> Dict[str, any]:
    """Merges keyword groups across batches, deduplicates keywords, and validates counts."""
    master_groups: Dict[str, Set[str]] = {}

    for resp in all_responses:
        for group in resp.keyword_groups:
            bt_clean = sanitize_keyword(group.basic_type)
            if not bt_clean:
                continue

            if bt_clean not in master_groups:
                master_groups[bt_clean] = set()

            for kw in group.generic_keywords:
                clean_kw = sanitize_keyword(kw)
                if clean_kw and len(clean_kw) >= 2:
                    master_groups[bt_clean].add(clean_kw)

    # Format into final sorted structure
    final_keyword_groups = []
    total_unique_keywords = set()

    for bt in sorted(master_groups.keys()):
        kw_list = sorted(list(master_groups[bt]))
        total_unique_keywords.update(kw_list)
        final_keyword_groups.append({
            "basic_type": bt,
            "keyword_count": len(kw_list),
            "generic_keywords": kw_list,
        })

    result = {
        "metadata": {
            "total_basic_types": len(final_keyword_groups),
            "total_unique_keywords": len(total_unique_keywords),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "keyword_groups": final_keyword_groups,
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate search autocomplete keywords for Market categories.")
    parser.add_argument("--job-ids", type=str, default="578-596", help="Job IDs or range (e.g., 578-596)")
    parser.add_argument(
        "--categories",
        type=str,
        default="Fresh Produce, Seafood, Poultry and Meat",
        help="Comma-separated catalog categories",
    )
    parser.add_argument("--subgroup", type=str, default="Fresh", help="Target subgroup name")
    parser.add_argument("--max-conf", type=float, default=0.75, help="Confidence threshold to select low-conf SKUs")
    parser.add_argument("--sim-threshold", type=float, default=0.85, help="Vector cosine similarity clustering threshold")
    parser.add_argument("--batch-size", type=int, default=100, help="SKUs per Gemini call batch")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Path to sqlite3 database")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment or .env file.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"Database not found at: {db_path}")
        sys.exit(1)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = EXPORTS_DIR / f"checkpoints_{args.subgroup.lower().replace(' ', '_')}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    job_ids = parse_job_id_range(args.job_ids)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    logger.info("=" * 60)
    logger.info(f"Target Subgroup : {args.subgroup}")
    logger.info(f"Categories      : {categories}")
    logger.info(f"Jobs            : {job_ids[0]} to {job_ids[-1]} ({len(job_ids)} jobs)")
    logger.info("=" * 60)

    # Step 1: Ingestion
    raw_skus = fetch_low_confidence_skus(db_path, job_ids, max_conf=args.max_conf)
    if not raw_skus:
        logger.error("No SKUs found matching criteria.")
        sys.exit(1)

    # Step 2: Vector Deduplication
    distinct_skus = vector_deduplicate_skus(
        raw_skus,
        onnx_path=ONNX_BGE_PATH,
        sim_threshold=args.sim_threshold,
    )
    logger.info(f"Proceeding with {len(distinct_skus)} distinct representative product SKUs.")

    # Step 3: Mine Catalog Few-Shots
    few_shots = fetch_catalog_few_shots(db_path, categories, limit_per_cat=6)
    system_prompt = build_system_prompt(few_shots, args.subgroup)

    # Step 4: Batch Processing with Gemini & State Persistence
    all_responses: List[KeywordResponse] = []
    total_skus = len(distinct_skus)

    for i in range(0, total_skus, args.batch_size):
        batch_idx = i // args.batch_size
        batch_skus = distinct_skus[i : i + args.batch_size]
        checkpoint_file = checkpoints_dir / f"batch_{batch_idx}.json"

        # Check if batch was already processed in a previous run
        if checkpoint_file.exists():
            logger.info(f"[Batch {batch_idx + 1}] Found existing checkpoint at {checkpoint_file.name}. Loading...")
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)
                    all_responses.append(KeywordResponse(**batch_data))
                    continue
            except Exception as e:
                logger.warning(f"Error reading checkpoint {checkpoint_file}: {e}. Reprocessing...")

        logger.info(f"[Batch {batch_idx + 1}/{(total_skus + args.batch_size - 1) // args.batch_size}] Generating keywords for {len(batch_skus)} SKUs...")
        try:
            resp = generate_keywords_batch(client, system_prompt, batch_skus)
            all_responses.append(resp)

            # Persist checkpoint immediately
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(resp.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"[Batch {batch_idx + 1}] Successfully saved checkpoint ({len(resp.keyword_groups)} basic types).")
        except Exception as e:
            logger.error(f"[Batch {batch_idx + 1}] Failed after retries: {e}")
            raise e

        time.sleep(2)  # Respect API rate limits

    # Step 5: Merge, Deduplicate, and Output Master JSON
    master_result = merge_and_finalize_keywords(all_responses)
    output_file = EXPORTS_DIR / f"master_keywords_{args.subgroup.lower().replace(' ', '_')}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(master_result, f, indent=2, ensure_ascii=False)

    meta = master_result["metadata"]
    logger.info("=" * 60)
    logger.info("SUCCESSFULLY COMPLETED KEYWORD GENERATION PIPELINE")
    logger.info(f"Total Basic Types Identified : {meta['total_basic_types']}")
    logger.info(f"Total Unique Keywords Created : {meta['total_unique_keywords']}")
    logger.info(f"Master Output File           : {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
