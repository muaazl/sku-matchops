import logging
import math
import time
from typing import Optional
import pandas as pd
import meilisearch
from meilisearch.errors import MeilisearchApiError
from engine import config
from engine.data_pipeline.cache_manager import clean_price

logger = logging.getLogger("matchops.meilisearch_sync")

_client: Optional[meilisearch.Client] = None

def get_meili_client() -> meilisearch.Client:
    """Returns the singleton Meilisearch client instance."""
    global _client
    if _client is None:
        _client = meilisearch.Client(config.MEILI_URL, config.MEILI_MASTER_KEY)
    return _client

def is_meili_healthy() -> bool:
    """Checks if the Meilisearch service is available and responding."""
    try:
        client = get_meili_client()
        health = client.health()
        return health.get("status") == "available"
    except Exception as e:
        logger.debug(f"Meilisearch health check failed: {e}")
        return False


def sync_dataframe_to_meili(df: pd.DataFrame, domain: str, chunk_size: int = 5000, clear_existing: bool = True):
    """
    Syncs the reference catalog DataFrame to Meilisearch in chunks.
    """
    if not is_meili_healthy():
        logger.warning(f"[MEILI] Meilisearch is not available. Skipping sync for domain '{domain}'.")
        return
    
    index_name = config.MEILI_INDEX_MARKET if domain == config.DOMAIN_MARKET else config.MEILI_INDEX_FOOD
    client = get_meili_client()
    index = client.index(index_name)
    
    if clear_existing:
        try:
            task = index.delete_all_documents()
            client.wait_for_task(task.task_uid)
            logger.info(f"[MEILI] Cleared existing documents in index '{index_name}' before sync.")
        except Exception as clear_err:
            logger.warning(f"[MEILI] Failed to clear documents in index '{index_name}': {clear_err}")
    
    df_copy = df.loc[:, ~df.columns.duplicated()].copy()
    if "id" in df_copy.columns and df_copy["id"].notna().any():
        df_copy["id"] = df_copy["id"].astype(str)
    else:
        df_copy["id"] = [str(idx) for idx in range(len(df_copy))]
    
    total_rows = len(df_copy)
    logger.info(f"[MEILI] Syncing {total_rows} rows to Meilisearch index '{index_name}' in chunks of {chunk_size}...")
    
    num_chunks = math.ceil(total_rows / chunk_size)
    for i in range(num_chunks):
        chunk_df = df_copy.iloc[i * chunk_size : (i + 1) * chunk_size]
        records = chunk_df.to_dict(orient="records")
        
        documents = []
        for r in records:
            doc = {
                "id": str(r.get("id")),
                "name": None if pd.isna(r.get("Name")) else r.get("Name"),
                "brand": None if pd.isna(r.get("Brand")) else r.get("Brand"),
                "price": clean_price(r.get("Price")),
                "sellercategory": None if pd.isna(r.get("SellerCategory")) else r.get("SellerCategory"),
                "category": None if pd.isna(r.get("category")) else r.get("category"),
                "gk": None if pd.isna(r.get("Generic keywords")) else r.get("Generic keywords"),
                "bt": None if pd.isna(r.get("basictype")) else r.get("basictype"),
                "merchant": None if pd.isna(r.get("Merchant")) else r.get("Merchant"),
                "flavor": None if pd.isna(r.get("Flavor")) else r.get("Flavor"),
                "description": None if pd.isna(r.get("Description")) else r.get("Description"),
                "region": None if pd.isna(r.get("region")) else r.get("region"),
            }
            
            for field in ("gk", "bt", "category", "region", "brand", "flavor"):
                val = doc[field]
                if isinstance(val, str):
                    doc[field] = [item.strip() for item in val.split(",") if item.strip()]
                elif val is None:
                    doc[field] = []
                    
            documents.append(doc)
            
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                task = index.add_documents(documents)
                task_info = client.wait_for_task(task.task_uid)
                t_status = task_info.get("status") if isinstance(task_info, dict) else getattr(task_info, "status", None)
                t_error = task_info.get("error") if isinstance(task_info, dict) else getattr(task_info, "error", None)
                
                if t_status == "failed":
                    raise Exception(f"Meilisearch indexing task failed: {t_error}")
                    
                logger.info(f"[MEILI] Sent chunk {i+1}/{num_chunks} ({len(documents)} documents) to index '{index_name}'.")
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"[MEILI] Attempt {attempt}/{max_retries} failed for chunk {i+1}/{num_chunks} in index '{index_name}': {e}. Retrying in {attempt * 2}s...")
                    time.sleep(attempt * 2)
                else:
                    logger.error(f"[MEILI] Failed to index chunk {i+1}/{num_chunks} in index '{index_name}' after {max_retries} attempts: {e}")
                    raise e
            
    logger.info(f"[MEILI] Domain '{domain}' synced successfully to Meilisearch index '{index_name}'!")


def sync_dictionaries_to_meili(domain: str, clear_existing: bool = True):
    """
    Syncs classifier dictionaries, brands, and BT-GK map with precomputed counts
    from SQLite into the Meilisearch dictionary index.
    """
    if not is_meili_healthy():
        logger.warning(f"[MEILI] Meilisearch is not available. Skipping dictionary sync for '{domain}'.")
        return

    from engine.core.db import ensure_db_initialized
    conn = ensure_db_initialized()
    cursor = conn.cursor()

    dict_index_name = config.MEILI_INDEX_MARKET_DICTS if domain == config.DOMAIN_MARKET else config.MEILI_INDEX_FOOD_DICTS
    client = get_meili_client()
    index = client.index(dict_index_name)

    if clear_existing:
        try:
            task = index.delete_all_documents()
            client.wait_for_task(task.task_uid)
            logger.info(f"[MEILI] Cleared existing documents in dictionary index '{dict_index_name}'.")
        except Exception as clear_err:
            logger.warning(f"[MEILI] Failed to clear documents in '{dict_index_name}': {clear_err}")

    documents = []

    # 1. Tags from classifier_dictionaries (gk, bt, category/region)
    cursor.execute("SELECT id, tag_type, tag, catalog_count FROM classifier_dictionaries WHERE domain = ?", (domain,))
    for row_id, tag_type, tag, count in cursor.fetchall():
        dataset_name = "category" if tag_type in ("category", "region") else tag_type
        documents.append({
            "id": f"{domain}_{tag_type}_{row_id}",
            "domain": domain,
            "dataset": dataset_name,
            "tag_type": tag_type,
            "name": tag,
            "count": count or 0,
            "aliases": "",
            "gks": "",
            "bt": "",
            "gk_count": 0,
            "is_weak": False,
            "is_meat": False,
            "is_vegetable": False,
            "is_seafood": False,
        })

    # 2. Brands/Flavors from brand_flavors
    cursor.execute("SELECT id, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, catalog_count FROM brand_flavors WHERE domain = ?", (domain,))
    for row_id, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, count in cursor.fetchall():
        documents.append({
            "id": f"{domain}_brand_{row_id}",
            "domain": domain,
            "dataset": "brands",
            "tag_type": "brand" if domain == config.DOMAIN_MARKET else "flavor",
            "name": name,
            "brand_name": name,
            "flavor_name": name,
            "count": count or 0,
            "aliases": aliases or "",
            "gks": "",
            "bt": "",
            "gk_count": 0,
            "is_weak": bool(is_weak),
            "is_meat": bool(is_meat),
            "is_vegetable": bool(is_vegetable),
            "is_seafood": bool(is_seafood),
        })

    # 3. BT-GK Map from bt_gk_map
    cursor.execute("SELECT id, basictype, generic_keywords, gk_count, catalog_count FROM bt_gk_map WHERE domain = ?", (domain,))
    for row_id, basictype, gks, gk_count, count in cursor.fetchall():
        documents.append({
            "id": f"{domain}_map_{row_id}",
            "domain": domain,
            "dataset": "bt_gk_map",
            "tag_type": "bt_gk_map",
            "name": basictype,
            "bt": basictype,
            "gks": gks or "",
            "gk_count": gk_count or 0,
            "count": count or 0,
            "aliases": "",
            "is_weak": False,
            "is_meat": False,
            "is_vegetable": False,
            "is_seafood": False,
        })

    conn.close()

    total_docs = len(documents)
    logger.info(f"[MEILI] Syncing {total_docs} dictionary items to '{dict_index_name}'...")

    chunk_size = 5000
    for i in range(0, total_docs, chunk_size):
        chunk = documents[i : i + chunk_size]
        task = index.add_documents(chunk)
        client.wait_for_task(task.task_uid)

    logger.info(f"[MEILI] Successfully synced {total_docs} dictionary documents to index '{dict_index_name}'.")
