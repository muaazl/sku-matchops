import logging
import math
import time
from typing import Optional
import pandas as pd
import meilisearch
from meilisearch.errors import MeilisearchApiError

from engine import config

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

def clean_price(val) -> Optional[float]:
    """Safely cleans and converts a price value to a float."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_str = val.strip().replace("$", "").replace(",", "")
        if not val_str:
            return None
        try:
            return float(val_str)
        except ValueError:
            return None
    return None

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
                    time.sleep(attempt * 2)
                else:
                    logger.error(f"[MEILI] Failed to index chunk {i+1}/{num_chunks} in index '{index_name}': {e}")
                    raise e
            
    logger.info(f"[MEILI] Domain '{domain}' synced successfully to Meilisearch index '{index_name}'!")
