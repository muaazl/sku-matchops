import logging
import math
import time
import meilisearch
from meilisearch.errors import MeilisearchApiError
import pandas as pd
from engine import config

logger = logging.getLogger("matchops.meilisearch_service")

_client = None

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
        # health() raises communication error if service is unreachable
        health = client.health()
        return health.get("status") == "available"
    except Exception as e:
        logger.debug(f"Meilisearch health check failed: {e}")
        return False

def setup_indexes():
    """Sets up indexes, primary keys, and configure searchable, filterable, and sortable settings."""
    logger.info("[MEILI] Setting up Meilisearch indexes and settings...")
    client = get_meili_client()
    for index_name in (config.MEILI_INDEX_MARKET, config.MEILI_INDEX_FOOD):
        try:
            # Check or create index
            try:
                index = client.get_index(index_name)
            except MeilisearchApiError as e:
                if e.code == "index_not_found":
                    logger.info(f"[MEILI] Creating index '{index_name}' with primaryKey='id'")
                    task = client.create_index(index_name, {"primaryKey": "id"})
                    client.wait_for_task(task.task_uid)
                    index = client.get_index(index_name)
                else:
                    raise e
            
            # Configure searchable attributes
            task = index.update_searchable_attributes([
                "name", "brand", "flavor", "basictype", "bt",
                "category", "region", "description", "gk"
            ])
            client.wait_for_task(task.task_uid)

            # Configure filterable attributes
            task = index.update_filterable_attributes([
                "price", "brand", "flavor", "basictype", "bt",
                "category", "region", "gk"
            ])
            client.wait_for_task(task.task_uid)

            # Configure sortable attributes
            task = index.update_sortable_attributes([
                "name", "brand", "flavor", "price", "basictype", "bt",
                "category", "region", "description"
            ])
            client.wait_for_task(task.task_uid)

            # Configure pagination limit to support large datasets (default is capped at 1000 estimated hits)
            task = index.update_settings({
                "pagination": {
                    "maxTotalHits": 100000
                }
            })
            client.wait_for_task(task.task_uid)

            logger.info(f"[MEILI] Configured settings successfully for '{index_name}'.")
        except Exception as e:
            logger.error(f"[MEILI] Failed to configure settings for '{index_name}': {e}")


def clean_price(val) -> float | None:
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
            logger.warning(f"[MEILI] Could not convert price string '{val}' to float.")
            return None
    return None

def sync_dataframe_to_meili(df: pd.DataFrame, domain: str, chunk_size: int = 5000, clear_existing: bool = True):
    """
    Syncs the reference catalog DataFrame to Meilisearch.
    To avoid excessive RAM usage and timeouts, the DataFrame is indexed in chunks.
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
    
    # Deduplicate DataFrame columns to prevent UserWarning and dropped data
    df_copy = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Ensure a clean, unique string 'id' column without column collisions
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
            
            # Map comma-separated string tags to lists for precise tag filtering in Meilisearch
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
                
                # Extract status & error dynamically (dict vs object)
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


def check_and_sync_meilisearch():
    """
    Called on startup. Checks connection, ensures indexes settings are initialized,
    and seeds indexes from the cached Feather files if currently empty.
    """
    logger.info("[MEILI] Executing startup verification...")
    if not is_meili_healthy():
        logger.warning("[MEILI] Meilisearch is not reachable. Self-healing check bypassed.")
        return
        
    # Standardize indexes settings
    setup_indexes()
    
    client = get_meili_client()
    for domain in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        index_name = config.MEILI_INDEX_MARKET if domain == config.DOMAIN_MARKET else config.MEILI_INDEX_FOOD
        try:
            stats = client.index(index_name).get_stats()
            doc_count = stats.number_of_documents
            logger.info(f"[MEILI] Index '{index_name}' contains {doc_count} documents.")
            
            if doc_count == 0:
                logger.info(f"[MEILI] Index '{index_name}' is empty. Seeding from local Feather cache...")
                # Lazy-import endpoint function to avoid any startup circular dependencies
                from backend.app.services.catalog_service import get_catalog_and_brands
                try:
                    cat_df, _ = get_catalog_and_brands(domain)
                    if cat_df is not None and not cat_df.empty:
                        sync_dataframe_to_meili(cat_df, domain)
                    else:
                        logger.warning(f"[MEILI] No cached Feather catalog found for domain '{domain}' to seed Meilisearch.")
                except Exception as seed_err:
                    logger.error(f"[MEILI] Failed to seed Meilisearch index '{index_name}' from cached file: {seed_err}")
        except Exception as e:
            logger.error(f"[MEILI] Error checking/seeding Meilisearch index '{index_name}': {e}")
