import logging
from meilisearch.errors import MeilisearchApiError
from engine import config
from engine.data_pipeline.meilisearch_sync import (
    get_meili_client,
    is_meili_healthy,
    sync_dataframe_to_meili,
    sync_dictionaries_to_meili,
)

logger = logging.getLogger("matchops.meilisearch_service")

def setup_indexes():
    """Sets up indexes, primary keys, and configure searchable, filterable, and sortable settings."""
    logger.info("[MEILI] Setting up Meilisearch indexes and settings...")
    client = get_meili_client()
    
    # 1. Setup Catalog Indexes
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

            # Configure pagination limit to support large datasets
            task = index.update_settings({
                "pagination": {
                    "maxTotalHits": 100000
                }
            })
            client.wait_for_task(task.task_uid)

            logger.info(f"[MEILI] Configured settings successfully for '{index_name}'.")
        except Exception as e:
            logger.error(f"[MEILI] Failed to configure settings for '{index_name}': {e}")

    # 2. Setup Dictionaries Indexes
    for dict_index in (config.MEILI_INDEX_MARKET_DICTS, config.MEILI_INDEX_FOOD_DICTS):
        try:
            try:
                index = client.get_index(dict_index)
            except MeilisearchApiError as e:
                if e.code == "index_not_found":
                    logger.info(f"[MEILI] Creating dictionary index '{dict_index}' with primaryKey='id'")
                    task = client.create_index(dict_index, {"primaryKey": "id"})
                    client.wait_for_task(task.task_uid)
                    index = client.get_index(dict_index)
                else:
                    raise e

            task = index.update_searchable_attributes([
                "name", "aliases", "gks", "bt"
            ])
            client.wait_for_task(task.task_uid)

            task = index.update_filterable_attributes([
                "domain", "dataset", "count", "is_weak", "is_meat", "is_vegetable", "is_seafood"
            ])
            client.wait_for_task(task.task_uid)

            task = index.update_sortable_attributes([
                "name", "count", "gk_count", "is_weak", "is_meat", "is_vegetable", "is_seafood"
            ])
            client.wait_for_task(task.task_uid)

            task = index.update_settings({
                "pagination": {
                    "maxTotalHits": 50000
                }
            })
            client.wait_for_task(task.task_uid)

            logger.info(f"[MEILI] Configured dictionary index settings successfully for '{dict_index}'.")
        except Exception as e:
            logger.error(f"[MEILI] Failed to configure settings for '{dict_index}': {e}")


def check_and_sync_meilisearch():
    """
    Called on startup. Checks connection, ensures indexes settings are initialized,
    and seeds indexes from the cached Feather files or SQLite if currently empty.
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
        dict_index_name = config.MEILI_INDEX_MARKET_DICTS if domain == config.DOMAIN_MARKET else config.MEILI_INDEX_FOOD_DICTS
        
        # Check Catalog Index
        try:
            stats = client.index(index_name).get_stats()
            doc_count = stats.number_of_documents
            logger.info(f"[MEILI] Index '{index_name}' contains {doc_count} documents.")
            
            if doc_count == 0:
                logger.info(f"[MEILI] Index '{index_name}' is empty. Seeding from local Feather cache...")
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

        # Check Dictionary Index
        try:
            dict_stats = client.index(dict_index_name).get_stats()
            dict_doc_count = dict_stats.number_of_documents
            logger.info(f"[MEILI] Index '{dict_index_name}' contains {dict_doc_count} documents.")
            if dict_doc_count == 0:
                logger.info(f"[MEILI] Dictionary index '{dict_index_name}' is empty. Seeding from SQLite...")
                sync_dictionaries_to_meili(domain)
        except Exception as dict_err:
            logger.error(f"[MEILI] Error checking/seeding dictionary index '{dict_index_name}': {dict_err}")
