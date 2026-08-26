import os
import sys
import argparse
import logging
import time
import sqlite3
import json
import joblib
import pyarrow.feather as feather
import pandas as pd

# Set up paths so we can import from backend and engine
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("matchops.sync_catalog")

from engine import config
from engine.db import init_db, ensure_db_initialized
from engine.data_pipeline.ingestion import DataIngestion
from engine.nlp.text_cleaner import TextPipeline
from engine.resource_loader import get_pipeline, get_classifier, _get_vector_store

def reset_sqlite_tables(domain: str = None):
    """Drops and recreates SQLite catalog tables for a clean fresh import."""
    conn = sqlite3.connect(config.DB_PATH)
    try:
        if domain and domain in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
            conn.execute("DELETE FROM catalog_items WHERE domain = ?", (domain,))
            conn.execute("DELETE FROM brand_flavors WHERE domain = ?", (domain,))
            conn.execute("DELETE FROM classifier_dictionaries WHERE domain = ?", (domain,))
            conn.execute("DELETE FROM bt_gk_map WHERE domain = ?", (domain,))
            conn.commit()
            logger.info(f"[{domain.upper()}] Deleted existing SQLite table rows for domain '{domain}'.")
        else:
            conn.execute("DROP TABLE IF EXISTS catalog_items;")
            conn.execute("DROP TABLE IF EXISTS brand_flavors;")
            conn.execute("DROP TABLE IF EXISTS classifier_dictionaries;")
            conn.execute("DROP TABLE IF EXISTS bt_gk_map;")
            conn.commit()
            logger.info("Dropped catalog_items, brand_flavors, classifier_dictionaries, and bt_gk_map tables.")
    except Exception as e:
        logger.warning(f"Error while resetting SQLite tables: {e}")
    finally:
        conn.close()

    init_db()

def wipe_all_caches(domains: list = None):
    """Completely purges all disk caches, trained classifier pickles, and feather files."""
    logger.info("Purging all local disk cache files...")
    if os.path.exists(config.CACHE_DIR):
        for item in os.listdir(config.CACHE_DIR):
            item_path = os.path.join(config.CACHE_DIR, item)
            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                except Exception as e:
                    logger.warning(f"Could not remove cache file {item_path}: {e}")

def rebuild_disk_caches(domain: str, sheet_id: str):
    """Rebuilds local disk caches (Feather mmap, BT-GK cache, dictionary JSON, metadata pkl)."""
    logger.info(f"[{domain.upper()}] Rebuilding Feather & pickle disk caches...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    # 1. Load catalog & brands (from SQLite or fallback to Sheets)
    cat_df, brands_df = DataIngestion.load_catalog(sheet_id, domain=domain, force_fetch=False)
    
    # 2. Write Feather files
    catalog_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_mmap.feather")
    brands_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_brands_mmap.feather")
    
    cat_df_feather = cat_df.copy()
    if "entities" in cat_df_feather.columns:
        cat_df_feather = cat_df_feather.drop(columns=["entities"])
    if "weight_val" in cat_df_feather.columns:
        cat_df_feather["weight_val"] = cat_df_feather["weight_val"].astype(str)
    cat_df_feather = cat_df_feather.reset_index(drop=True)
    brands_df_feather = brands_df.reset_index(drop=True)
    
    feather.write_feather(cat_df_feather, catalog_cache_path, compression="lz4")
    feather.write_feather(brands_df_feather, brands_cache_path, compression="lz4")
    logger.info(f"[{domain.upper()}] ✓ Feather caches written ({len(cat_df_feather)} catalog items, {len(brands_df_feather)} brands).")
    
    # 3. Write Classifier Dictionaries JSON
    dicts = DataIngestion.load_classifier_dictionaries(sheet_id, domain=domain, force_fetch=False)
    dicts_cache = os.path.join(config.CACHE_DIR, f"{domain}_classifier_dicts.json")
    with open(dicts_cache, "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2)
    logger.info(f"[{domain.upper()}] ✓ Classifier dictionaries JSON cached ({sum(len(v) for v in dicts.values())} total tags).")

    # 4. Invalidate old classifier pickle so next load trains fresh
    clf_cache = os.path.join(config.CACHE_DIR, f"{domain}_classifier_model.pkl")
    if os.path.exists(clf_cache):
        try:
            os.remove(clf_cache)
            logger.info(f"[{domain.upper()}] ✓ Cleared old classifier pickle to ensure fresh training.")
        except Exception:
            pass

    # 5. Build and save catalog metadata pickle
    metadata_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_metadata.pkl")
    meta_df = cat_df.copy()
    meta_df["clean_text"] = meta_df["Name"].fillna("").astype(str).apply(
        lambda x: TextPipeline.normalize_final(TextPipeline.standardize_units(x))
    )
    meta_df["weight_val"] = meta_df["clean_text"].apply(TextPipeline.extract_weight_feature)
    meta_df["token_count"] = meta_df["clean_text"].fillna("").astype(str).apply(lambda s: len(s.split()))
    meta_df["clean_no_weights"] = meta_df["clean_text"].apply(TextPipeline.strip_weights)
    joblib.dump(meta_df, metadata_path)
    logger.info(f"[{domain.upper()}] ✓ Catalog NLP metadata pickle saved.")

def sync_qdrant_vectors(domain: str, force_reset: bool = False):
    """Fits models, generates dense/sparse embeddings, and syncs vectors to Qdrant."""
    logger.info(f"[{domain.upper()}] Training classifier & Vectorizing catalog into Qdrant...")
    
    if force_reset:
        vs = _get_vector_store()
        try:
            vs.delete_collection(domain)
            logger.info(f"[{domain.upper()}] Deleted existing Qdrant catalog collection for fresh rebuild.")
        except Exception as e:
            logger.warning(f"[{domain.upper()}] Could not delete collection: {e}")

    # 1. Fit classifier and upsert tag embeddings to Qdrant
    clf = get_classifier(domain)
    logger.info(f"[{domain.upper()}] ✓ Classifier trained and tag vectors synced.")

    # 2. Build matcher pipeline (embeds catalog items with BGE-M3 Dense + Sparse and upserts to Qdrant)
    pipe = get_pipeline(domain)
    logger.info(f"[{domain.upper()}] ✓ Match pipeline built and catalog vectors synced to Qdrant.")

def run_sync(
    do_sqlite: bool, 
    do_cache: bool, 
    do_qdrant: bool, 
    target_domains: list, 
    force_reset: bool = False,
    from_staged: bool = False,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
    keep_staged: bool = False
):
    logger.info("=" * 60)
    logger.info("STARTING CATALOG INGESTION & VECTOR SYNC")
    logger.info(f"Operations: SQLite={do_sqlite}, Cache={do_cache}, Qdrant={do_qdrant}")
    logger.info(f"Domains: {[d.upper() for d in target_domains]}, Force Reset={force_reset}")
    logger.info(f"Mode: Sample Mode={from_sample}, From Staged={from_staged}, Keep Staged={keep_staged}")
    logger.info("=" * 60)
    
    sheet_id = config.GOOGLE_SHEET_ID
    if from_sample:
        sheet_id = "SAMPLE_WORKBOOK"
    elif not sheet_id and do_sqlite and not from_staged:
        logger.error("GOOGLE_SHEET_ID is not configured in .env file (or use --sample for offline demo mode).")
        sys.exit(1)
        
    # Ensure database tables exist
    init_db()

    # If sample mode or force reset, purge old caches
    if from_sample or force_reset:
        wipe_all_caches(target_domains)
        reset_sqlite_tables()

    # --- Phase 0: Pre-Fetching & Staging ---
    if from_sample:
        logger.info(f"\n>>> EXTRACTING SAMPLE DATA FROM '{sample_file}' <<<")
        if not os.path.exists(sample_file):
            logger.critical(f"Sample data file not found at '{sample_file}'.")
            sys.exit(1)
        try:
            DataIngestion.stage_all_from_excel(sample_file, domains=target_domains)
            logger.info("✓ All sample sheets successfully extracted into local staging cache.")
        except Exception as e:
            logger.critical(f"✖ Failed during sample data extraction: {e}", exc_info=True)
            sys.exit(1)
    elif do_sqlite and not from_staged:
        logger.info("\n>>> PHASE 0: PRE-FETCHING & STAGING ALL GOOGLE SHEETS <<<")
        try:
            DataIngestion.stage_all_sheets(sheet_id, domains=target_domains)
            logger.info("✓ All Google Sheets successfully downloaded and staged locally.")
            logger.info("✓ Remaining pipeline will execute 100% OFFLINE (immune to network drops).")
        except Exception as e:
            logger.critical(f"✖ Failed during upfront Google Sheets staging: {e}", exc_info=True)
            logger.critical("Aborting sync BEFORE modifying any database tables or caches. Existing database is 100% safe.")
            sys.exit(1)
    elif from_staged:
        logger.info("\n>>> USING PREVIOUSLY STAGED GOOGLE SHEETS (OFFLINE MODE) <<<")

    try:
        for domain in target_domains:
            logger.info(f"\n>>> PROCESSING DOMAIN: {domain.upper()} <<<")
            
            # --- 1. SQLite Database Sync ---
            if do_sqlite:
                logger.info(f"[{domain.upper()}] Step 1: Populating SQLite database from staged data...")
                try:
                    reset_sqlite_tables(domain)
                    DataIngestion.load_catalog(sheet_id, domain=domain, force_fetch=True)
                    DataIngestion.load_classifier_dictionaries(sheet_id, domain=domain, force_fetch=True)
                    DataIngestion.load_bt_gk_map_from_sheets(sheet_id, domain=domain, force_fetch=True)
                    logger.info(f"[{domain.upper()}] ✓ SQLite database successfully populated from staged data.")
                except Exception as e:
                    logger.error(f"[{domain.upper()}] Failed to sync SQLite database: {e}", exc_info=True)
                    sys.exit(1)

            # --- 2. Disk Cache Rebuild ---
            if do_cache:
                logger.info(f"[{domain.upper()}] Step 2: Rebuilding local disk caches (Feather, Dicts, Pickles)...")
                try:
                    rebuild_disk_caches(domain, sheet_id)
                except Exception as e:
                    logger.error(f"[{domain.upper()}] Failed to rebuild disk caches: {e}", exc_info=True)
                    sys.exit(1)

            # --- 3. Qdrant Vectors & Classifier Training ---
            if do_qdrant:
                logger.info(f"[{domain.upper()}] Step 3: Training classifiers & Syncing Qdrant vectors...")
                try:
                    sync_qdrant_vectors(domain, force_reset=force_reset or from_sample)
                except Exception as e:
                    logger.error(f"[{domain.upper()}] Failed to sync Qdrant vectors / train classifiers: {e}", exc_info=True)
                    sys.exit(1)

        # --- Phase 4: Post-Sync Cleanup of Staged Files ---
        if not keep_staged:
            logger.info("\n>>> CLEANING UP TEMPORARY STAGED SHEETS <<<")
            DataIngestion.cleanup_staged_sheets()

        logger.info("\n" + "=" * 60)
        logger.info("CATALOG SYNC & VECTORIZATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as general_err:
        logger.critical(f"Catalog sync encountered an unexpected error: {general_err}", exc_info=True)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="SKU MatchOps Catalog Ingestion, Cache Rebuild & Vector Indexing Script."
    )
    parser.add_argument(
        "--sample",
        dest="from_sample",
        action="store_true",
        help="Perform full reset and ingestion from data/sample/SampleData.xlsx (Offline Demo Mode)."
    )
    parser.add_argument(
        "--sample-file",
        dest="sample_file",
        type=str,
        default="data/sample/SampleData.xlsx",
        help="Path to sample Excel workbook (default: 'data/sample/SampleData.xlsx')."
    )
    parser.add_argument(
        "--sqlite", "--sqlitedb", "--db",
        dest="do_sqlite",
        action="store_true",
        help="Sync SQLite database from Google Sheets. Drops existing domain tables and re-imports fresh."
    )
    parser.add_argument(
        "--cache",
        dest="do_cache",
        action="store_true",
        help="Rebuild all local disk caches (Feather mmap files, BT-GK maps, metadata pickles, dictionary JSONs)."
    )
    parser.add_argument(
        "--qdrant", "--vectors",
        dest="do_qdrant",
        action="store_true",
        help="Vectorize catalog and dictionary tags, then sync/upsert to Qdrant vector database."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "market", "food"],
        help="Target domain to process ('market', 'food', or 'all'). Default: 'all'."
    )
    parser.add_argument(
        "--reset", "--force",
        dest="force_reset",
        action="store_true",
        help="Force complete reset and clean re-embedding of vector collections."
    )
    parser.add_argument(
        "--from-staged", "--offline",
        dest="from_staged",
        action="store_true",
        help="Run sync using existing staged sheet files without downloading from Google Sheets."
    )
    parser.add_argument(
        "--keep-staged",
        dest="keep_staged",
        action="store_true",
        help="Do not delete temporary staged CSV files after sync finishes."
    )
    args = parser.parse_args()

    # If sample mode, run everything fresh
    if args.from_sample:
        do_sqlite = True
        do_cache = True
        do_qdrant = True
        force_reset = True
    else:
        # If no specific operation flags are passed, perform the full workflow
        run_all = not (args.do_sqlite or args.do_cache or args.do_qdrant)
        do_sqlite = args.do_sqlite or run_all
        do_cache = args.do_cache or run_all
        do_qdrant = args.do_qdrant or run_all
        force_reset = args.force_reset

    # Target domains
    if args.domain == "market":
        target_domains = [config.DOMAIN_MARKET]
    elif args.domain == "food":
        target_domains = [config.DOMAIN_FOOD]
    else:
        target_domains = [config.DOMAIN_MARKET, config.DOMAIN_FOOD]

    run_sync(
        do_sqlite=do_sqlite,
        do_cache=do_cache,
        do_qdrant=do_qdrant,
        target_domains=target_domains,
        force_reset=force_reset,
        from_staged=args.from_staged,
        from_sample=args.from_sample,
        sample_file=args.sample_file,
        keep_staged=args.keep_staged
    )

if __name__ == "__main__":
    main()
