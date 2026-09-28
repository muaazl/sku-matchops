"""
SKU MatchOps - Catalog Sync Script

Keeps three independent things up to date, each with an explicit mode:

  --db      SQLite (catalog_items, brand_flavors, classifier_dictionaries, bt_gk_map) —
            local to this checkout, populated from Google Sheets.
  --cache   Local disk cache files (Feather mmap, BT-GK pickle, dictionary JSON,
            catalog metadata pickle) — local to this checkout, derived from --db.
  --qdrant  Qdrant vector collections (catalog + tag embeddings) and classifier training —
            this is the piece that is typically SHARED across multiple checkouts pointed
            at the same Qdrant/Meilisearch instance.

Each target takes a mode:
  sync     Incremental — only touches what actually changed/is missing. Safe default.
  rebuild  Wipes that target and regenerates it from scratch. Use when you don't trust
           the current state, or after a very large catalog change.

Examples:
  python -m scripts.catalog.sync                          # sync everything (default)
  python -m scripts.catalog.sync --qdrant rebuild          # only rebuild Qdrant + classifier
  python -m scripts.catalog.sync --db rebuild --cache sync # rebuild SQLite, sync local cache
  python -m scripts.catalog.sync --sample                 # offline demo mode (full rebuild)

Why --qdrant sync is safe to run from a different checkout pointed at the same shared
Qdrant instance: it reconciles this checkout's local "already embedded" bookkeeping
against Qdrant's own actual contents before deciding what (if anything) needs embedding,
instead of trusting only this checkout's own history.
"""
import os
import sys
import argparse
import json
import logging
import sqlite3
import joblib
import pyarrow.feather as feather

# Set up paths so we can import from backend and engine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ENGINE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("matchops.sync_catalog")

from engine.core import config
from engine.core.db import init_db, ensure_db_initialized, clear_db_cache
from engine.data_pipeline.ingestion import DataIngestion
from engine.nlp.text_cleaner import TextPipeline
from engine.core.resource_loader import get_pipeline, get_classifier, _get_vector_store


def reset_sqlite_tables(domain: str = None):
    """Drops and recreates SQLite catalog tables for a clean fresh import."""
    init_db(force=False)
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
            clear_db_cache()
    except Exception as e:
        logger.warning(f"Error while resetting SQLite tables: {e}")
    finally:
        conn.close()

    init_db(force=True)


def wipe_local_cache_files():
    """Purges all local disk cache files (Feather, pickles, dictionary JSON, hash files)."""
    logger.info("Purging local disk cache files...")
    if os.path.exists(config.CACHE_DIR):
        for item in os.listdir(config.CACHE_DIR):
            item_path = os.path.join(config.CACHE_DIR, item)
            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                except Exception as e:
                    logger.warning(f"Could not remove cache file {item_path}: {e}")


# --- Target 1: --db -----------------------------------------------------------------

def sync_db(domain: str, sheet_id: str, mode: str):
    """
    sync:    Incrementally imports Sheets -> SQLite. Existing DataIngestion logic already
             diffs by row_hash/db_uid and only inserts/updates/deletes what actually changed.
    rebuild: Wipes this domain's SQLite rows first, then does a full fresh import.
    """
    if mode == "rebuild":
        reset_sqlite_tables(domain)
    else:
        init_db(force=False)

    cat_df, _ = DataIngestion.load_catalog(sheet_id, domain=domain, force_fetch=True)
    DataIngestion.load_classifier_dictionaries(sheet_id, domain=domain, force_fetch=True)
    DataIngestion.load_bt_gk_map_from_sheets(sheet_id, domain=domain, force_fetch=True, cat_df=cat_df)
    DataIngestion.compute_and_store_dictionary_counts(domain, cat_df=cat_df)

    try:
        from engine.data_pipeline.meilisearch_sync import sync_dictionaries_to_meili
        sync_dictionaries_to_meili(domain)
        logger.info(f"[{domain.upper()}] Meilisearch dictionary indexes updated.")
    except Exception as e:
        logger.warning(f"[{domain.upper()}] Could not sync dictionaries to Meilisearch: {e}")

    return cat_df


# --- Target 2: --cache ---------------------------------------------------------------

def sync_cache(domain: str, sheet_id: str, mode: str):
    """
    sync:    Rebuilds only whichever local cache artifacts are missing; reuses the rest.
    rebuild: Regenerates every local cache artifact unconditionally.

    Scope is deliberately local-disk-only (Feather mmap, BT-GK pickle, dictionary JSON,
    catalog metadata pickle). The classifier model and its Qdrant tag-vector upsert are
    handled under --qdrant instead, since training always also pushes tag embeddings to
    Qdrant — that pairing doesn't split cleanly into a "cache-only" step.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cat_df, brands_df = DataIngestion.load_catalog(sheet_id, domain=domain, force_fetch=False)

    catalog_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_mmap.feather")
    brands_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_brands_mmap.feather")
    dicts_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_classifier_dicts.json")
    metadata_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_metadata.pkl")

    rebuild = mode == "rebuild"

    if rebuild or not os.path.exists(catalog_cache_path) or not os.path.exists(brands_cache_path):
        cat_df_feather = cat_df.copy()
        if "entities" in cat_df_feather.columns:
            cat_df_feather = cat_df_feather.drop(columns=["entities"])
        if "weight_val" in cat_df_feather.columns:
            cat_df_feather["weight_val"] = cat_df_feather["weight_val"].astype(str)
        cat_df_feather = cat_df_feather.reset_index(drop=True)
        brands_df_feather = brands_df.reset_index(drop=True)

        feather.write_feather(cat_df_feather, catalog_cache_path, compression="lz4")
        feather.write_feather(brands_df_feather, brands_cache_path, compression="lz4")
        logger.info(f"[{domain.upper()}] Feather caches written ({len(cat_df_feather)} catalog items, {len(brands_df_feather)} brands).")
    else:
        logger.info(f"[{domain.upper()}] Feather caches already present — skipped.")

    if rebuild or not os.path.exists(dicts_cache_path):
        dicts = DataIngestion.load_classifier_dictionaries(sheet_id, domain=domain, force_fetch=False)
        with open(dicts_cache_path, "w", encoding="utf-8") as f:
            json.dump(dicts, f, indent=2)
        logger.info(f"[{domain.upper()}] Classifier dictionaries JSON cached.")
    else:
        logger.info(f"[{domain.upper()}] Classifier dictionaries JSON already present — skipped.")

    if rebuild or not os.path.exists(metadata_path):
        meta_df = cat_df.copy()
        meta_df["clean_text"] = meta_df["Name"].fillna("").astype(str).apply(
            lambda x: TextPipeline.normalize_final(TextPipeline.standardize_units(x))
        )
        meta_df["weight_val"] = meta_df["clean_text"].apply(TextPipeline.extract_weight_feature)
        meta_df["token_count"] = meta_df["clean_text"].fillna("").astype(str).apply(lambda s: len(s.split()))
        meta_df["clean_no_weights"] = meta_df["clean_text"].apply(TextPipeline.strip_weights)
        joblib.dump(meta_df, metadata_path)
        logger.info(f"[{domain.upper()}] Catalog NLP metadata pickle saved.")
    else:
        logger.info(f"[{domain.upper()}] Catalog metadata pickle already present — skipped.")


# --- Target 3: --qdrant ---------------------------------------------------------------

def sync_qdrant(domain: str, mode: str):
    """
    sync:    Reconciles this checkout's local hash bookkeeping against Qdrant's actual
             contents (see VectorStore.get_existing_hashes), then does an incremental
             catalog embed/upsert (unchanged rows are skipped) and trains the classifier
             only if its cached model is missing.
    rebuild: Deletes this domain's Qdrant collections outright and re-embeds/re-trains
             everything from scratch.
    """
    vs = _get_vector_store()
    do_full_rebuild = mode == "rebuild"

    if do_full_rebuild:
        for col in [vs._get_collection_name(domain), f"{domain}_tags"]:
            try:
                if vs.client.collection_exists(col):
                    vs.client.delete_collection(col)
                    logger.info(f"[{domain.upper()}] Deleted existing Qdrant collection '{col}' for a clean rebuild.")
            except Exception as e:
                logger.warning(f"[{domain.upper()}] Could not delete collection '{col}': {e}")
    else:
        try:
            remote_hashes = vs.get_existing_hashes(domain)
        except Exception as e:
            logger.warning(f"[{domain.upper()}] Could not fetch existing hashes from Qdrant (continuing with local state only): {e}")
            remote_hashes = {}

        if remote_hashes:
            hash_file = os.path.join(config.CACHE_DIR, f"{domain}_row_hashes.json")
            local_hashes = {}
            if os.path.exists(hash_file):
                try:
                    with open(hash_file, "r") as f:
                        local_hashes = json.load(f)
                except Exception:
                    local_hashes = {}
            # Local entries (this checkout's own most recent sync) win on conflict;
            # Qdrant fills in anything this checkout doesn't know about yet.
            merged = {**remote_hashes, **local_hashes}
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            with open(hash_file, "w") as f:
                json.dump(merged, f)
            logger.info(
                f"[{domain.upper()}] Reconciled {len(remote_hashes)} row hashes from Qdrant's actual "
                f"contents into local sync state ({len(merged)} total known rows)."
            )

    # A missing classifier pickle means training is needed regardless of whether the catalog
    # embeddings themselves are up to date — independent of whether a full re-embed is needed.
    clf_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_classifier_model.pkl")
    need_classifier_train = do_full_rebuild or not os.path.exists(clf_cache_path)

    logger.info(f"[{domain.upper()}] Training classifier & syncing catalog vectors to Qdrant...")
    get_classifier(domain, force_reset=need_classifier_train)
    # force_sync forces a full catalog re-embed — only ever set for an explicit rebuild.
    # A merely-missing classifier pickle must NOT force re-embedding the whole catalog.
    get_pipeline(domain, check_for_updates=True, force_sync=do_full_rebuild)
    logger.info(f"[{domain.upper()}] Qdrant sync complete.")


# --- Orchestration ---------------------------------------------------------------------

def run(
    db_mode: str,
    cache_mode: str,
    qdrant_mode: str,
    target_domains: list,
    from_staged: bool = False,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
    keep_staged: bool = False,
):
    logger.info("=" * 60)
    logger.info("SKU MATCHOPS CATALOG SYNC")
    logger.info(f"--db={db_mode or 'skip'}  --cache={cache_mode or 'skip'}  --qdrant={qdrant_mode or 'skip'}")
    logger.info(f"Domains: {[d.upper() for d in target_domains]}")
    logger.info("=" * 60)

    sheet_id = config.GOOGLE_SHEET_ID
    if from_sample:
        sheet_id = "SAMPLE_WORKBOOK"
    elif not sheet_id and db_mode and not from_staged:
        logger.error("GOOGLE_SHEET_ID is not configured in .env file (or use --sample for offline demo mode).")
        sys.exit(1)

    init_db()

    if from_sample:
        wipe_local_cache_files()
        reset_sqlite_tables()
        logger.info(f"\n>>> EXTRACTING SAMPLE DATA FROM '{sample_file}' <<<")
        if not os.path.exists(sample_file):
            logger.critical(f"Sample data file not found at '{sample_file}'.")
            sys.exit(1)
        try:
            DataIngestion.stage_all_from_excel(sample_file, domains=target_domains)
        except Exception as e:
            logger.critical(f"Failed during sample data extraction: {e}", exc_info=True)
            sys.exit(1)
        db_mode = cache_mode = qdrant_mode = "rebuild"
    elif db_mode and not from_staged:
        logger.info("\n>>> PRE-FETCHING & STAGING GOOGLE SHEETS <<<")
        try:
            DataIngestion.stage_all_sheets(sheet_id, domains=target_domains)
            logger.info("Staged locally — remaining pipeline runs 100% offline.")
        except Exception as e:
            logger.critical(f"Failed during Google Sheets staging: {e}", exc_info=True)
            logger.critical("Aborting before modifying any database tables or caches.")
            sys.exit(1)
    elif from_staged:
        logger.info("\n>>> USING PREVIOUSLY STAGED GOOGLE SHEETS (OFFLINE MODE) <<<")

    try:
        for domain in target_domains:
            logger.info(f"\n>>> DOMAIN: {domain.upper()} <<<")

            if db_mode:
                logger.info(f"[{domain.upper()}] --db ({db_mode})...")
                try:
                    sync_db(domain, sheet_id, db_mode)
                except Exception as e:
                    logger.error(f"[{domain.upper()}] --db failed: {e}", exc_info=True)
                    sys.exit(1)

            if cache_mode:
                logger.info(f"[{domain.upper()}] --cache ({cache_mode})...")
                try:
                    sync_cache(domain, sheet_id, cache_mode)
                except Exception as e:
                    logger.error(f"[{domain.upper()}] --cache failed: {e}", exc_info=True)
                    sys.exit(1)

            if qdrant_mode:
                logger.info(f"[{domain.upper()}] --qdrant ({qdrant_mode})...")
                try:
                    from scripts.ml.export_onnx import export_all_models_if_needed
                    export_all_models_if_needed()
                except Exception as exp_err:
                    logger.warning(f"ONNX export verification check: {exp_err}")

                try:
                    sync_qdrant(domain, qdrant_mode)
                except Exception as e:
                    logger.error(f"[{domain.upper()}] --qdrant failed: {e}", exc_info=True)
                    sys.exit(1)

        if not keep_staged:
            logger.info("\n>>> CLEANING UP TEMPORARY STAGED SHEETS <<<")
            DataIngestion.cleanup_staged_sheets()

        logger.info("\n" + "=" * 60)
        logger.info("CATALOG SYNC COMPLETE.")
        logger.info("=" * 60)

    except Exception as general_err:
        logger.critical(f"Catalog sync encountered an unexpected error: {general_err}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SKU MatchOps Catalog Sync: --db / --cache / --qdrant, each 'sync' or 'rebuild'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", choices=["sync", "rebuild"], default=None,
                         help="SQLite catalog tables. sync=incremental import, rebuild=wipe+reimport.")
    parser.add_argument("--cache", choices=["sync", "rebuild"], default=None,
                         help="Local disk cache files. sync=fill in what's missing, rebuild=regenerate all.")
    parser.add_argument("--qdrant", choices=["sync", "rebuild"], default=None,
                         help="Qdrant vectors + classifier training. sync=incremental (reconciled against "
                              "Qdrant's actual contents), rebuild=wipe collections + retrain from scratch.")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "market", "food"],
                         help="Target domain to process. Default: 'all'.")
    parser.add_argument("--sample", dest="from_sample", action="store_true",
                         help="Offline demo mode: full rebuild of everything from data/sample/SampleData.xlsx.")
    parser.add_argument("--sample-file", dest="sample_file", type=str, default="data/sample/SampleData.xlsx",
                         help="Path to sample Excel workbook (default: 'data/sample/SampleData.xlsx').")
    parser.add_argument("--from-staged", "--offline", dest="from_staged", action="store_true",
                         help="Use previously staged sheet files instead of downloading from Google Sheets.")
    parser.add_argument("--keep-staged", dest="keep_staged", action="store_true",
                         help="Don't delete temporary staged CSV files after sync finishes.")
    args = parser.parse_args()

    if args.from_sample:
        db_mode = cache_mode = qdrant_mode = "rebuild"
    else:
        # No target flags at all -> sync everything (safe default).
        if not (args.db or args.cache or args.qdrant):
            db_mode = cache_mode = qdrant_mode = "sync"
        else:
            db_mode, cache_mode, qdrant_mode = args.db, args.cache, args.qdrant

    if args.domain == "market":
        target_domains = [config.DOMAIN_MARKET]
    elif args.domain == "food":
        target_domains = [config.DOMAIN_FOOD]
    else:
        target_domains = [config.DOMAIN_MARKET, config.DOMAIN_FOOD]

    run(
        db_mode=db_mode,
        cache_mode=cache_mode,
        qdrant_mode=qdrant_mode,
        target_domains=target_domains,
        from_staged=args.from_staged,
        from_sample=args.from_sample,
        sample_file=args.sample_file,
        keep_staged=args.keep_staged,
    )


if __name__ == "__main__":
    main()
