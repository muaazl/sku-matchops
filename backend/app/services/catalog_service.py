import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import requests

from engine.core import config as engine_config
from engine.data_pipeline.cache_manager import calculate_row_hash, clean_price
from engine.data_pipeline.ingestion import DataIngestion

logger = logging.getLogger("matchops.catalog_service")

_build_lock = threading.Lock()
_build_in_progress = False


def get_catalog_and_brands(domain: str):
    """Load catalog and brands/flavors from local Feather cache or SQLite database."""
    return DataIngestion.load_catalog(engine_config.GOOGLE_SHEET_ID, domain, force_fetch=False)


def get_classifier_dicts(domain: str) -> Dict[str, Any]:
    """Load classifier dictionaries (GK, BT, category/region tags) from local cache or SQLite."""
    dicts_cache = os.path.join(engine_config.CACHE_DIR, f"{domain}_classifier_dicts.json")
    if os.path.exists(dicts_cache):
        try:
            with open(dicts_cache, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read dict cache for domain {domain}: {e}")
            
    # Load from SQLite or local cache with force_fetch=False (never downloads from Google Sheets)
    dicts = DataIngestion.load_classifier_dictionaries(engine_config.GOOGLE_SHEET_ID, domain, force_fetch=False)
    try:
        os.makedirs(engine_config.CACHE_DIR, exist_ok=True)
        with open(dicts_cache, "w", encoding="utf-8") as f:
            json.dump(dicts, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save dict cache for domain {domain}: {e}")
    return dicts


def get_bt_gk_cache(domain: str) -> Dict[str, Any]:
    """Load the cached BT-GK map and umbrella tags from pkl file, or reconstruct from SQLite."""
    cache_path = os.path.join(engine_config.CACHE_DIR, f"{domain}_bt_gk_cache.pkl")
    if os.path.exists(cache_path):
        try:
            return joblib.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load BT-GK cache for domain {domain}: {e}")

    # Reconstruct from SQLite bt_gk_map table if pkl cache is missing
    try:
        from engine.core.db import ensure_db_initialized
        conn = ensure_db_initialized()
        cur = conn.cursor()
        cur.execute("SELECT basictype, generic_keywords FROM bt_gk_map WHERE domain = ?", (domain,))
        rows = cur.fetchall()
        conn.close()
        if rows:
            bt_gk_map = {}
            for bt, gks in rows:
                bt_gk_map[bt] = [k.strip() for k in gks.split(",") if k.strip()]
            return {"bt_gk_map": bt_gk_map, "umbrella": {}, "third_tag_map": {}}
    except Exception as db_err:
        logger.warning(f"Failed to load BT-GK map from SQLite for domain {domain}: {db_err}")

    return {"bt_gk_map": {}, "umbrella": {}, "third_tag_map": {}}


def clean_record_nans(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert pd.isna/NaN values in records to None for clean JSON serialization."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in records]


def escape_meili_filter_value(val: str) -> str:
    """Escapes double quotes and backslashes for safe inclusion in Meilisearch filter strings."""
    return val.replace("\\", "\\\\").replace('"', '\\"')


def _build_occurrence_uids(df: pd.DataFrame, name_col: str = "Name"):
    """
    Vectorized replacement for an iterrows()-based per-name occurrence counter.
    Returns (raw_names, uids) as lists aligned to df's row order, where
    uid = "{raw_name}#occ_{n}" (n = 1-based count of that name seen so far).

    Uses .map(str) rather than .astype(str) so a real NaN stringifies to "nan"
    (matching the str(x) semantics used elsewhere in this function) instead of
    pandas treating it as a missing value, which would make groupby() drop
    those rows and silently coerce cumcount()'s output to float.
    """
    if len(df) == 0:
        return [], []
    if name_col in df.columns:
        raw_names = df[name_col].astype(object).map(str).str.strip().str.lower()
    else:
        raw_names = pd.Series([""] * len(df), index=df.index)
    occ = raw_names.groupby(raw_names).cumcount() + 1
    uids = (raw_names + "#occ_" + occ.astype(str)).tolist()
    return raw_names.tolist(), uids


def check_changes_for_domain(domain: str, limit: int = 50) -> dict:
    """Checks differences between Google Sheets and local SQLite cached catalog items."""
    import sqlite3
    import polars as pl

    conn = sqlite3.connect(engine_config.DB_PATH)
    cached_df = pd.DataFrame()
    old_hashes = {}
    try:
        cached_df = pd.read_sql_query("SELECT * FROM catalog_items WHERE domain = ?", conn, params=[domain])
        cached_df = cached_df.rename(columns={
            "generic_keywords": "Generic keywords",
            "name": "Name",
            "price": "Price",
            "description": "Description"
        })
        if domain == "market":
            cached_df = cached_df.rename(columns={"brand": "Brand"})
        else:
            cached_df = cached_df.rename(columns={"flavor": "Flavor"})
            
        cached_by_uid = {}
        cached_by_name = {}
        cached_raw_names, cached_uids = _build_occurrence_uids(cached_df)
        if cached_uids:
            row_hash_col = cached_df["row_hash"] if "row_hash" in cached_df.columns else pd.Series([""] * len(cached_df), index=cached_df.index)
            cached_records = cached_df.to_dict("records")
            old_hashes = dict(zip(cached_uids, row_hash_col))
            cached_by_uid = dict(zip(cached_uids, cached_records))
            for raw_name, record in zip(cached_raw_names, cached_records):
                if raw_name not in cached_by_name:
                    cached_by_name[raw_name] = record
    except Exception as db_err:
        logger.warning(f"Failed to load cached items from SQLite: {db_err}")
    finally:
        conn.close()
            
    sheet_name = engine_config.FOOD_CATALOG_SHEET if domain == "food" else engine_config.MARKET_CATALOG_SHEET
    url = f"https://docs.google.com/spreadsheets/d/{engine_config.GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name.replace(' ', '%20')}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    
    catalog_pl = pl.read_csv(r.content, ignore_errors=True)
    catalog_pl = catalog_pl.select([c for c in catalog_pl.columns if not c.startswith("Unnamed")])
    col_map = engine_config.CATALOG_COL_MAP_FOOD if domain == "food" else engine_config.CATALOG_COL_MAP_MARKET
    catalog_pl = catalog_pl.rename({k: v for k, v in col_map.items() if k in catalog_pl.columns})
    new_df = catalog_pl.to_pandas()
    
    if "Name" in new_df.columns:
        new_df = new_df[
            (new_df["Name"].astype(str).str.strip() != "") & 
            (new_df["Name"].notna()) &
            (new_df["Name"].astype(str).str.lower() != "nan") &
            (new_df["Name"].astype(str).str.lower() != "none")
        ]
        
    compare_cols = ["Name", "basictype", "Generic keywords", "Description", "Price"]
    if domain == "food":
        compare_cols.extend(["region", "Flavor"])
    else:
        compare_cols.extend(["category", "Brand"])
        
    new_rows = []
    changed_rows = []
    total_new_count = 0
    total_changed_count = 0

    new_raw_names, new_uids = _build_occurrence_uids(new_df)
    new_records = new_df.to_dict("records") if len(new_df) else []

    for idx, row, uid, raw_name in zip(new_df.index, new_records, new_uids, new_raw_names):
        row_hash = calculate_row_hash(row, domain=domain)
        
        cleaned = {}
        for c in compare_cols:
            val = row.get(c, "")
            if pd.isna(val) or val is None:
                val = ""
            elif c == "Price":
                p = clean_price(val)
                val = f"{p:.2f}" if p is not None else ""
            elif isinstance(val, (int, float)):
                val = f"{float(val):.2f}"
            else:
                val = str(val).strip()
            cleaned[c] = val
        
        if uid not in old_hashes:
            total_new_count += 1
            if len(new_rows) < limit:
                new_rows.append({
                    "index": idx,
                    "name": cleaned["Name"],
                    "details": {c: cleaned[c] for c in compare_cols if cleaned[c]}
                })
        elif old_hashes[uid] != row_hash:
            old_row = None
            if not cached_df.empty:
                if idx < len(cached_df) and str(cached_df.iloc[idx].get("Name", "")) == str(row.get("Name", "")):
                    old_row = cached_df.iloc[idx]
                else:
                    old_row = cached_by_uid.get(uid)
                    if old_row is None:
                        old_row = cached_by_name.get(raw_name)
            
            diffs = {}
            if old_row is not None:
                for c in compare_cols:
                    old_val = old_row.get(c, "")
                    if pd.isna(old_val) or old_val is None:
                        old_val = ""
                    elif c == "Price":
                        p = clean_price(old_val)
                        old_val = f"{p:.2f}" if p is not None else ""
                    elif isinstance(old_val, (int, float)):
                        old_val = f"{float(old_val):.2f}"
                    else:
                        old_val = str(old_val).strip()
                        
                    if old_val != cleaned[c]:
                        diffs[c] = {"old": old_val, "new": cleaned[c]}
            else:
                diffs = {"content": {"old": "(unknown)", "new": "updated"}}
                
            if diffs:
                total_changed_count += 1
                if len(changed_rows) < limit:
                    changed_rows.append({
                        "index": idx,
                        "name": cleaned["Name"],
                        "diffs": diffs
                    })
                
    return {
        "new_count": total_new_count,
        "changed_count": total_changed_count,
        "new_rows": new_rows,
        "changed_rows": changed_rows,
        "is_capped": total_new_count > len(new_rows) or total_changed_count > len(changed_rows)
    }


def _bg_build_cache():
    """Background task to rebuild Feather caches and warm up classifiers/pipelines."""
    global _build_in_progress
    logger.info("Background cache build and pre-training started...")
    try:
        for domain in ("market", "food"):
            logger.info(f"Rebuilding Feather cache for {domain}...")
            cat_df, _ = DataIngestion.load_catalog(engine_config.GOOGLE_SHEET_ID, domain=domain, force_fetch=True)
            try:
                DataIngestion.load_classifier_dictionaries(engine_config.GOOGLE_SHEET_ID, domain=domain, force_fetch=True)
                DataIngestion.load_bt_gk_map_from_sheets(engine_config.GOOGLE_SHEET_ID, domain=domain, force_fetch=True)
                DataIngestion.compute_and_store_dictionary_counts(domain, cat_df=cat_df)
                from engine.data_pipeline.meilisearch_sync import sync_dictionaries_to_meili
                sync_dictionaries_to_meili(domain)
            except Exception as dict_err:
                logger.warning(f"Could not refresh dictionaries for {domain} in background build: {dict_err}")
            
        from backend.app.services.engine_client import reload_engine_models
        logger.info("Notifying ML Engine to reload models and rebuild pipelines...")
        reload_engine_models()
            
        logger.info("Background cache build and pre-training completed successfully.")
    except Exception as e:
        logger.error(f"Failed to build cache in background: {e}")
    finally:
        _build_in_progress = False


def trigger_build_cache() -> tuple[bool, str]:
    """
    Attempts to acquire the build lock and initiate background cache rebuilding.
    Returns (started, message).
    """
    global _build_in_progress
    if _build_in_progress:
        return False, "Cache building is already in progress."
        
    acquired = _build_lock.acquire(blocking=False)
    if not acquired:
        return False, "Cache building already active."
        
    _build_in_progress = True
    return True, "Cache building and pre-training initiated in background."


def get_build_task_callable():
    """Returns the wrapped background task callable that releases the lock upon completion."""
    def run_with_lock():
        try:
            _bg_build_cache()
        finally:
            _build_lock.release()
    return run_with_lock
