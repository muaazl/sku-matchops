import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd
import polars as pl
import requests
import hashlib
import json
import io
import shutil

from engine import config
from engine.db import ensure_db_initialized

logger = logging.getLogger("matchops.ingestion")

class DataIngestion:
    """Handles data ingestion from external sources (Google Sheets)."""

    _catalog_mem_cache = {}  # domain -> DataFrame
    _brands_mem_cache = {}   # domain -> DataFrame

    @staticmethod
    def clear_mem_cache(domain = None):
        """Clears the in-memory DataFrames cache."""
        if domain:
            DataIngestion._catalog_mem_cache.pop(domain, None)
            DataIngestion._brands_mem_cache.pop(domain, None)
            logger.info(f"[LOAD] [{domain.upper()}] Cleared in-memory cache.")
        else:
            DataIngestion._catalog_mem_cache.clear()
            DataIngestion._brands_mem_cache.clear()
            logger.info("[LOAD] Cleared all in-memory caches.")

    @staticmethod
    def _get_staged_path(sheet_name: str) -> str:
        """Returns the local staging filepath for a given sheet name."""
        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in sheet_name) + ".csv"
        return os.path.join(config.STAGING_DIR, safe_name)

    @staticmethod
    def is_sheet_staged(sheet_name: str) -> bool:
        """Checks if a sheet has already been staged locally."""
        path = DataIngestion._get_staged_path(sheet_name)
        return os.path.exists(path) and os.path.getsize(path) > 0

    @staticmethod
    def _download_sheet_with_retries(
        sheet_id: str, 
        sheet_name: str, 
        max_retries: int = 3, 
        backoff_factor: float = 1.5, 
        timeout: int = 30
    ) -> str:
        """Downloads a Google Sheet tab as raw CSV text with retries and exponential backoff."""
        encoded_name = sheet_name.replace(" ", "%20")
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded_name}"
        
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                text = r.text
                if not text or not text.strip():
                    raise ValueError(f"Received empty response from Google Sheets for tab '{sheet_name}'.")
                return text
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"[DOWNLOAD] (Attempt {attempt}/{max_retries}) Failed to download sheet '{sheet_name}': {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.critical(
                        f"[DOWNLOAD] (Attempt {attempt}/{max_retries}) Final failure downloading sheet '{sheet_name}': {e}"
                    )
        raise ConnectionError(
            f"Failed to download Google Sheet '{sheet_name}' after {max_retries} attempts. "
            f"Original error: {last_exception}"
        ) from last_exception

    @staticmethod
    def get_required_sheets_for_domain(domain: str) -> Dict[str, str]:
        """Returns the dictionary of logical keys to actual Google Sheet tab names for a domain."""
        third_tag_key = config.get_third_tag_col(domain)
        third_tag_sheet = "Region" if domain == config.DOMAIN_FOOD else "Category"
        domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
        
        return {
            "catalog": config.FOOD_CATALOG_SHEET if domain == config.DOMAIN_FOOD else config.MARKET_CATALOG_SHEET,
            "brands_or_flavors": "Food_Flavors" if domain == config.DOMAIN_FOOD else "Market_Brands",
            "gk": f"{domain_suffix}_GK",
            "bt": f"{domain_suffix}_BT",
            third_tag_key: f"{domain_suffix}_{third_tag_sheet}",
            "bt_gk_map": f"Classifier_BT_GK_Map_{domain_suffix}"
        }

    @staticmethod
    def stage_all_sheets(
        sheet_id: str, 
        domains: List[str] = None, 
        max_retries: int = 3
    ) -> Dict[str, str]:
        """
        Pre-downloads all Google Sheet tabs for all specified domains upfront into the local staging cache.
        This insulates the entire ingestion, cache build, training, and vectorization process against Wi-Fi/network drops.
        """
        if not domains:
            domains = [config.DOMAIN_MARKET, config.DOMAIN_FOOD]
            
        os.makedirs(config.STAGING_DIR, exist_ok=True)
        staged_files = {}
        
        # Collect all unique sheet tab names across domains
        sheets_to_fetch = {}
        for d in domains:
            domain_sheets = DataIngestion.get_required_sheets_for_domain(d)
            for role, sheet_name in domain_sheets.items():
                if sheet_name not in sheets_to_fetch:
                    sheets_to_fetch[sheet_name] = []
                sheets_to_fetch[sheet_name].append((d, role))
                
        total = len(sheets_to_fetch)
        logger.info(f"[STAGE] Pre-fetching {total} Google Sheet tabs for domains {[d.upper() for d in domains]}...")
        
        manifest = {
            "timestamp": time.time(),
            "domains": domains,
            "sheets": {}
        }
        
        for idx, (sheet_name, usages) in enumerate(sheets_to_fetch.items(), start=1):
            staged_path = DataIngestion._get_staged_path(sheet_name)
            logger.info(f"[STAGE] ({idx}/{total}) Downloading '{sheet_name}'...")
            
            try:
                content = DataIngestion._download_sheet_with_retries(sheet_id, sheet_name, max_retries=max_retries)
                with open(staged_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    
                file_size_kb = len(content.encode("utf-8")) / 1024.0
                content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
                manifest["sheets"][sheet_name] = {
                    "path": staged_path,
                    "hash": content_hash,
                    "size_kb": round(file_size_kb, 2),
                    "usages": usages
                }
                staged_files[sheet_name] = staged_path
                logger.info(f"[STAGE] ✓ ({idx}/{total}) Staged '{sheet_name}' ({file_size_kb:.1f} KB).")
            except Exception as e:
                logger.critical(f"[STAGE] ✖ Failed to stage sheet '{sheet_name}': {e}")
                raise e
                
        manifest_path = os.path.join(config.STAGING_DIR, "staging_manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
            
        logger.info(f"[STAGE] ✓ All {total} Google Sheets successfully staged locally in {config.STAGING_DIR}.")
        return staged_files

    @staticmethod
    def stage_all_from_excel(excel_path: str, domains: List[str] = None) -> Dict[str, str]:
        """
        Reads an Excel workbook (e.g. data/sample/SampleData.xlsx) and exports all tabs
        into the local staging cache (config.STAGING_DIR) as CSVs for offline ingestion.
        """
        if not domains:
            domains = [config.DOMAIN_MARKET, config.DOMAIN_FOOD]
            
        os.makedirs(config.STAGING_DIR, exist_ok=True)
        staged_files = {}
        
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names
        logger.info(f"[STAGE] Extracting {len(sheet_names)} sheets from '{excel_path}' into {config.STAGING_DIR}...")
        
        for name in sheet_names:
            df = xl.parse(name)
            staged_path = DataIngestion._get_staged_path(name)
            df.to_csv(staged_path, index=False, encoding="utf-8")
            staged_files[name] = staged_path
            
            # Handle aliases (e.g. Market_Cat -> Market_Category)
            if name == "Market_Cat":
                alt_path = DataIngestion._get_staged_path("Market_Category")
                df.to_csv(alt_path, index=False, encoding="utf-8")
                staged_files["Market_Category"] = alt_path
            elif name == "Market_Category":
                alt_path = DataIngestion._get_staged_path("Market_Cat")
                df.to_csv(alt_path, index=False, encoding="utf-8")
                staged_files["Market_Cat"] = alt_path

        # Generate BT-GK maps if not present in the workbook
        for domain in domains:
            domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
            map_name = f"Classifier_BT_GK_Map_{domain_suffix}"
            if map_name not in sheet_names:
                cat_sheet = config.FOOD_CATALOG_SHEET if domain == config.DOMAIN_FOOD else config.MARKET_CATALOG_SHEET
                if cat_sheet in sheet_names:
                    cat_df = xl.parse(cat_sheet)
                    col_bt = "BasicType" if "BasicType" in cat_df.columns else "basictype"
                    col_gk = "GenericKeywords" if "GenericKeywords" in cat_df.columns else "Generic keywords"
                    if col_bt in cat_df.columns and col_gk in cat_df.columns:
                        pairs = cat_df[[col_bt, col_gk]].dropna().drop_duplicates()
                        grouped = pairs.groupby(col_bt)[col_gk].apply(lambda x: ", ".join(x.unique())).reset_index()
                        grouped.columns = ["basictype", "generic keywords"]
                        map_path = DataIngestion._get_staged_path(map_name)
                        grouped.to_csv(map_path, index=False, encoding="utf-8")
                        staged_files[map_name] = map_path

        logger.info(f"[STAGE] ✓ Staged {len(staged_files)} tabs successfully from '{excel_path}'.")
        return staged_files

    @staticmethod
    def cleanup_staged_sheets():
        """Cleans up temporary staged Google Sheet CSV files and removes the staging directory."""
        if not os.path.exists(config.STAGING_DIR):
            return
            
        try:
            shutil.rmtree(config.STAGING_DIR, ignore_errors=True)
            logger.info(f"[STAGE] Removed temporary staging folder: {config.STAGING_DIR}")
        except Exception as e:
            logger.warning(f"[STAGE] Error while cleaning up staging directory: {e}")

    @staticmethod
    def _fetch_sheet_as_csv(sheet_id: str, sheet_name: str, **kwargs) -> pd.DataFrame:
        """Helper to fetch a specific Google Sheet tab as a CSV, reading from staged cache if available."""
        staged_path = DataIngestion._get_staged_path(sheet_name)
        if os.path.exists(staged_path) and os.path.getsize(staged_path) > 0:
            with open(staged_path, "rb") as f:
                content_bytes = f.read()
        else:
            text = DataIngestion._download_sheet_with_retries(sheet_id, sheet_name)
            content_bytes = text.encode("utf-8")

        # Handle header arguments if passed
        pl_kwargs = {"ignore_errors": True}
        if "header" in kwargs and kwargs["header"] is None:
            pl_kwargs["has_header"] = False

        df = pl.read_csv(content_bytes, **pl_kwargs)
        df = df.select([c for c in df.columns if not c.startswith("Unnamed")])
        pd_df = df.to_pandas()

        if "dtype" in kwargs and kwargs["dtype"] == str:
            pd_df = pd_df.astype(str)
        elif "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
            for col, dtype in kwargs["dtype"].items():
                if col in pd_df.columns:
                    pd_df[col] = pd_df[col].astype(dtype)
        return pd_df

    @staticmethod
    def load_catalog(sheet_id: str, domain: str = config.DOMAIN_MARKET, force_fetch: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads catalog and brands data from memory cache, local SQLite database if present, else downloads from Sheets."""
        # 1. Return from in-memory cache if available and not force_fetch
        if not force_fetch and domain in DataIngestion._catalog_mem_cache and domain in DataIngestion._brands_mem_cache:
            return DataIngestion._catalog_mem_cache[domain], DataIngestion._brands_mem_cache[domain]

        import sqlite3
        conn = ensure_db_initialized()
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM catalog_items WHERE domain = ?", (domain,))
            cat_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM brand_flavors WHERE domain = ?", (domain,))
            brand_count = cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"[LOAD] [{domain.upper()}] SQLite check failed: {e}. Falling back to sheet fetch.")
            cat_count = 0
            brand_count = 0
            
        if not force_fetch and cat_count > 0 and brand_count > 0:
            try:
                # Query catalog items
                cat_df = pd.read_sql_query("SELECT * FROM catalog_items WHERE domain = ?", conn, params=[domain])
                
                # Query brand flavors
                brands_df = pd.read_sql_query("SELECT * FROM brand_flavors WHERE domain = ?", conn, params=[domain])
                
                # Close DB connection
                conn.close()
                
                # Cleanup sqlite-specific column conversions
                entities_list = []
                for _, val in cat_df["entities_json"].items():
                    if pd.isna(val) or not val:
                        entities_list.append(None)
                    else:
                        try:
                            parsed = json.loads(val)
                            if isinstance(parsed, list):
                                res = []
                                for item in parsed:
                                    if isinstance(item, dict):
                                        res.append({k: set(v) if isinstance(v, list) else v for k, v in item.items()})
                                    else:
                                        res.append(item)
                                entities_list.append(res)
                            else:
                                entities_list.append(parsed)
                        except Exception:
                            entities_list.append(None)
                cat_df["entities"] = entities_list

                # Convert weight_val from JSON string to tuple, or reconstruct if numeric float
                from engine.nlp.text_cleaner import TextPipeline
                weight_list = []
                for idx, val in cat_df["weight_val"].items():
                    if pd.isna(val) or not val:
                        weight_list.append(None)
                    elif isinstance(val, (int, float)):
                        # Reconstruct tuple from clean_text if SQLite had a raw float
                        clean_t = cat_df.at[idx, "clean_text"]
                        if pd.isna(clean_t) or not clean_t:
                            weight_list.append((float(val), None, None))
                        else:
                            weight_list.append(TextPipeline.extract_weight_feature(clean_t))
                    else:
                        try:
                            parsed = json.loads(str(val))
                            weight_list.append(tuple(parsed) if isinstance(parsed, list) else parsed)
                        except Exception:
                            weight_list.append(None)
                cat_df["weight_val"] = weight_list
                
                # Rename columns back to DataFrame naming conventions
                cat_df = cat_df.rename(columns={
                    "generic_keywords": "Generic keywords",
                    "name": "Name",
                    "price": "Price",
                    "description": "Description"
                })
                
                if domain == config.DOMAIN_MARKET:
                    cat_df = cat_df.rename(columns={"brand": "Brand"})
                else:
                    cat_df = cat_df.rename(columns={"flavor": "Flavor"})
                    
                # Restore stable name sequence UIDs
                name_counts = {}
                db_uids = []
                for idx, row in cat_df.iterrows():
                    raw_name = str(row.get("Name", "")).strip().lower()
                    name_counts[raw_name] = name_counts.get(raw_name, 0) + 1
                    uid = f"{raw_name}#occ_{name_counts[raw_name]}"
                    db_uids.append(uid)
                cat_df["db_uid"] = db_uids
                
                # Sort out brand_flavors column names
                brands_df = brands_df.rename(columns={
                    "name": "Flavor Name" if domain == config.DOMAIN_FOOD else "Brand Name",
                    "aliases": "Aliases",
                    "is_weak": "Is_Weak",
                    "is_meat": "Is_Meat",
                    "is_vegetable": "Is_Vegetable",
                    "is_seafood": "Is_Seafood"
                })
                
                for col in ["Is_Weak", "Is_Meat", "Is_Vegetable", "Is_Seafood"]:
                    if col in brands_df.columns:
                        brands_df[col] = brands_df[col].astype(bool)
                        
                logger.info(f"[LOAD] [{domain.upper()}] Read {len(cat_df)} catalog rows and {len(brands_df)} brands from SQLite database.")
                
                # Cache in Python memory
                DataIngestion._catalog_mem_cache[domain] = cat_df
                DataIngestion._brands_mem_cache[domain] = brands_df
                
                return cat_df, brands_df
            except Exception as e:
                logger.error(f"[LOAD] [{domain.upper()}] Failed to query SQL tables: {e}. Falling back to fetch...")
                if conn:
                    conn.close()

        return DataIngestion.load_catalog_from_sheets(sheet_id, domain)

    @staticmethod
    def load_catalog_from_sheets(sheet_id: str, domain: str = config.DOMAIN_MARKET) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Downloads fresh data from Google Sheets, imports it incrementally into SQLite tables, and returns dataframes."""
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        hash_file = os.path.join(config.CACHE_DIR, f"{domain}_source_hashes.json")

        sheet_name = config.FOOD_CATALOG_SHEET if domain == config.DOMAIN_FOOD else config.MARKET_CATALOG_SHEET
        brand_sheet = "Food_Flavors" if domain == config.DOMAIN_FOOD else "Market_Brands"
        
        logger.info(f"[LOAD] [{domain.upper()}] Loading catalog and brands from staged cache / Google Sheets...")
        
        cat_staged = DataIngestion._get_staged_path(sheet_name)
        brand_staged = DataIngestion._get_staged_path(brand_sheet)
        
        try:
            if os.path.exists(cat_staged) and os.path.getsize(cat_staged) > 0:
                with open(cat_staged, "r", encoding="utf-8") as f:
                    catalog_text = f.read()
            else:
                catalog_text = DataIngestion._download_sheet_with_retries(sheet_id, sheet_name)
            catalog_hash = hashlib.md5(catalog_text.encode("utf-8")).hexdigest()
            
            if os.path.exists(brand_staged) and os.path.getsize(brand_staged) > 0:
                with open(brand_staged, "r", encoding="utf-8") as f:
                    brands_text = f.read()
            else:
                brands_text = DataIngestion._download_sheet_with_retries(sheet_id, brand_sheet)
            brands_hash = hashlib.md5(brands_text.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.critical(f"[LOAD] Failed to load catalog/brand sheets: {e}")
            raise e

        # Clean/Parse Polars DataFrames
        catalog_pl = pl.read_csv(catalog_text.encode("utf-8"), ignore_errors=True)
        catalog_pl = catalog_pl.select([c for c in catalog_pl.columns if not c.startswith("Unnamed")])
        col_map = config.CATALOG_COL_MAP_FOOD if domain == config.DOMAIN_FOOD else config.CATALOG_COL_MAP_MARKET
        catalog_pl = catalog_pl.rename({k: v for k, v in col_map.items() if k in catalog_pl.columns})
        
        if "Name" in catalog_pl.columns:
            catalog_pl = catalog_pl.filter(
                (pl.col("Name").is_not_null()) & 
                (pl.col("Name").str.strip_chars() != "") &
                (pl.col("Name").str.to_lowercase() != "nan") &
                (pl.col("Name").str.to_lowercase() != "none")
            )
            
        brands_pl = pl.read_csv(brands_text.encode("utf-8"), ignore_errors=True)
        brands_pl = brands_pl.select([c for c in brands_pl.columns if not c.startswith("Unnamed")])
        
        brand_col = "Flavor Name" if domain == config.DOMAIN_FOOD else "Brand Name"
        if brand_col in brands_pl.columns:
            brands_pl = brands_pl.filter(
                (pl.col(brand_col).is_not_null()) & 
                (pl.col(brand_col).str.strip_chars() != "") &
                (pl.col(brand_col).str.to_lowercase() != "nan") &
                (pl.col(brand_col).str.to_lowercase() != "none")
            )

        # Import directly to SQLite database (Incrementally)
        import sqlite3
        conn = ensure_db_initialized()
        try:
            # First, fetch existing items to make incremental comparisons
            old_rows = conn.execute("SELECT id, name, row_hash FROM catalog_items WHERE domain = ? ORDER BY id ASC", (domain,)).fetchall()
            old_uids_map = {}
            name_counts_old = {}
            for row_id, name, row_hash in old_rows:
                raw_name = str(name).strip().lower()
                name_counts_old[raw_name] = name_counts_old.get(raw_name, 0) + 1
                uid = f"{raw_name}#occ_{name_counts_old[raw_name]}"
                old_uids_map[uid] = (row_id, row_hash)
                
            old_brands = conn.execute("SELECT id, name, row_hash FROM brand_flavors WHERE domain = ? ORDER BY id ASC", (domain,)).fetchall()
            old_brands_map = {}
            brand_counts_old = {}
            for brand_id, name, row_hash in old_brands:
                raw_name = str(name).strip().lower()
                brand_counts_old[raw_name] = brand_counts_old.get(raw_name, 0) + 1
                uid = f"{raw_name}#occ_{brand_counts_old[raw_name]}"
                old_brands_map[uid] = (brand_id, row_hash)

            from engine.data_pipeline.cache_manager import calculate_row_hash, clean_price
            
            # --- 1. Incrementally update catalog_items ---
            cat_df_raw = catalog_pl.to_pandas()
            new_uids_seen = set()
            name_counts_new = {}
            
            # Keep counts of insert/update/delete for logging
            cat_inserted = 0
            cat_updated = 0
            cat_deleted = 0
            
            from engine.nlp.text_cleaner import TextPipeline
            for idx, row in cat_df_raw.iterrows():
                raw_name = str(row.get("Name", "")).strip().lower()
                name_counts_new[raw_name] = name_counts_new.get(raw_name, 0) + 1
                uid = f"{raw_name}#occ_{name_counts_new[raw_name]}"
                new_uids_seen.add(uid)
                
                row_hash = calculate_row_hash(row, domain=domain)
                
                # Pre-calculate NLP metadata columns for instant SQLite persistence
                name_val = str(row.get("Name", "") or "")
                clean_text = TextPipeline.normalize_final(TextPipeline.standardize_units(name_val))
                clean_no_weights = TextPipeline.strip_weights(clean_text)
                w_extracted = TextPipeline.extract_weight_feature(clean_text)
                weight_val_num = w_extracted[0] if (isinstance(w_extracted, tuple) and len(w_extracted) > 0 and w_extracted[0] is not None) else None
                token_count = len(clean_text.split())
                
                if uid not in old_uids_map:
                    conn.execute("""
                        INSERT INTO catalog_items (
                            domain, name, brand, flavor, basictype, generic_keywords, price, category, region, description,
                            clean_text, clean_no_weights, weight_val, token_count, row_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        domain,
                        row.get("Name"),
                        row.get("Brand") if domain == config.DOMAIN_MARKET else None,
                        row.get("Flavor") if domain == config.DOMAIN_FOOD else None,
                        row.get("basictype"),
                        row.get("Generic keywords"),
                        clean_price(row.get("Price")),
                        row.get("category") if domain == config.DOMAIN_MARKET else None,
                        row.get("region") if domain == config.DOMAIN_FOOD else None,
                        row.get("Description"),
                        clean_text,
                        clean_no_weights,
                        weight_val_num,
                        token_count,
                        row_hash
                    ))
                    cat_inserted += 1
                else:
                    row_id, old_hash = old_uids_map[uid]
                    if old_hash != row_hash:
                        conn.execute("""
                            UPDATE catalog_items
                            SET name = ?, brand = ?, flavor = ?, basictype = ?, generic_keywords = ?, price = ?, category = ?, region = ?, description = ?,
                                clean_text = ?, clean_no_weights = ?, weight_val = ?, token_count = ?, row_hash = ?
                            WHERE id = ?
                        """, (
                            row.get("Name"),
                            row.get("Brand") if domain == config.DOMAIN_MARKET else None,
                            row.get("Flavor") if domain == config.DOMAIN_FOOD else None,
                            row.get("basictype"),
                            row.get("Generic keywords"),
                            clean_price(row.get("Price")),
                            row.get("category") if domain == config.DOMAIN_MARKET else None,
                            row.get("region") if domain == config.DOMAIN_FOOD else None,
                            row.get("Description"),
                            clean_text,
                            clean_no_weights,
                            weight_val_num,
                            token_count,
                            row_hash,
                            row_id
                        ))
                        cat_updated += 1

            # Delete catalog items that are no longer in Sheets
            deleted_cat_uids = set(old_uids_map.keys()) - new_uids_seen
            for uid in deleted_cat_uids:
                row_id, _ = old_uids_map[uid]
                conn.execute("DELETE FROM catalog_items WHERE id = ?", (row_id,))
                cat_deleted += 1

            # --- 2. Incrementally update brand_flavors ---
            brands_df_raw = brands_pl.to_pandas()
            new_brands_seen = set()
            brand_counts_new = {}
            
            brands_inserted = 0
            brands_updated = 0
            brands_deleted = 0
            
            for idx, row in brands_df_raw.iterrows():
                name_val = row.get("Flavor Name") if domain == config.DOMAIN_FOOD else row.get("Brand Name")
                raw_name = str(name_val).strip().lower()
                brand_counts_new[raw_name] = brand_counts_new.get(raw_name, 0) + 1
                uid = f"{raw_name}#occ_{brand_counts_new[raw_name]}"
                new_brands_seen.add(uid)
                
                row_str = str(name_val) + str(row.get("Aliases", "")) + config.CACHE_SALT
                row_hash = hashlib.md5(row_str.encode("utf-8")).hexdigest()
                
                if uid not in old_brands_map:
                    conn.execute("""
                        INSERT INTO brand_flavors (
                            domain, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, row_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        domain,
                        name_val,
                        row.get("Aliases"),
                        int(row.get("Is_Weak")) if row.get("Is_Weak") is not None and not pd.isna(row.get("Is_Weak")) else 0,
                        int(row.get("Is_Meat")) if row.get("Is_Meat") is not None and not pd.isna(row.get("Is_Meat")) else 0,
                        int(row.get("Is_Vegetable")) if row.get("Is_Vegetable") is not None and not pd.isna(row.get("Is_Vegetable")) else 0,
                        int(row.get("Is_Seafood")) if row.get("Is_Seafood") is not None and not pd.isna(row.get("Is_Seafood")) else 0,
                        row_hash
                    ))
                    brands_inserted += 1
                else:
                    brand_id, old_hash = old_brands_map[uid]
                    if old_hash != row_hash:
                        conn.execute("""
                            UPDATE brand_flavors
                            SET name = ?, aliases = ?, is_weak = ?, is_meat = ?, is_vegetable = ?, is_seafood = ?, row_hash = ?
                            WHERE id = ?
                        """, (
                            name_val,
                            row.get("Aliases"),
                            int(row.get("Is_Weak")) if row.get("Is_Weak") is not None and not pd.isna(row.get("Is_Weak")) else 0,
                            int(row.get("Is_Meat")) if row.get("Is_Meat") is not None and not pd.isna(row.get("Is_Meat")) else 0,
                            int(row.get("Is_Vegetable")) if row.get("Is_Vegetable") is not None and not pd.isna(row.get("Is_Vegetable")) else 0,
                            int(row.get("Is_Seafood")) if row.get("Is_Seafood") is not None and not pd.isna(row.get("Is_Seafood")) else 0,
                            row_hash,
                            brand_id
                        ))
                        brands_updated += 1
                        
            # Delete brand flavors that are no longer in Sheets
            deleted_brand_uids = set(old_brands_map.keys()) - new_brands_seen
            for uid in deleted_brand_uids:
                brand_id, _ = old_brands_map[uid]
                conn.execute("DELETE FROM brand_flavors WHERE id = ?", (brand_id,))
                brands_deleted += 1
            
            # Commit SQLite changes
            conn.commit()
            
            # Clear memory cache so next queries fetch fresh database rows
            DataIngestion.clear_mem_cache(domain)
            
            logger.info(
                f"[LOAD] [{domain.upper()}] SQLite Incremental Update Complete: "
                f"Catalog ({cat_inserted} inserted, {cat_updated} updated, {cat_deleted} deleted), "
                f"Brands ({brands_inserted} inserted, {brands_updated} updated, {brands_deleted} deleted)."
            )
            
            # Sync to Meilisearch
            try:
                from backend.app.services.meilisearch_service import sync_dataframe_to_meili
                sync_dataframe_to_meili(cat_df_raw, domain)
            except Exception as meili_err:
                logger.error(f"[MEILI] Failed to sync updated catalog to Meilisearch: {meili_err}")

            with open(hash_file, "w") as f:
                json.dump({"catalog": catalog_hash, "brands": brands_hash}, f)
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[LOAD] Transaction rollback due to error: {e}")
            raise e
        finally:
            conn.close()

        # Re-query DB to get clean standardized DataFrames (populates in-memory cache)
        cat_df, brands_df = DataIngestion.load_catalog(sheet_id, domain, force_fetch=False)

        # Write high-performance Feather files for the rules engine
        try:
            import pyarrow.feather as feather
            catalog_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_mmap.feather")
            brands_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_brands_mmap.feather")
            
            # Clean non-serializable columns (like custom object lists in 'entities')
            cat_df_feather = cat_df.copy()
            if "entities" in cat_df_feather.columns:
                cat_df_feather = cat_df_feather.drop(columns=["entities"])
            if "weight_val" in cat_df_feather.columns:
                cat_df_feather["weight_val"] = cat_df_feather["weight_val"].astype(str)
                
            cat_df_feather = cat_df_feather.reset_index(drop=True)
            brands_df_feather = brands_df.reset_index(drop=True)
            
            feather.write_feather(cat_df_feather, catalog_cache_path, compression="lz4")
            feather.write_feather(brands_df_feather, brands_cache_path, compression="lz4")
            logger.info(f"[LOAD] Cached Feather files successfully for '{domain}' domain.")
        except Exception as fe_err:
            logger.error(f"[LOAD] Failed to write Feather cache files: {fe_err}")

        return cat_df, brands_df

    @staticmethod
    def load_classifier_dictionaries(sheet_id: str, domain: str = config.DOMAIN_MARKET, force_fetch: bool = False) -> Dict[str, List[str]]:
        """Loads tag dictionaries from SQLite DB. If DB is empty, imports from Google Sheets."""
        import sqlite3
        conn = ensure_db_initialized()
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM classifier_dictionaries WHERE domain = ?", (domain,))
            count = cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"[LOAD] classifier_dictionaries table check failed: {e}")
            count = 0
            
        if not force_fetch and count > 0:
            try:
                cursor.execute("SELECT tag_type, tag FROM classifier_dictionaries WHERE domain = ?", (domain,))
                rows = cursor.fetchall()
                conn.close()
                
                dicts = {"gk": [], "bt": []}
                third_tag_key = config.get_third_tag_col(domain)
                dicts[third_tag_key] = []
                
                for tag_type, tag in rows:
                    if tag_type in dicts:
                        dicts[tag_type].append(tag)
                return dicts
            except Exception as e:
                logger.error(f"[LOAD] Failed to load classifier dictionaries from SQLite: {e}")
                if conn:
                    conn.close()

        # Fetch from Sheets
        logger.info(f"[LOAD] Syncing classifier dictionaries from Google Sheets ({domain})...")
        dicts = {}
        third_tag_key = config.get_third_tag_col(domain)
        third_tag_sheet = "Region" if domain == config.DOMAIN_FOOD else "Category"
        domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
        sheets_to_load = {
            "gk": f"{domain_suffix}_GK",
            "bt": f"{domain_suffix}_BT",
            third_tag_key: f"{domain_suffix}_{third_tag_sheet}",
        }

        conn = ensure_db_initialized()
        try:
            conn.execute("DELETE FROM classifier_dictionaries WHERE domain = ?", (domain,))
            
            for key, sheet_name in sheets_to_load.items():
                df = DataIngestion._fetch_sheet_as_csv(sheet_id, sheet_name, header=None, dtype=str)
                keywords = df.iloc[:, 0].dropna().str.strip().tolist()
                keywords = [k for k in keywords if k]
                
                dicts[key] = keywords
                for kw in keywords:
                    conn.execute("""
                        INSERT INTO classifier_dictionaries (domain, tag_type, tag) VALUES (?, ?, ?)
                    """, (domain, key, kw))
                    
            conn.commit()
            logger.info(f"[LOAD] Imported classifier dictionaries into SQLite ({domain}).")
        except Exception as e:
            conn.rollback()
            logger.error(f"[LOAD] Failed to import classifier dictionaries: {e}")
            raise e
        finally:
            conn.close()
            
        return dicts

    @staticmethod
    def load_bt_gk_map_from_sheets(sheet_id: str, domain: str = config.DOMAIN_MARKET, force_fetch: bool = False) -> Dict[str, List[str]]:
        """Loads BT-GK map from SQLite DB. If DB is empty, imports from Google Sheets."""
        import sqlite3
        conn = ensure_db_initialized()
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bt_gk_map WHERE domain = ?", (domain,))
            count = cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"[LOAD] bt_gk_map table check failed: {e}")
            count = 0
            
        if not force_fetch and count > 0:
            try:
                cursor.execute("SELECT basictype, generic_keywords FROM bt_gk_map WHERE domain = ?", (domain,))
                rows = cursor.fetchall()
                conn.close()
                
                result = {}
                for bt, gks in rows:
                    result[bt] = [k.strip() for k in gks.split(",") if k.strip()]
                return result
            except Exception as e:
                logger.error(f"[LOAD] Failed to load BT-GK map from SQLite: {e}")
                if conn:
                    conn.close()

        # Fetch from Sheets
        domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
        sheet_name = f"Classifier_BT_GK_Map_{domain_suffix}"
        logger.info(f"[LOAD] Syncing BT→GK map from Google Sheets ({domain})...")

        df = DataIngestion._fetch_sheet_as_csv(sheet_id, sheet_name, dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]

        conn = ensure_db_initialized()
        try:
            conn.execute("DELETE FROM bt_gk_map WHERE domain = ?", (domain,))
            
            result = {}
            for _, row in df.iterrows():
                bt = str(row.get("basictype", "")).strip()
                gks = str(row.get("generic keywords", "")).strip()
                if not bt or not gks:
                    continue
                result[bt] = [k.strip() for k in gks.split(",") if k.strip()]
                
                conn.execute("""
                    INSERT INTO bt_gk_map (domain, basictype, generic_keywords) VALUES (?, ?, ?)
                """, (domain, bt, gks))
                
            conn.commit()
            logger.info(f"[LOAD] Imported BT→GK map into SQLite ({domain}).")
        except Exception as e:
            conn.rollback()
            logger.error(f"[LOAD] Failed to import BT→GK map: {e}")
            raise e
        finally:
            conn.close()
            
        return result

