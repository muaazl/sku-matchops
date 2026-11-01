import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time

import pandas as pd
import polars as pl
import requests

from engine import config

logger = logging.getLogger("matchops.ingestion")


def _nan_to_none(row: dict) -> dict:
    """pandas NaN -> None."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def _flag(val) -> int:
    return int(val) if val is not None and not pd.isna(val) else 0


class DataIngestion:
    """Handles data ingestion from external sources (Google Sheets) and local cache."""

    _catalog_mem_cache = {}  # domain -> DataFrame
    _brands_mem_cache = {}  # domain -> DataFrame
    _load_lock = threading.RLock()

    @staticmethod
    def clear_mem_cache(domain=None):
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
        safe_name = (
            "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in sheet_name) + ".csv"
        )
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
        timeout: int = 30,
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
                    raise ValueError(
                        f"Received empty response from Google Sheets for tab '{sheet_name}'."
                    )
                return text
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = backoff_factor**attempt
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
    def get_required_sheets_for_domain(domain: str) -> dict[str, str]:
        """Returns the dictionary of logical keys to actual Google Sheet tab names for a domain."""
        third_tag_key = config.get_third_tag_col(domain)
        third_tag_sheet = "Region" if domain == config.DOMAIN_FOOD else "Category"
        domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"

        return {
            "catalog": config.FOOD_CATALOG_SHEET
            if domain == config.DOMAIN_FOOD
            else config.MARKET_CATALOG_SHEET,
            "brands_or_flavors": "Food_Flavors"
            if domain == config.DOMAIN_FOOD
            else "Market_Brands",
            "gk": f"{domain_suffix}_GK",
            "bt": f"{domain_suffix}_BT",
            third_tag_key: f"{domain_suffix}_{third_tag_sheet}",
        }

    @staticmethod
    def stage_all_sheets(
        sheet_id: str, domains: list[str] = None, max_retries: int = 3
    ) -> dict[str, str]:
        """
        Pre-downloads all Google Sheet tabs for all specified domains upfront into the local staging cache.
        This insulates the entire ingestion, cache build, training, and vectorization process against Wi-Fi/network drops.
        """
        if not domains:
            domains = [config.DOMAIN_MARKET, config.DOMAIN_FOOD]

        os.makedirs(config.STAGING_DIR, exist_ok=True)
        staged_files = {}

        sheets_to_fetch = {}
        for d in domains:
            domain_sheets = DataIngestion.get_required_sheets_for_domain(d)
            for role, sheet_name in domain_sheets.items():
                if sheet_name not in sheets_to_fetch:
                    sheets_to_fetch[sheet_name] = []
                sheets_to_fetch[sheet_name].append((d, role))

        total = len(sheets_to_fetch)
        logger.info(
            f"[STAGE] Pre-fetching {total} Google Sheet tabs for domains {[d.upper() for d in domains]}..."
        )

        manifest = {"timestamp": time.time(), "domains": domains, "sheets": {}}

        for idx, (sheet_name, usages) in enumerate(sheets_to_fetch.items(), start=1):
            staged_path = DataIngestion._get_staged_path(sheet_name)
            logger.info(f"[STAGE] ({idx}/{total}) Downloading '{sheet_name}'...")

            try:
                content = DataIngestion._download_sheet_with_retries(
                    sheet_id, sheet_name, max_retries=max_retries
                )
                with open(staged_path, "w", encoding="utf-8") as f:
                    f.write(content)

                file_size_kb = len(content.encode("utf-8")) / 1024.0
                content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
                manifest["sheets"][sheet_name] = {
                    "path": staged_path,
                    "hash": content_hash,
                    "size_kb": round(file_size_kb, 2),
                    "usages": usages,
                }
                staged_files[sheet_name] = staged_path
                logger.info(
                    f"[STAGE] ✓ ({idx}/{total}) Staged '{sheet_name}' ({file_size_kb:.1f} KB)."
                )
            except Exception as e:
                logger.critical(f"[STAGE] ✖ Failed to stage sheet '{sheet_name}': {e}")
                raise e

        # Generate BT-GK map CSVs on the fly directly from staged catalog sheets
        for d in domains:
            domain_suffix = "Food" if d == config.DOMAIN_FOOD else "Market"
            map_name = f"Classifier_BT_GK_Map_{domain_suffix}"
            cat_name = (
                config.FOOD_CATALOG_SHEET
                if d == config.DOMAIN_FOOD
                else config.MARKET_CATALOG_SHEET
            )
            map_path = DataIngestion._get_staged_path(map_name)
            cat_path = DataIngestion._get_staged_path(cat_name)

            if os.path.exists(cat_path):
                try:
                    cat_df = pd.read_csv(cat_path, dtype=str)
                    col_bt = next(
                        (
                            c
                            for c in cat_df.columns
                            if c.strip().lower() in ("basictype", "basic type")
                        ),
                        None,
                    )
                    col_gk = next(
                        (
                            c
                            for c in cat_df.columns
                            if c.strip().lower() in ("generickeywords", "generic keywords")
                        ),
                        None,
                    )
                    if col_bt and col_gk:
                        pairs = cat_df[[col_bt, col_gk]].dropna().drop_duplicates()
                        grouped = (
                            pairs.groupby(col_bt)[col_gk]
                            .apply(
                                lambda x: ", ".join(
                                    [k.strip() for k in x.unique() if str(k).strip()]
                                )
                            )
                            .reset_index()
                        )
                        grouped.columns = ["basictype", "generic keywords"]
                        grouped.to_csv(map_path, index=False, encoding="utf-8")
                        staged_files[map_name] = map_path
                        logger.info(
                            f"[STAGE] ✓ Built on-the-fly BT→GK map for '{d.upper()}' ({len(grouped)} entries)."
                        )
                except Exception as gen_err:
                    logger.warning(
                        f"[STAGE] Could not generate on-the-fly BT-GK map for '{d}': {gen_err}"
                    )

        manifest_path = os.path.join(config.STAGING_DIR, "staging_manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

        logger.info(
            f"[STAGE] ✓ All {total} Google Sheets successfully staged locally in {config.STAGING_DIR}."
        )
        return staged_files

    @staticmethod
    def stage_all_from_excel(excel_path: str, domains: list[str] = None) -> dict[str, str]:
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
        logger.info(
            f"[STAGE] Extracting {len(sheet_names)} sheets from '{excel_path}' into {config.STAGING_DIR}..."
        )

        for name in sheet_names:
            df = xl.parse(name)
            staged_path = DataIngestion._get_staged_path(name)
            df.to_csv(staged_path, index=False, encoding="utf-8")
            staged_files[name] = staged_path

            if name == "Market_Cat":
                alt_path = DataIngestion._get_staged_path("Market_Category")
                df.to_csv(alt_path, index=False, encoding="utf-8")
                staged_files["Market_Category"] = alt_path
            elif name == "Market_Category":
                alt_path = DataIngestion._get_staged_path("Market_Cat")
                df.to_csv(alt_path, index=False, encoding="utf-8")
                staged_files["Market_Cat"] = alt_path

        for domain in domains:
            domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
            map_name = f"Classifier_BT_GK_Map_{domain_suffix}"
            if map_name not in sheet_names:
                cat_sheet = (
                    config.FOOD_CATALOG_SHEET
                    if domain == config.DOMAIN_FOOD
                    else config.MARKET_CATALOG_SHEET
                )
                if cat_sheet in sheet_names:
                    cat_df = xl.parse(cat_sheet)
                    col_bt = "BasicType" if "BasicType" in cat_df.columns else "basictype"
                    col_gk = (
                        "GenericKeywords"
                        if "GenericKeywords" in cat_df.columns
                        else "Generic keywords"
                    )
                    if col_bt in cat_df.columns and col_gk in cat_df.columns:
                        pairs = cat_df[[col_bt, col_gk]].dropna().drop_duplicates()
                        grouped = (
                            pairs.groupby(col_bt)[col_gk]
                            .apply(lambda x: ", ".join(x.unique()))
                            .reset_index()
                        )
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

        pl_kwargs = {"ignore_errors": True}
        if "header" in kwargs and kwargs["header"] is None:
            pl_kwargs["has_header"] = False

        df = pl.read_csv(content_bytes, **pl_kwargs)
        df = df.select([c for c in df.columns if not c.startswith("Unnamed")])
        pd_df = df.to_pandas()

        if "dtype" in kwargs and kwargs["dtype"] is str:
            pd_df = pd_df.astype(str)
        elif "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
            for col, dtype in kwargs["dtype"].items():
                if col in pd_df.columns:
                    pd_df[col] = pd_df[col].astype(dtype)
        return pd_df

    @staticmethod
    def load_catalog(
        sheet_id: str = config.GOOGLE_SHEET_ID,
        domain: str = config.DOMAIN_MARKET,
        force_fetch: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads catalog and brands data from memory cache or local Feather cache."""
        if (
            not force_fetch
            and domain in DataIngestion._catalog_mem_cache
            and domain in DataIngestion._brands_mem_cache
        ):
            return DataIngestion._catalog_mem_cache[domain], DataIngestion._brands_mem_cache[domain]

        with DataIngestion._load_lock:
            if (
                not force_fetch
                and domain in DataIngestion._catalog_mem_cache
                and domain in DataIngestion._brands_mem_cache
            ):
                return (
                    DataIngestion._catalog_mem_cache[domain],
                    DataIngestion._brands_mem_cache[domain],
                )

            # Check local Feather cache
            if not force_fetch:
                cat_feather = os.path.join(config.CACHE_DIR, f"{domain}_catalog_mmap.feather")
                brands_feather = os.path.join(config.CACHE_DIR, f"{domain}_brands_mmap.feather")
                if os.path.exists(cat_feather) and os.path.exists(brands_feather):
                    try:
                        cat_df = pd.read_feather(cat_feather)
                        brands_df = pd.read_feather(brands_feather)
                        if not cat_df.empty and not brands_df.empty:
                            logger.info(
                                f"[LOAD] [{domain.upper()}] Loaded {len(cat_df)} catalog items and {len(brands_df)} brands from local Feather cache."
                            )
                            DataIngestion._catalog_mem_cache[domain] = cat_df
                            DataIngestion._brands_mem_cache[domain] = brands_df
                            return cat_df, brands_df
                    except Exception as fe_err:
                        logger.warning(
                            f"[LOAD] [{domain.upper()}] Failed to read Feather cache: {fe_err}."
                        )

            # Check if staged sheets exist or force_fetch is requested
            sheet_name = (
                config.FOOD_CATALOG_SHEET
                if domain == config.DOMAIN_FOOD
                else config.MARKET_CATALOG_SHEET
            )
            brand_sheet = "Food_Flavors" if domain == config.DOMAIN_FOOD else "Market_Brands"
            can_load_sheets = force_fetch or (
                DataIngestion.is_sheet_staged(sheet_name)
                and DataIngestion.is_sheet_staged(brand_sheet)
            )

            if not can_load_sheets:
                raise RuntimeError(
                    f"[LOAD] No catalog data found in memory or Feather cache for domain '{domain}'. "
                    f"Please run 'python -m scripts.catalog.sync' to initialize the catalog."
                )

            return DataIngestion.load_catalog_from_sheets(sheet_id, domain)

    @staticmethod
    def load_catalog_from_sheets(
        sheet_id: str, domain: str = config.DOMAIN_MARKET
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Downloads or loads staged sheets, parses into standardized DataFrames, and writes Feather caches."""
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        sheet_name = (
            config.FOOD_CATALOG_SHEET
            if domain == config.DOMAIN_FOOD
            else config.MARKET_CATALOG_SHEET
        )
        brand_sheet = "Food_Flavors" if domain == config.DOMAIN_FOOD else "Market_Brands"

        cat_staged = DataIngestion._get_staged_path(sheet_name)
        brand_staged = DataIngestion._get_staged_path(brand_sheet)

        if os.path.exists(cat_staged) and os.path.getsize(cat_staged) > 0:
            with open(cat_staged, encoding="utf-8") as f:
                catalog_text = f.read()
        else:
            catalog_text = DataIngestion._download_sheet_with_retries(sheet_id, sheet_name)

        if os.path.exists(brand_staged) and os.path.getsize(brand_staged) > 0:
            with open(brand_staged, encoding="utf-8") as f:
                brands_text = f.read()
        else:
            brands_text = DataIngestion._download_sheet_with_retries(sheet_id, brand_sheet)

        catalog_pl = pl.read_csv(catalog_text.encode("utf-8"), ignore_errors=True)
        catalog_pl = catalog_pl.select(
            [c for c in catalog_pl.columns if not c.startswith("Unnamed")]
        )
        col_map = (
            config.CATALOG_COL_MAP_FOOD
            if domain == config.DOMAIN_FOOD
            else config.CATALOG_COL_MAP_MARKET
        )
        catalog_pl = catalog_pl.rename(
            {k: v for k, v in col_map.items() if k in catalog_pl.columns}
        )

        if "Name" in catalog_pl.columns:
            catalog_pl = catalog_pl.filter(
                (pl.col("Name").is_not_null())
                & (pl.col("Name").str.strip_chars() != "")
                & (pl.col("Name").str.to_lowercase() != "nan")
                & (pl.col("Name").str.to_lowercase() != "none")
            )

        brands_pl = pl.read_csv(brands_text.encode("utf-8"), ignore_errors=True)
        brands_pl = brands_pl.select([c for c in brands_pl.columns if not c.startswith("Unnamed")])

        brand_col = "Flavor Name" if domain == config.DOMAIN_FOOD else "Brand Name"
        if brand_col in brands_pl.columns:
            brands_pl = brands_pl.filter(
                (pl.col(brand_col).is_not_null())
                & (pl.col(brand_col).str.strip_chars() != "")
                & (pl.col(brand_col).str.to_lowercase() != "nan")
                & (pl.col(brand_col).str.to_lowercase() != "none")
            )

        cat_df = catalog_pl.to_pandas()
        brands_df = brands_pl.to_pandas()

        from engine.data_pipeline.cache_manager import calculate_row_hash, clean_price
        from engine.nlp.text_cleaner import TextPipeline

        raw_names = cat_df["Name"].fillna("").astype(str).str.strip().str.lower()
        occ_counts = raw_names.groupby(raw_names).cumcount() + 1
        cat_df["db_uid"] = raw_names + "#occ_" + occ_counts.astype(str)

        clean_texts = []
        clean_no_weights = []
        weights = []
        token_counts = []
        hashes = []
        for row in cat_df.to_dict(orient="records"):
            name_val = str(row.get("Name", "") or "")
            ct = TextPipeline.normalize_final(TextPipeline.standardize_units(name_val))
            clean_texts.append(ct)
            clean_no_weights.append(TextPipeline.strip_weights(ct))
            weights.append(TextPipeline.extract_weight_feature(ct))
            token_counts.append(len(ct.split()))
            hashes.append(calculate_row_hash(row, domain=domain))

        cat_df["clean_text"] = clean_texts
        cat_df["clean_no_weights"] = clean_no_weights
        cat_df["weight_val"] = weights
        cat_df["token_count"] = token_counts
        cat_df["row_hash"] = hashes
        if "Price" in cat_df.columns:
            cat_df["price"] = cat_df["Price"].apply(clean_price)

        if domain == config.DOMAIN_MARKET:
            cat_df = cat_df.rename(columns={"brand": "Brand"})
        else:
            cat_df = cat_df.rename(columns={"flavor": "Flavor"})

        brands_df = brands_df.rename(
            columns={
                "name": "Flavor Name" if domain == config.DOMAIN_FOOD else "Brand Name",
                "aliases": "Aliases",
                "is_weak": "Is_Weak",
                "is_meat": "Is_Meat",
                "is_vegetable": "Is_Vegetable",
                "is_seafood": "Is_Seafood",
            }
        )

        for col in ["Is_Weak", "Is_Meat", "Is_Vegetable", "Is_Seafood"]:
            if col in brands_df.columns:
                brands_df[col] = brands_df[col].astype(bool)

        DataIngestion._catalog_mem_cache[domain] = cat_df
        DataIngestion._brands_mem_cache[domain] = brands_df

        try:
            import pyarrow.feather as feather

            cat_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_mmap.feather")
            brands_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_brands_mmap.feather")

            cat_df_feather = cat_df.copy()
            if "entities" in cat_df_feather.columns:
                cat_df_feather = cat_df_feather.drop(columns=["entities"])
            if "weight_val" in cat_df_feather.columns:
                cat_df_feather["weight_val"] = cat_df_feather["weight_val"].astype(str)

            feather.write_feather(
                cat_df_feather.reset_index(drop=True), cat_cache_path, compression="lz4"
            )
            feather.write_feather(
                brands_df.reset_index(drop=True), brands_cache_path, compression="lz4"
            )
            logger.info(f"[LOAD] Cached Feather files successfully for '{domain}' domain.")
        except Exception as fe_err:
            logger.warning(f"[LOAD] Could not write Feather files: {fe_err}")

        return cat_df, brands_df

    @staticmethod
    def load_classifier_dictionaries(
        sheet_id: str = config.GOOGLE_SHEET_ID,
        domain: str = config.DOMAIN_MARKET,
        force_fetch: bool = False,
    ) -> dict[str, list[str]]:
        """Loads tag dictionaries from local JSON cache or staged sheets."""
        dicts_cache = os.path.join(config.CACHE_DIR, f"{domain}_classifier_dicts.json")
        if not force_fetch and os.path.exists(dicts_cache):
            try:
                with open(dicts_cache, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[LOAD] Failed to read JSON dict cache: {e}")

        third_tag_key = config.get_third_tag_col(domain)
        third_tag_sheet = "Region" if domain == config.DOMAIN_FOOD else "Category"
        domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
        sheets_to_load = {
            "gk": f"{domain_suffix}_GK",
            "bt": f"{domain_suffix}_BT",
            third_tag_key: f"{domain_suffix}_{third_tag_sheet}",
        }

        can_load = force_fetch or all(
            DataIngestion.is_sheet_staged(s) for s in sheets_to_load.values()
        )
        if not can_load:
            raise RuntimeError(
                f"[LOAD] No classifier dictionaries found in local cache for domain '{domain}'. "
                f"Please run 'python -m scripts.catalog.sync' to initialize dictionaries."
            )

        logger.info(
            f"[LOAD] Loading classifier dictionaries from staged sheets / Google Sheets ({domain})..."
        )
        dicts = {}
        for key, sheet_name in sheets_to_load.items():
            df = DataIngestion._fetch_sheet_as_csv(sheet_id, sheet_name, header=None, dtype=str)
            keywords = df.iloc[:, 0].dropna().str.strip().tolist()
            dicts[key] = [k for k in keywords if k]

        os.makedirs(config.CACHE_DIR, exist_ok=True)
        try:
            with open(dicts_cache, "w", encoding="utf-8") as f:
                json.dump(dicts, f, indent=2)
        except Exception as e:
            logger.warning(f"[LOAD] Failed to save JSON dict cache: {e}")

        return dicts

    @staticmethod
    def build_bt_gk_map_from_catalog(
        domain: str = config.DOMAIN_MARKET,
        cat_df: pd.DataFrame | None = None,
    ) -> dict[str, list[str]]:
        """Builds the BT->GK map on the fly from catalog DataFrame purely in-memory."""
        from collections import defaultdict

        if cat_df is None or cat_df.empty:
            cat_df, _ = DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, domain=domain)

        grouped = defaultdict(set)
        col_bt = next(
            (c for c in cat_df.columns if c.strip().lower() in ("basictype", "basic type")),
            None,
        )
        col_gk = next(
            (
                c
                for c in cat_df.columns
                if c.strip().lower() in ("generickeywords", "generic keywords", "generic_keywords")
            ),
            None,
        )
        if col_bt and col_gk:
            bt_list = cat_df[col_bt].tolist()
            gk_list = cat_df[col_gk].tolist()
            for raw_bt, raw_gk in zip(bt_list, gk_list):
                bt_clean = str(raw_bt or "").strip()
                gk_str = str(raw_gk or "").strip()
                if (
                    not bt_clean
                    or not gk_str
                    or bt_clean.lower() in ("nan", "none")
                    or gk_str.lower() in ("nan", "none")
                ):
                    continue
                for k in gk_str.split(","):
                    k_clean = k.strip()
                    if k_clean and k_clean.lower() != "nan":
                        grouped[bt_clean].add(k_clean)

        return {bt: sorted(gks) for bt, gks in grouped.items()}

    @staticmethod
    def load_bt_gk_map_from_sheets(
        sheet_id: str = None,
        domain: str = config.DOMAIN_MARKET,
        force_fetch: bool = False,
        cat_df: pd.DataFrame | None = None,
    ) -> dict[str, list[str]]:
        """Loads or builds BT-GK map in memory."""
        cache_path = os.path.join(config.CACHE_DIR, f"{domain}_bt_gk_cache.pkl")
        if not force_fetch and os.path.exists(cache_path):
            try:
                import joblib

                cache_data = joblib.load(cache_path)
                if isinstance(cache_data, dict) and "bt_gk_map" in cache_data:
                    return cache_data["bt_gk_map"]
                elif isinstance(cache_data, dict):
                    return cache_data
            except Exception as e:
                logger.warning(f"[LOAD] Failed to read BT-GK pickle cache: {e}")

        return DataIngestion.build_bt_gk_map_from_catalog(domain=domain, cat_df=cat_df)

    @staticmethod
    def compute_and_store_dictionary_counts(domain: str, cat_df: pd.DataFrame | None = None):
        """Deprecated on engine; database persistence is restricted to backend services."""
        logger.debug(
            f"[LOAD] [{domain.upper()}] compute_and_store_dictionary_counts is handled by backend service."
        )
