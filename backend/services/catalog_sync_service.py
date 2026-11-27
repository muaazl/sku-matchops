"""Google Sheets catalog synchronization, dictionary extraction, and frequency computation."""

import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import requests

from backend.core import config
from backend.core.constants import (
    CACHE_SALT,
    CATALOG_COL_MAP_FOOD,
    CATALOG_COL_MAP_MARKET,
    DOMAIN_FOOD,
    DOMAIN_MARKET,
    FOOD_CATALOG_SHEET,
    MARKET_CATALOG_SHEET,
    get_third_tag_col,
)
from backend.core.helpers import (
    build_occurrence_uids,
    calculate_row_hash,
    clean_price,
    coerce_int_flag,
    extract_normalized_weight,
    normalize_sku_text,
    replace_nan_with_none,
    standardize_sku_units,
    strip_weight_tokens,
)
from backend.repositories import catalog_repository

logger = logging.getLogger("matchops.services.catalog_sync")


def download_sheet_csv(
    sheet_id: str,
    sheet_name: str,
    *,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    timeout: int = 30,
) -> str:
    """Downloads a Google Sheet tab as CSV text with exponential backoff retries."""
    encoded_name = sheet_name.replace(" ", "%20")
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={encoded_name}"
    )
    last_exception: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            text_data = resp.text
            if not text_data or not text_data.strip():
                raise ValueError(
                    f"Received empty response from Google Sheets for tab '{sheet_name}'."
                )
            return text_data
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


def _read_staged_or_remote_text(sheet_id: str | None, sheet_name: str) -> str:
    staged_path = Path(catalog_repository.resolve_staged_sheet_path(sheet_name))
    if staged_path.is_file() and staged_path.stat().st_size > 0:
        return staged_path.read_text(encoding="utf-8")
    if not sheet_id:
        raise ValueError(f"GOOGLE_SHEET_ID is required when staged sheet '{sheet_name}' is absent.")
    return download_sheet_csv(sheet_id, sheet_name)


def load_sheet_dataframe(
    sheet_id: str | None,
    sheet_name: str,
    header: int | None = 0,
    dtype: Any = None,
) -> pd.DataFrame:
    """Loads a staged or remote Google Sheet tab into a cleaned Pandas DataFrame."""
    content_bytes = _read_staged_or_remote_text(sheet_id, sheet_name).encode("utf-8")
    pl_kwargs: dict[str, Any] = {"ignore_errors": True}
    if header is None:
        pl_kwargs["has_header"] = False

    df = pl.read_csv(content_bytes, **pl_kwargs)
    df = df.select([c for c in df.columns if not c.startswith("Unnamed")])
    pd_df = df.to_pandas()

    if dtype is str:
        pd_df = pd_df.astype(str)
    elif isinstance(dtype, dict):
        for col, d_type in dtype.items():
            if col in pd_df.columns:
                pd_df[col] = pd_df[col].astype(d_type)
    return pd_df


def sync_catalog_to_database(
    sheet_id: str | None = None,
    domain: str = DOMAIN_MARKET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Incrementally diffs staged/Google catalog & brand sheets against PostgreSQL."""
    if not sheet_id:
        sheet_id = config.GOOGLE_SHEET_ID
    sheet_name = FOOD_CATALOG_SHEET if domain == DOMAIN_FOOD else MARKET_CATALOG_SHEET
    brand_sheet = "Food_Flavors" if domain == DOMAIN_FOOD else "Market_Brands"

    logger.info(
        f"[SYNC] [{domain.upper()}] Loading catalog and brands from staged cache / Google Sheets..."
    )

    catalog_text = _read_staged_or_remote_text(sheet_id, sheet_name)
    brands_text = _read_staged_or_remote_text(sheet_id, brand_sheet)

    catalog_pl = pl.read_csv(catalog_text.encode("utf-8"), ignore_errors=True)
    catalog_pl = catalog_pl.select([c for c in catalog_pl.columns if not c.startswith("Unnamed")])
    col_map = CATALOG_COL_MAP_FOOD if domain == DOMAIN_FOOD else CATALOG_COL_MAP_MARKET
    catalog_pl = catalog_pl.rename({k: v for k, v in col_map.items() if k in catalog_pl.columns})

    if "Name" in catalog_pl.columns:
        catalog_pl = catalog_pl.filter(
            (pl.col("Name").is_not_null())
            & (pl.col("Name").str.strip_chars() != "")
            & (pl.col("Name").str.to_lowercase() != "nan")
            & (pl.col("Name").str.to_lowercase() != "none")
        )

    brands_pl = pl.read_csv(brands_text.encode("utf-8"), ignore_errors=True)
    brands_pl = brands_pl.select([c for c in brands_pl.columns if not c.startswith("Unnamed")])
    brand_col = "Flavor Name" if domain == DOMAIN_FOOD else "Brand Name"
    if brand_col in brands_pl.columns:
        brands_pl = brands_pl.filter(
            (pl.col(brand_col).is_not_null())
            & (pl.col(brand_col).str.strip_chars() != "")
            & (pl.col(brand_col).str.to_lowercase() != "nan")
            & (pl.col(brand_col).str.to_lowercase() != "none")
        )

    old_cat_rows, old_brand_rows = catalog_repository.fetch_catalog_and_brand_hashes(domain)

    old_uids_map: dict[str, tuple[int, str]] = {}
    name_counts_old: dict[str, int] = {}
    for row_id, name, row_hash in old_cat_rows:
        raw_name = str(name).strip().lower()
        name_counts_old[raw_name] = name_counts_old.get(raw_name, 0) + 1
        old_uids_map[f"{raw_name}#occ_{name_counts_old[raw_name]}"] = (row_id, row_hash)

    old_brands_map: dict[str, tuple[int, str]] = {}
    brand_counts_old: dict[str, int] = {}
    for brand_id, name, row_hash in old_brand_rows:
        raw_name = str(name).strip().lower()
        brand_counts_old[raw_name] = brand_counts_old.get(raw_name, 0) + 1
        old_brands_map[f"{raw_name}#occ_{brand_counts_old[raw_name]}"] = (brand_id, row_hash)

    cat_df_raw = catalog_pl.to_pandas()
    new_uids_seen: set[str] = set()
    name_counts_new: dict[str, int] = {}
    cat_inserts: list[dict[str, Any]] = []
    cat_updates: list[dict[str, Any]] = []

    for row in cat_df_raw.to_dict(orient="records"):
        row = replace_nan_with_none(row)
        raw_name = str(row.get("Name", "")).strip().lower()
        name_counts_new[raw_name] = name_counts_new.get(raw_name, 0) + 1
        uid = f"{raw_name}#occ_{name_counts_new[raw_name]}"
        new_uids_seen.add(uid)

        row_hash = calculate_row_hash(row, domain=domain)
        name_val = str(row.get("Name", "") or "")
        clean_text = normalize_sku_text(standardize_sku_units(name_val))
        clean_no_weights = strip_weight_tokens(clean_text)
        w_extracted = extract_normalized_weight(clean_text)
        weight_val_num = (
            w_extracted[0]
            if (
                isinstance(w_extracted, tuple)
                and len(w_extracted) > 0
                and w_extracted[0] is not None
            )
            else None
        )
        token_count = len(clean_text.split())

        values = {
            "name": row.get("Name"),
            "brand": row.get("Brand") if domain == DOMAIN_MARKET else None,
            "flavor": row.get("Flavor") if domain == DOMAIN_FOOD else None,
            "basictype": row.get("basictype"),
            "generic_keywords": row.get("Generic keywords"),
            "price": clean_price(row.get("Price")),
            "category": row.get("category") if domain == DOMAIN_MARKET else None,
            "region": row.get("region") if domain == DOMAIN_FOOD else None,
            "description": row.get("Description"),
            "clean_text": clean_text,
            "clean_no_weights": clean_no_weights,
            "weight_val": json.dumps(weight_val_num) if weight_val_num is not None else None,
            "token_count": token_count,
            "row_hash": row_hash,
        }
        if uid not in old_uids_map:
            cat_inserts.append({"domain": domain, **values})
        else:
            row_id, old_hash = old_uids_map[uid]
            if old_hash != row_hash:
                cat_updates.append({"id": row_id, **values})

    deleted_cat_ids = [old_uids_map[uid][0] for uid in set(old_uids_map) - new_uids_seen]

    brands_df_raw = brands_pl.to_pandas()
    new_brands_seen: set[str] = set()
    brand_counts_new: dict[str, int] = {}
    brand_inserts: list[dict[str, Any]] = []
    brand_updates: list[dict[str, Any]] = []

    for row in brands_df_raw.to_dict(orient="records"):
        row = replace_nan_with_none(row)
        name_val = row.get("Flavor Name") if domain == DOMAIN_FOOD else row.get("Brand Name")
        raw_name = str(name_val).strip().lower()
        brand_counts_new[raw_name] = brand_counts_new.get(raw_name, 0) + 1
        uid = f"{raw_name}#occ_{brand_counts_new[raw_name]}"
        new_brands_seen.add(uid)

        row_str = str(name_val) + str(row.get("Aliases", "")) + CACHE_SALT
        row_hash = hashlib.sha256(row_str.encode("utf-8")).hexdigest()

        values = {
            "name": name_val,
            "aliases": row.get("Aliases"),
            "is_weak": coerce_int_flag(row.get("Is_Weak")),
            "is_meat": coerce_int_flag(row.get("Is_Meat")),
            "is_vegetable": coerce_int_flag(row.get("Is_Vegetable")),
            "is_seafood": coerce_int_flag(row.get("Is_Seafood")),
            "row_hash": row_hash,
        }
        if uid not in old_brands_map:
            brand_inserts.append({"domain": domain, **values})
        else:
            brand_id, old_hash = old_brands_map[uid]
            if old_hash != row_hash:
                brand_updates.append({"id": brand_id, **values})

    deleted_brand_ids = [old_brands_map[uid][0] for uid in set(old_brands_map) - new_brands_seen]

    catalog_repository.apply_catalog_incremental_diff(
        cat_inserts=cat_inserts,
        cat_updates=cat_updates,
        deleted_cat_ids=deleted_cat_ids,
        brand_inserts=brand_inserts,
        brand_updates=brand_updates,
        deleted_brand_ids=deleted_brand_ids,
    )
    logger.info(
        f"[SYNC] [{domain.upper()}] Database Incremental Update Complete: "
        f"Catalog ({len(cat_inserts)} inserted, {len(cat_updates)} updated, {len(deleted_cat_ids)} deleted), "
        f"Brands ({len(brand_inserts)} inserted, {len(brand_updates)} updated, {len(deleted_brand_ids)} deleted)."
    )
    return query_catalog_from_database(domain)


def query_catalog_from_database(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Queries `catalog_items` and `brand_flavors` and restores normalized DataFrames."""
    cat_df, brands_df = catalog_repository.fetch_catalog_and_brand_rows(domain)

    entities_list = []
    for _, val in cat_df["entities_json"].items() if "entities_json" in cat_df.columns else []:
        if pd.isna(val) or not val:
            entities_list.append(None)
        else:
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    entities_list.append(
                        [
                            {k: set(v) if isinstance(v, list) else v for k, v in item.items()}
                            if isinstance(item, dict)
                            else item
                            for item in parsed
                        ]
                    )
                else:
                    entities_list.append(parsed)
            except Exception:
                entities_list.append(None)
    if entities_list:
        cat_df["entities"] = entities_list

    weight_list = []
    if "weight_val" in cat_df.columns:
        for idx, val in cat_df["weight_val"].items():
            if pd.isna(val) or not val:
                weight_list.append(None)
                continue
            try:
                parsed = json.loads(str(val))
            except Exception:
                weight_list.append(None)
                continue
            if isinstance(parsed, (int, float)):
                clean_t = cat_df.at[idx, "clean_text"]
                if pd.isna(clean_t) or not clean_t:
                    weight_list.append((float(parsed), None, None))
                else:
                    weight_list.append(extract_normalized_weight(clean_t))
            else:
                weight_list.append(tuple(parsed) if isinstance(parsed, list) else parsed)
        cat_df["weight_val"] = weight_list

    cat_df = cat_df.rename(
        columns={
            "generic_keywords": "Generic keywords",
            "name": "Name",
            "price": "Price",
            "description": "Description",
        }
    )
    if domain == DOMAIN_MARKET:
        cat_df = cat_df.rename(columns={"brand": "Brand"})
    else:
        cat_df = cat_df.rename(columns={"flavor": "Flavor"})

    _, uids = build_occurrence_uids(cat_df, name_col="Name")
    cat_df["db_uid"] = uids

    brands_df = brands_df.rename(
        columns={
            "name": "Flavor Name" if domain == DOMAIN_FOOD else "Brand Name",
            "aliases": "Aliases",
            "is_weak": "Is_Weak",
            "is_meat": "Is_Meat",
            "is_vegetable": "Is_Vegetable",
            "is_seafood": "Is_Seafood",
        }
    )
    for col in ("Is_Weak", "Is_Meat", "Is_Vegetable", "Is_Seafood"):
        if col in brands_df.columns:
            brands_df[col] = brands_df[col].astype(bool)

    return cat_df, brands_df


def sync_classifier_dictionaries_to_database(
    sheet_id: str | None = None,
    domain: str = DOMAIN_MARKET,
) -> dict[str, list[str]]:
    """Loads GK/BT/Region-Category taxonomy sheets and replaces PostgreSQL dictionaries."""
    if not sheet_id:
        sheet_id = config.GOOGLE_SHEET_ID
    dicts: dict[str, list[str]] = {}
    third_tag_key = get_third_tag_col(domain)
    third_tag_sheet = "Region" if domain == DOMAIN_FOOD else "Category"
    domain_suffix = "Food" if domain == DOMAIN_FOOD else "Market"
    sheets_to_load = {
        "gk": f"{domain_suffix}_GK",
        "bt": f"{domain_suffix}_BT",
        third_tag_key: f"{domain_suffix}_{third_tag_sheet}",
    }

    for key, sheet_name in sheets_to_load.items():
        df = load_sheet_dataframe(sheet_id, sheet_name, header=None, dtype=str)
        keywords = [k for k in df.iloc[:, 0].dropna().str.strip().tolist() if k]
        dicts[key] = keywords

    catalog_repository.replace_classifier_dictionaries(domain, dicts)
    logger.info(f"[SYNC] Imported classifier dictionaries into PostgreSQL ({domain}).")
    return dicts


def sync_bt_gk_map_to_database(
    sheet_id: str | None = None,
    domain: str = DOMAIN_MARKET,
    cat_df: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Builds BT->GK mappings from `cat_df` or PostgreSQL and persists them."""
    grouped: dict[str, set[str]] = defaultdict(set)
    if cat_df is not None and not cat_df.empty:
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
            for raw_bt, raw_gk in zip(cat_df[col_bt].tolist(), cat_df[col_gk].tolist()):
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
    else:
        for bt_val, gk_val in catalog_repository.fetch_catalog_tag_source_rows(domain):
            bt_clean = str(bt_val).strip()
            if not bt_clean:
                continue
            for k in str(gk_val).split(","):
                k_clean = k.strip()
                if k_clean:
                    grouped[bt_clean].add(k_clean)

    result: dict[str, list[str]] = {}
    inserts: list[dict[str, Any]] = []
    for bt, gks_set in grouped.items():
        sorted_gks = sorted(gks_set)
        result[bt] = sorted_gks
        inserts.append({"domain": domain, "bt": bt, "gks": ", ".join(sorted_gks)})

    catalog_repository.replace_bt_gk_mappings(domain, inserts)
    logger.info(
        f"[SYNC] [{domain.upper()}] Imported {len(inserts)} BT-GK map entries into PostgreSQL."
    )
    return result


def compute_and_store_dictionary_counts(domain: str, cat_df: pd.DataFrame | None = None) -> None:
    """Precomputes tag frequencies across catalog items and stores them in PostgreSQL."""
    cat_col = "category" if domain == DOMAIN_MARKET else "region"
    brand_col = "Brand" if domain == DOMAIN_MARKET else "Flavor"
    brand_db_col = "brand" if domain == DOMAIN_MARKET else "flavor"

    gk_c: Counter = Counter()
    bt_c: Counter = Counter()
    cat_c: Counter = Counter()
    brand_c: Counter = Counter()

    if cat_df is not None and len(cat_df) > 0:
        if "Generic keywords" in cat_df.columns:
            for gks in cat_df["Generic keywords"].dropna():
                for item in str(gks).split(","):
                    t = item.strip().lower()
                    if t:
                        gk_c[t] += 1
        if "basictype" in cat_df.columns:
            for bts in cat_df["basictype"].dropna():
                for item in str(bts).split(","):
                    t = item.strip().lower()
                    if t:
                        bt_c[t] += 1
        if cat_col in cat_df.columns:
            for cats in cat_df[cat_col].dropna():
                t = str(cats).strip().lower()
                if t:
                    cat_c[t] += 1
        if brand_col in cat_df.columns:
            for brs in cat_df[brand_col].dropna():
                for item in str(brs).split(","):
                    t = item.strip().lower()
                    if t:
                        brand_c[t] += 1
    else:
        for gks, bts, cats, brs in catalog_repository.fetch_catalog_frequency_source_rows(
            domain, cat_col, brand_db_col
        ):
            if gks:
                for item in str(gks).split(","):
                    t = item.strip().lower()
                    if t:
                        gk_c[t] += 1
            if bts:
                for item in str(bts).split(","):
                    t = item.strip().lower()
                    if t:
                        bt_c[t] += 1
            if cats:
                t = str(cats).strip().lower()
                if t:
                    cat_c[t] += 1
            if brs:
                for item in str(brs).split(","):
                    t = item.strip().lower()
                    if t:
                        brand_c[t] += 1

    catalog_repository.update_dictionary_catalog_counts(domain, gk_c, bt_c, cat_c, brand_c)
    logger.info(
        f"[SYNC] [{domain.upper()}] Precomputed and stored dictionary counts in PostgreSQL."
    )
