"""Catalog search, snapshot Feather export, Google Sheets diff preview, and background rebuilds."""

import io
import logging
import threading
from typing import Any

import pandas as pd
import polars as pl
import pyarrow.feather as feather

from backend.core import config
from backend.core.constants import FOOD_CATALOG_SHEET, MARKET_CATALOG_SHEET, get_third_tag_col
from backend.core.exceptions import ResourceNotFoundError, ServiceValidationError
from backend.core.helpers import (
    build_occurrence_uids,
    calculate_row_hash,
    format_catalog_field,
)
from backend.repositories import catalog_lock_repository, catalog_repository
from backend.services import catalog_sync_service, engine_gateway_service

logger = logging.getLogger("matchops.services.catalog")

_build_lock = threading.Lock()
_build_in_progress = False


def validate_domain(domain: str) -> None:
    """Validates that the catalog domain is either 'market' or 'food'."""
    if domain not in ("market", "food"):
        raise ServiceValidationError("Invalid domain. Must be 'market' or 'food'.")


def get_catalog_and_brands(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads catalog and brand/flavor DataFrames from PostgreSQL."""
    cat_df, brands_df = catalog_sync_service.query_catalog_from_database(domain)
    if cat_df.empty:
        raise RuntimeError(f"No catalog data found in PostgreSQL for domain '{domain}'.")
    return cat_df, brands_df


def get_classifier_dicts(domain: str) -> dict[str, Any]:
    """Loads classifier dictionaries (GK, BT, category/region tags) from PostgreSQL."""
    third_tag_key = get_third_tag_col(domain)
    dicts: dict[str, list[str]] = {"gk": [], "bt": [], third_tag_key: []}
    rows = catalog_repository.fetch_classifier_dictionary_rows(domain)
    if not rows:
        raise RuntimeError(f"No classifier dictionaries found in PostgreSQL for domain '{domain}'.")
    for tag_type, tag in rows:
        dicts.setdefault(tag_type, []).append(tag)
    return dicts


def get_bt_gk_cache(domain: str) -> dict[str, Any]:
    """Loads the BT-GK mapping dictionary from PostgreSQL."""
    try:
        rows = catalog_repository.fetch_bt_gk_map_rows(domain)
        if rows:
            bt_gk_map = {bt: [k.strip() for k in gks.split(",") if k.strip()] for bt, gks in rows}
            return {"bt_gk_map": bt_gk_map, "umbrella": {}, "third_tag_map": {}}
    except Exception as db_err:
        logger.warning(f"Failed to load BT-GK map from the database for domain {domain}: {db_err}")
    return {"bt_gk_map": {}, "umbrella": {}, "third_tag_map": {}}


def clean_record_nans(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converts pd.isna/NaN values in records to None for JSON serialization."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in records]


def fetch_catalog_summary_stats(domain: str) -> dict[str, Any]:
    """Returns counts across all 6 catalog/dictionary datasets for the given domain."""
    validate_domain(domain)
    counts = catalog_repository.fetch_catalog_summary_counts(domain)
    dict_counts = counts["dict_counts"]
    return {
        "domain": domain,
        "catalog": counts["catalog"],
        "gk": dict_counts.get("gk", 0),
        "bt": dict_counts.get("bt", 0),
        "category": dict_counts.get("region" if domain == "food" else "category", 0),
        "brands": counts["brands"],
        "bt_gk_map": counts["bt_gk_map"],
    }


def search_catalog(
    *,
    dataset: str = "catalog",
    domain: str = "market",
    query: str | None = None,
    page: int = 1,
    page_size: int = 25,
    sort_by: str | None = None,
    sort_order: str = "asc",
    min_price: float | None = None,
    max_price: float | None = None,
    region: str | None = None,
    category: str | None = None,
    gk_contains: str | None = None,
    brand: str | None = None,
    basictype: str | None = None,
) -> dict[str, Any]:
    """Searches catalog items or taxonomy dictionaries backed by pg_trgm + ILIKE."""
    validate_domain(domain)
    normalized_ds = dataset.lower().strip()
    valid_datasets = ("catalog", "gk", "bt", "category", "brands", "bt_gk_map")
    if normalized_ds not in valid_datasets:
        raise ServiceValidationError(f"Invalid dataset. Must be one of {valid_datasets}")

    offset = max(0, (page - 1) * page_size)
    limit = max(1, page_size)
    raw_q = (query or "").strip()

    if normalized_ds == "catalog":
        total, rows = catalog_repository.search_catalog_items(
            domain=domain,
            raw_q=raw_q,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            min_price=min_price,
            max_price=max_price,
            region=region,
            category=category,
            gk_contains=gk_contains,
            brand=brand,
            basictype=basictype,
        )
        results = [
            {
                "id": str(r[0]),
                "name": r[1],
                "brand": r[2],
                "flavor": r[3],
                "basictype": r[4],
                "generic_keywords": r[5],
                "price": r[6],
                "category": r[7],
                "region": r[8],
                "description": r[9],
                "count": 1,
            }
            for r in rows
        ]
    elif normalized_ds in ("gk", "bt", "category"):
        tag_type = "region" if (domain == "food" and normalized_ds == "category") else normalized_ds
        total, rows = catalog_repository.search_classifier_tags(
            domain=domain,
            tag_type=tag_type,
            raw_q=raw_q,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        results = [{"id": f"{normalized_ds}_{r[0]}", "name": r[1], "count": r[2]} for r in rows]
    elif normalized_ds == "brands":
        total, rows = catalog_repository.search_brand_flavors(
            domain=domain,
            raw_q=raw_q,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        results = []
        for r in rows:
            item: dict[str, Any] = {
                "id": str(r[0]),
                "name": r[1],
                "aliases": r[2],
                "count": r[7],
            }
            if domain == "market":
                item["is_weak"] = bool(r[3])
            else:
                item["is_meat"] = bool(r[4])
                item["is_vegetable"] = bool(r[5])
                item["is_seafood"] = bool(r[6])
            results.append(item)
    else:
        total, rows = catalog_repository.search_bt_gk_mappings(
            domain=domain,
            raw_q=raw_q,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        results = [
            {
                "id": f"map_{r[0]}",
                "bt": r[1],
                "gks": r[2],
                "gk_count": r[3],
                "count": r[4],
            }
            for r in rows
        ]

    return {
        "results": clean_record_nans(results),
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 1,
    }


def export_catalog_feather(domain: str) -> bytes:
    """Serializes the domain catalog DataFrame into LZ4-compressed Feather bytes."""
    validate_domain(domain)
    try:
        cat_df, _ = get_catalog_and_brands(domain)
    except RuntimeError as e:
        raise ResourceNotFoundError(str(e)) from e
    cat_feather = cat_df.copy()
    if "entities" in cat_feather.columns:
        cat_feather = cat_feather.drop(columns=["entities"])
    if "weight_val" in cat_feather.columns:
        cat_feather["weight_val"] = cat_feather["weight_val"].astype(str)
    buf = io.BytesIO()
    feather.write_feather(cat_feather.reset_index(drop=True), buf, compression="lz4")
    return buf.getvalue()


def export_brands_feather(domain: str) -> bytes:
    """Serializes the domain brand_flavors DataFrame into LZ4-compressed Feather bytes."""
    validate_domain(domain)
    try:
        _, brands_df = get_catalog_and_brands(domain)
    except RuntimeError as e:
        raise ResourceNotFoundError(str(e)) from e
    buf = io.BytesIO()
    feather.write_feather(brands_df.reset_index(drop=True), buf, compression="lz4")
    return buf.getvalue()


def export_dictionaries_json(domain: str) -> dict[str, Any]:
    """Fetches classifier dictionaries JSON payload for the domain."""
    validate_domain(domain)
    try:
        return get_classifier_dicts(domain)
    except RuntimeError as e:
        raise ResourceNotFoundError(str(e)) from e


def check_changes_for_domain(domain: str, limit: int = 50) -> dict[str, Any]:
    """Diffs live Google Sheets catalog data against PostgreSQL cached catalog rows."""
    cached_df = pd.DataFrame()
    old_hashes: dict[str, str] = {}
    cached_by_uid: dict[str, dict[str, Any]] = {}
    cached_by_name: dict[str, dict[str, Any]] = {}

    try:
        cached_df = catalog_repository.fetch_catalog_items_for_diff(domain)
        cached_df = cached_df.rename(
            columns={
                "generic_keywords": "Generic keywords",
                "name": "Name",
                "price": "Price",
                "description": "Description",
            }
        )
        if domain == "market":
            cached_df = cached_df.rename(columns={"brand": "Brand"})
        else:
            cached_df = cached_df.rename(columns={"flavor": "Flavor"})

        cached_raw_names, cached_uids = build_occurrence_uids(cached_df)
        if cached_uids:
            row_hash_col = (
                cached_df["row_hash"]
                if "row_hash" in cached_df.columns
                else pd.Series([""] * len(cached_df), index=cached_df.index)
            )
            cached_records = cached_df.to_dict("records")
            old_hashes = dict(zip(cached_uids, row_hash_col))
            cached_by_uid = dict(zip(cached_uids, cached_records))
            for raw_name, record in zip(cached_raw_names, cached_records):
                if raw_name not in cached_by_name:
                    cached_by_name[raw_name] = record
    except Exception as db_err:
        logger.warning(f"Failed to load cached items from the database: {db_err}")

    sheet_name = FOOD_CATALOG_SHEET if domain == "food" else MARKET_CATALOG_SHEET
    csv_text = catalog_sync_service.download_sheet_csv(
        str(config.GOOGLE_SHEET_ID or ""), sheet_name, max_retries=1, timeout=15
    )

    from backend.core.constants import CATALOG_COL_MAP_FOOD, CATALOG_COL_MAP_MARKET

    catalog_pl = pl.read_csv(csv_text.encode("utf-8"), ignore_errors=True)
    catalog_pl = catalog_pl.select([c for c in catalog_pl.columns if not c.startswith("Unnamed")])
    col_map = CATALOG_COL_MAP_FOOD if domain == "food" else CATALOG_COL_MAP_MARKET
    catalog_pl = catalog_pl.rename({k: v for k, v in col_map.items() if k in catalog_pl.columns})
    new_df = catalog_pl.to_pandas()

    if "Name" in new_df.columns:
        new_df = new_df[
            (new_df["Name"].astype(str).str.strip() != "")
            & (new_df["Name"].notna())
            & (new_df["Name"].astype(str).str.lower() != "nan")
            & (new_df["Name"].astype(str).str.lower() != "none")
        ]

    compare_cols = ["Name", "basictype", "Generic keywords", "Description", "Price"]
    if domain == "food":
        compare_cols.extend(["region", "Flavor"])
    else:
        compare_cols.extend(["category", "Brand"])

    new_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    total_new_count = 0
    total_changed_count = 0

    new_raw_names, new_uids = build_occurrence_uids(new_df)
    new_records = new_df.to_dict("records") if len(new_df) else []

    for idx, row, uid, raw_name in zip(new_df.index, new_records, new_uids, new_raw_names):
        row_hash = calculate_row_hash(row, domain=domain)
        cleaned = {c: format_catalog_field(c, row.get(c, "")) for c in compare_cols}

        if uid not in old_hashes:
            total_new_count += 1
            if len(new_rows) < limit:
                new_rows.append(
                    {
                        "index": idx,
                        "name": cleaned["Name"],
                        "details": {c: cleaned[c] for c in compare_cols if cleaned[c]},
                    }
                )
        elif old_hashes[uid] != row_hash:
            old_row = None
            if not cached_df.empty:
                if idx < len(cached_df) and str(cached_df.iloc[idx].get("Name", "")) == str(
                    row.get("Name", "")
                ):
                    old_row = cached_df.iloc[idx]
                else:
                    old_row = cached_by_uid.get(uid) or cached_by_name.get(raw_name)

            diffs: dict[str, dict[str, str]] = {}
            if old_row is not None:
                for c in compare_cols:
                    old_val = format_catalog_field(c, old_row.get(c, ""))
                    if old_val != cleaned[c]:
                        diffs[c] = {"old": old_val, "new": cleaned[c]}
            else:
                diffs = {"content": {"old": "(unknown)", "new": "updated"}}

            if diffs:
                total_changed_count += 1
                if len(changed_rows) < limit:
                    changed_rows.append({"index": idx, "name": cleaned["Name"], "diffs": diffs})

    return {
        "new_count": total_new_count,
        "changed_count": total_changed_count,
        "new_rows": new_rows,
        "changed_rows": changed_rows,
        "is_capped": total_new_count > len(new_rows) or total_changed_count > len(changed_rows),
    }


def check_all_domain_changes(limit: int = 50) -> dict[str, Any]:
    """Checks Google Sheets catalog diff previews for both market and food domains."""
    results = {
        domain: check_changes_for_domain(domain, limit=limit) for domain in ("market", "food")
    }
    return {
        "status": "success",
        "has_changes": any(r["new_count"] > 0 or r["changed_count"] > 0 for r in results.values()),
        "details": results,
    }


def execute_background_cache_build() -> None:
    """Syncs source sheets into PostgreSQL and signals the ML Engine to reload caches/models."""
    global _build_in_progress
    logger.info("Background cache build and pre-training started...")
    try:
        for domain in ("market", "food"):
            logger.info(f"Syncing {domain} catalog into PostgreSQL...")
            cat_df, _ = catalog_sync_service.sync_catalog_to_database(
                config.GOOGLE_SHEET_ID, domain=domain
            )
            try:
                catalog_sync_service.sync_classifier_dictionaries_to_database(
                    config.GOOGLE_SHEET_ID, domain=domain
                )
                catalog_sync_service.sync_bt_gk_map_to_database(
                    config.GOOGLE_SHEET_ID, domain=domain, cat_df=cat_df
                )
                catalog_sync_service.compute_and_store_dictionary_counts(domain, cat_df=cat_df)
            except Exception as dict_err:
                logger.warning(
                    f"Could not refresh dictionaries for {domain} in background build: {dict_err}"
                )

        logger.info("Notifying ML Engine to reload models and rebuild pipelines...")
        engine_gateway_service.reload_engine_models()
        logger.info("Background cache build and pre-training completed successfully.")
    except Exception as e:
        logger.error(f"Failed to build cache in background: {e}")
    finally:
        _build_in_progress = False


def trigger_build_cache() -> tuple[bool, str]:
    """Acquires distributed/local lock and initiates background catalog cache rebuilding."""
    global _build_in_progress
    if _build_in_progress:
        return False, "Cache building is already in progress."

    acquired = catalog_lock_repository.acquire_distributed_build_lock(ttl_seconds=1800)
    if acquired is False:
        return False, "Catalog cache build is already in progress across cluster."
    if acquired is None:
        if not _build_lock.acquire(blocking=False):
            return False, "Cache building already active."

    _build_in_progress = True
    return True, "Cache building and pre-training initiated in background."


def get_build_task_callable():
    """Returns a FastAPI background task wrapper releasing distributed and local locks."""

    def run_with_lock():
        global _build_in_progress
        try:
            execute_background_cache_build()
        finally:
            _build_in_progress = False
            catalog_lock_repository.release_distributed_build_lock()
            if _build_lock.locked():
                try:
                    _build_lock.release()
                except Exception:
                    pass

    return run_with_lock
