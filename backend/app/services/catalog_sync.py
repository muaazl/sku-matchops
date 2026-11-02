import hashlib
import json
import logging
import math
import os
from collections import Counter, defaultdict

import pandas as pd
import polars as pl
from sqlalchemy import text

from backend.app.db.session import get_connection
from engine import config
from engine.data_pipeline.cache_manager import calculate_row_hash, clean_price
from engine.data_pipeline.ingestion import DataIngestion
from engine.nlp.text_cleaner import TextPipeline

logger = logging.getLogger("matchops.catalog_sync")


def _nan_to_none(row: dict) -> dict:
    """pandas NaN -> None, so it binds as SQL NULL (Postgres rejects NaN for TEXT columns)."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def _flag(val) -> int:
    return int(val) if val is not None and not pd.isna(val) else 0


def sync_catalog_to_database(
    sheet_id: str = config.GOOGLE_SHEET_ID,
    domain: str = config.DOMAIN_MARKET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads or reads staged Google Sheets data, incrementally diffs against
    PostgreSQL catalog_items and brand_flavors tables, persists changes,
    and updates the local Feather cache.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    hash_file = os.path.join(config.CACHE_DIR, f"{domain}_source_hashes.json")

    sheet_name = (
        config.FOOD_CATALOG_SHEET if domain == config.DOMAIN_FOOD else config.MARKET_CATALOG_SHEET
    )
    brand_sheet = "Food_Flavors" if domain == config.DOMAIN_FOOD else "Market_Brands"

    logger.info(
        f"[SYNC] [{domain.upper()}] Loading catalog and brands from staged cache / Google Sheets..."
    )

    cat_staged = DataIngestion._get_staged_path(sheet_name)
    brand_staged = DataIngestion._get_staged_path(brand_sheet)

    if os.path.exists(cat_staged) and os.path.getsize(cat_staged) > 0:
        with open(cat_staged, encoding="utf-8") as f:
            catalog_text = f.read()
    else:
        catalog_text = DataIngestion._download_sheet_with_retries(sheet_id, sheet_name)
    catalog_hash = hashlib.md5(catalog_text.encode("utf-8")).hexdigest()

    if os.path.exists(brand_staged) and os.path.getsize(brand_staged) > 0:
        with open(brand_staged, encoding="utf-8") as f:
            brands_text = f.read()
    else:
        brands_text = DataIngestion._download_sheet_with_retries(sheet_id, brand_sheet)
    brands_hash = hashlib.md5(brands_text.encode("utf-8")).hexdigest()

    # Clean/Parse Polars DataFrames
    catalog_pl = pl.read_csv(catalog_text.encode("utf-8"), ignore_errors=True)
    catalog_pl = catalog_pl.select([c for c in catalog_pl.columns if not c.startswith("Unnamed")])
    col_map = (
        config.CATALOG_COL_MAP_FOOD
        if domain == config.DOMAIN_FOOD
        else config.CATALOG_COL_MAP_MARKET
    )
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

    brand_col = "Flavor Name" if domain == config.DOMAIN_FOOD else "Brand Name"
    if brand_col in brands_pl.columns:
        brands_pl = brands_pl.filter(
            (pl.col(brand_col).is_not_null())
            & (pl.col(brand_col).str.strip_chars() != "")
            & (pl.col(brand_col).str.to_lowercase() != "nan")
            & (pl.col(brand_col).str.to_lowercase() != "none")
        )

    conn = get_connection()
    try:
        old_rows = conn.execute(
            text(
                "SELECT id, name, row_hash FROM catalog_items WHERE domain = :domain ORDER BY id ASC"
            ),
            {"domain": domain},
        ).fetchall()
        old_uids_map = {}
        name_counts_old = {}
        for row_id, name, row_hash in old_rows:
            raw_name = str(name).strip().lower()
            name_counts_old[raw_name] = name_counts_old.get(raw_name, 0) + 1
            uid = f"{raw_name}#occ_{name_counts_old[raw_name]}"
            old_uids_map[uid] = (row_id, row_hash)

        old_brands = conn.execute(
            text(
                "SELECT id, name, row_hash FROM brand_flavors WHERE domain = :domain ORDER BY id ASC"
            ),
            {"domain": domain},
        ).fetchall()
        old_brands_map = {}
        brand_counts_old = {}
        for brand_id, name, row_hash in old_brands:
            raw_name = str(name).strip().lower()
            brand_counts_old[raw_name] = brand_counts_old.get(raw_name, 0) + 1
            uid = f"{raw_name}#occ_{brand_counts_old[raw_name]}"
            old_brands_map[uid] = (brand_id, row_hash)

        cat_df_raw = catalog_pl.to_pandas()
        new_uids_seen = set()
        name_counts_new = {}

        cat_inserts = []
        cat_updates = []

        for row in cat_df_raw.to_dict(orient="records"):
            row = _nan_to_none(row)
            raw_name = str(row.get("Name", "")).strip().lower()
            name_counts_new[raw_name] = name_counts_new.get(raw_name, 0) + 1
            uid = f"{raw_name}#occ_{name_counts_new[raw_name]}"
            new_uids_seen.add(uid)

            row_hash = calculate_row_hash(row, domain=domain)
            name_val = str(row.get("Name", "") or "")
            clean_text = TextPipeline.normalize_final(TextPipeline.standardize_units(name_val))
            clean_no_weights = TextPipeline.strip_weights(clean_text)
            w_extracted = TextPipeline.extract_weight_feature(clean_text)
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
                "brand": row.get("Brand") if domain == config.DOMAIN_MARKET else None,
                "flavor": row.get("Flavor") if domain == config.DOMAIN_FOOD else None,
                "basictype": row.get("basictype"),
                "generic_keywords": row.get("Generic keywords"),
                "price": clean_price(row.get("Price")),
                "category": row.get("category") if domain == config.DOMAIN_MARKET else None,
                "region": row.get("region") if domain == config.DOMAIN_FOOD else None,
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

        if cat_inserts:
            conn.execute(
                text("""
                INSERT INTO catalog_items (
                    domain, name, brand, flavor, basictype, generic_keywords, price, category, region, description,
                    clean_text, clean_no_weights, weight_val, token_count, row_hash
                ) VALUES (
                    :domain, :name, :brand, :flavor, :basictype, :generic_keywords, :price, :category, :region, :description,
                    :clean_text, :clean_no_weights, :weight_val, :token_count, :row_hash
                )
            """),
                cat_inserts,
            )
        if cat_updates:
            conn.execute(
                text("""
                UPDATE catalog_items
                SET name = :name, brand = :brand, flavor = :flavor, basictype = :basictype,
                    generic_keywords = :generic_keywords, price = :price, category = :category,
                    region = :region, description = :description, clean_text = :clean_text,
                    clean_no_weights = :clean_no_weights, weight_val = :weight_val,
                    token_count = :token_count, row_hash = :row_hash
                WHERE id = :id
            """),
                cat_updates,
            )

        deleted_cat_ids = [old_uids_map[uid][0] for uid in set(old_uids_map) - new_uids_seen]
        if deleted_cat_ids:
            conn.execute(
                text("DELETE FROM catalog_items WHERE id = ANY(:ids)"), {"ids": deleted_cat_ids}
            )

        cat_inserted, cat_updated, cat_deleted = (
            len(cat_inserts),
            len(cat_updates),
            len(deleted_cat_ids),
        )

        # Brands
        brands_df_raw = brands_pl.to_pandas()
        new_brands_seen = set()
        brand_counts_new = {}
        brand_inserts = []
        brand_updates = []

        for row in brands_df_raw.to_dict(orient="records"):
            row = _nan_to_none(row)
            name_val = (
                row.get("Flavor Name") if domain == config.DOMAIN_FOOD else row.get("Brand Name")
            )
            raw_name = str(name_val).strip().lower()
            brand_counts_new[raw_name] = brand_counts_new.get(raw_name, 0) + 1
            uid = f"{raw_name}#occ_{brand_counts_new[raw_name]}"
            new_brands_seen.add(uid)

            row_str = str(name_val) + str(row.get("Aliases", "")) + config.CACHE_SALT
            row_hash = hashlib.md5(row_str.encode("utf-8")).hexdigest()

            values = {
                "name": name_val,
                "aliases": row.get("Aliases"),
                "is_weak": _flag(row.get("Is_Weak")),
                "is_meat": _flag(row.get("Is_Meat")),
                "is_vegetable": _flag(row.get("Is_Vegetable")),
                "is_seafood": _flag(row.get("Is_Seafood")),
                "row_hash": row_hash,
            }
            if uid not in old_brands_map:
                brand_inserts.append({"domain": domain, **values})
            else:
                brand_id, old_hash = old_brands_map[uid]
                if old_hash != row_hash:
                    brand_updates.append({"id": brand_id, **values})

        if brand_inserts:
            conn.execute(
                text("""
                INSERT INTO brand_flavors (
                    domain, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, row_hash
                ) VALUES (
                    :domain, :name, :aliases, :is_weak, :is_meat, :is_vegetable, :is_seafood, :row_hash
                )
            """),
                brand_inserts,
            )
        if brand_updates:
            conn.execute(
                text("""
                UPDATE brand_flavors
                SET name = :name, aliases = :aliases, is_weak = :is_weak, is_meat = :is_meat,
                    is_vegetable = :is_vegetable, is_seafood = :is_seafood, row_hash = :row_hash
                WHERE id = :id
            """),
                brand_updates,
            )

        deleted_brand_ids = [
            old_brands_map[uid][0] for uid in set(old_brands_map) - new_brands_seen
        ]
        if deleted_brand_ids:
            conn.execute(
                text("DELETE FROM brand_flavors WHERE id = ANY(:ids)"),
                {"ids": deleted_brand_ids},
            )

        brands_inserted, brands_updated, brands_deleted = (
            len(brand_inserts),
            len(brand_updates),
            len(deleted_brand_ids),
        )

        conn.commit()

        logger.info(
            f"[SYNC] [{domain.upper()}] Database Incremental Update Complete: "
            f"Catalog ({cat_inserted} inserted, {cat_updated} updated, {cat_deleted} deleted), "
            f"Brands ({brands_inserted} inserted, {brands_updated} updated, {brands_deleted} deleted)."
        )

        with open(hash_file, "w") as f:
            json.dump({"catalog": catalog_hash, "brands": brands_hash}, f)

    except Exception as e:
        conn.rollback()
        logger.error(f"[SYNC] Transaction rollback due to error: {e}")
        raise e
    finally:
        conn.close()

    # Re-query DB to produce clean DataFrames and write Feather caches
    cat_df, brands_df = query_catalog_from_database(domain)

    # Write high-performance Feather files
    try:
        import pyarrow.feather as feather

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
        logger.info(f"[SYNC] Cached Feather files successfully for '{domain}' domain.")
    except Exception as fe_err:
        logger.error(f"[SYNC] Failed to write Feather cache files: {fe_err}")

    # Clear engine in-memory cache so fresh data is loaded
    DataIngestion.clear_mem_cache(domain)

    return cat_df, brands_df


def query_catalog_from_database(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Queries catalog_items and brand_flavors from PostgreSQL and returns standardized DataFrames."""
    conn = get_connection()
    p = {"domain": domain}
    try:
        cat_df = pd.read_sql_query(
            text("SELECT * FROM catalog_items WHERE domain = :domain"), conn, params=p
        )
        brands_df = pd.read_sql_query(
            text("SELECT * FROM brand_flavors WHERE domain = :domain"), conn, params=p
        )

        # Decode entities_json
        entities_list = []
        for _, val in cat_df["entities_json"].items() if "entities_json" in cat_df.columns else []:
            if pd.isna(val) or not val:
                entities_list.append(None)
            else:
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        res = [
                            {k: set(v) if isinstance(v, list) else v for k, v in item.items()}
                            if isinstance(item, dict)
                            else item
                            for item in parsed
                        ]
                        entities_list.append(res)
                    else:
                        entities_list.append(parsed)
                except Exception:
                    entities_list.append(None)
        if entities_list:
            cat_df["entities"] = entities_list

        # Decode weight_val
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
                        weight_list.append(TextPipeline.extract_weight_feature(clean_t))
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

        if domain == config.DOMAIN_MARKET:
            cat_df = cat_df.rename(columns={"brand": "Brand"})
        else:
            cat_df = cat_df.rename(columns={"flavor": "Flavor"})

        raw_names = cat_df["Name"].fillna("").astype(str).str.strip().str.lower()
        occ_counts = raw_names.groupby(raw_names).cumcount() + 1
        cat_df["db_uid"] = raw_names + "#occ_" + occ_counts.astype(str)

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

        return cat_df, brands_df
    finally:
        conn.close()


def sync_classifier_dictionaries_to_database(
    sheet_id: str = config.GOOGLE_SHEET_ID,
    domain: str = config.DOMAIN_MARKET,
) -> dict[str, list[str]]:
    """Fetches dictionary tags from staged/online sheets and synchronizes to PostgreSQL."""
    dicts = {}
    third_tag_key = config.get_third_tag_col(domain)
    third_tag_sheet = "Region" if domain == config.DOMAIN_FOOD else "Category"
    domain_suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
    sheets_to_load = {
        "gk": f"{domain_suffix}_GK",
        "bt": f"{domain_suffix}_BT",
        third_tag_key: f"{domain_suffix}_{third_tag_sheet}",
    }

    conn = get_connection()
    try:
        conn.execute(
            text("DELETE FROM classifier_dictionaries WHERE domain = :domain"),
            {"domain": domain},
        )

        for key, sheet_name in sheets_to_load.items():
            df = DataIngestion._fetch_sheet_as_csv(sheet_id, sheet_name, header=None, dtype=str)
            keywords = df.iloc[:, 0].dropna().str.strip().tolist()
            keywords = [k for k in keywords if k]

            dicts[key] = keywords
            if keywords:
                conn.execute(
                    text(
                        "INSERT INTO classifier_dictionaries (domain, tag_type, tag) VALUES (:domain, :tag_type, :tag)"
                    ),
                    [{"domain": domain, "tag_type": key, "tag": kw} for kw in keywords],
                )

        conn.commit()
        logger.info(f"[SYNC] Imported classifier dictionaries into PostgreSQL ({domain}).")

        # Also write local JSON cache
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        dicts_cache = os.path.join(config.CACHE_DIR, f"{domain}_classifier_dicts.json")
        with open(dicts_cache, "w", encoding="utf-8") as f:
            json.dump(dicts, f, indent=2)

        return dicts
    except Exception as e:
        conn.rollback()
        logger.error(f"[SYNC] Failed to import classifier dictionaries: {e}")
        raise e
    finally:
        conn.close()


def sync_bt_gk_map_to_database(
    sheet_id: str = config.GOOGLE_SHEET_ID,
    domain: str = config.DOMAIN_MARKET,
    cat_df: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Builds BT->GK mappings and synchronizes them to PostgreSQL."""
    conn = get_connection()
    try:
        grouped = defaultdict(set)
        if cat_df is not None and not cat_df.empty:
            col_bt = next(
                (c for c in cat_df.columns if c.strip().lower() in ("basictype", "basic type")),
                None,
            )
            col_gk = next(
                (
                    c
                    for c in cat_df.columns
                    if c.strip().lower()
                    in ("generickeywords", "generic keywords", "generic_keywords")
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
            cat_rows = conn.execute(
                text(
                    "SELECT basictype, generic_keywords FROM catalog_items WHERE domain = :domain AND basictype IS NOT NULL AND generic_keywords IS NOT NULL"
                ),
                {"domain": domain},
            ).fetchall()
            for bt_val, gk_val in cat_rows:
                bt_clean = str(bt_val).strip()
                if not bt_clean:
                    continue
                for k in str(gk_val).split(","):
                    k_clean = k.strip()
                    if k_clean:
                        grouped[bt_clean].add(k_clean)

        result = {}
        inserts = []
        for bt, gks_set in grouped.items():
            sorted_gks = sorted(gks_set)
            gks_str = ", ".join(sorted_gks)
            result[bt] = sorted_gks
            inserts.append({"domain": domain, "bt": bt, "gks": gks_str})

        conn.execute(text("DELETE FROM bt_gk_map WHERE domain = :domain"), {"domain": domain})
        if inserts:
            conn.execute(
                text(
                    "INSERT INTO bt_gk_map (domain, basictype, generic_keywords) VALUES (:domain, :bt, :gks)"
                ),
                inserts,
            )
        conn.commit()
        logger.info(
            f"[SYNC] [{domain.upper()}] Imported {len(inserts)} BT-GK map entries into PostgreSQL."
        )
        return result
    except Exception as e:
        conn.rollback()
        logger.error(f"[SYNC] [{domain.upper()}] Failed to build and sync BT-GK map: {e}")
        raise e
    finally:
        conn.close()


def compute_and_store_dictionary_counts(domain: str, cat_df: pd.DataFrame | None = None):
    """
    Precomputes the occurrence frequencies of all dictionary tags across catalog items
    and persists the counts into the PostgreSQL database.
    """
    conn = get_connection()
    try:
        p = {"domain": domain}
        cat_col = "category" if domain == config.DOMAIN_MARKET else "region"
        brand_col = "Brand" if domain == config.DOMAIN_MARKET else "Flavor"
        brand_db_col = "brand" if domain == config.DOMAIN_MARKET else "flavor"

        gk_c = Counter()
        bt_c = Counter()
        cat_c = Counter()
        brand_c = Counter()

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
            for gks, bts, cats, brs in conn.execute(
                text(
                    f"SELECT generic_keywords, basictype, {cat_col}, {brand_db_col} FROM catalog_items WHERE domain = :domain"
                ),
                p,
            ).fetchall():
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

        # 1. Update classifier_dictionaries
        dict_rows = conn.execute(
            text("SELECT id, tag_type, tag FROM classifier_dictionaries WHERE domain = :domain"),
            p,
        ).fetchall()
        dict_updates = []
        for row_id, tag_type, tag in dict_rows:
            t = tag.strip().lower()
            if tag_type == "gk":
                cnt = gk_c.get(t, 0)
            elif tag_type == "bt":
                cnt = bt_c.get(t, 0)
            else:
                cnt = cat_c.get(t, 0)
            dict_updates.append({"cnt": cnt, "id": row_id})
        if dict_updates:
            conn.execute(
                text("UPDATE classifier_dictionaries SET catalog_count = :cnt WHERE id = :id"),
                dict_updates,
            )

        # 2. Update brand_flavors
        brand_rows = conn.execute(
            text("SELECT id, name FROM brand_flavors WHERE domain = :domain"), p
        ).fetchall()
        brand_updates = []
        for row_id, name in brand_rows:
            cnt = brand_c.get(name.strip().lower(), 0)
            brand_updates.append({"cnt": cnt, "id": row_id})
        if brand_updates:
            conn.execute(
                text("UPDATE brand_flavors SET catalog_count = :cnt WHERE id = :id"),
                brand_updates,
            )

        # 3. Update bt_gk_map
        map_rows = conn.execute(
            text("SELECT id, basictype, generic_keywords FROM bt_gk_map WHERE domain = :domain"),
            p,
        ).fetchall()
        map_updates = []
        for row_id, bt, gks in map_rows:
            cat_cnt = bt_c.get(bt.strip().lower(), 0)
            gk_cnt = len([k for k in str(gks).split(",") if k.strip()])
            map_updates.append({"cat_cnt": cat_cnt, "gk_cnt": gk_cnt, "id": row_id})
        if map_updates:
            conn.execute(
                text(
                    "UPDATE bt_gk_map SET catalog_count = :cat_cnt, gk_count = :gk_cnt WHERE id = :id"
                ),
                map_updates,
            )

        conn.commit()
        logger.info(
            f"[SYNC] [{domain.upper()}] Precomputed and stored dictionary counts in PostgreSQL."
        )
    except Exception as e:
        conn.rollback()
        logger.error(f"[SYNC] [{domain.upper()}] Failed to store dictionary counts: {e}")
        raise e
    finally:
        conn.close()
