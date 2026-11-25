"""SQLAlchemy repository and staged-file path resolver for catalog and taxonomy data."""

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from backend.core import config
from backend.core.database import get_connection


def resolve_staged_sheet_path(sheet_name: str) -> Path:
    """Resolves the local staged CSV cache file path for a Google Sheet tab."""
    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in sheet_name) + ".csv"
    return Path(config.STAGING_DIR) / safe_name


def fetch_catalog_summary_counts(domain: str) -> dict[str, Any]:
    """Queries counts for all 6 catalog/dictionary datasets for the given domain."""
    with get_connection() as conn:
        params = {"domain": domain}
        cat_count = (
            conn.execute(
                text("SELECT COUNT(*) FROM catalog_items WHERE domain = :domain"), params
            ).scalar()
            or 0
        )
        dict_counts = dict(
            conn.execute(
                text(
                    "SELECT tag_type, COUNT(*) FROM classifier_dictionaries WHERE domain = :domain GROUP BY tag_type"
                ),
                params,
            ).fetchall()
        )
        brand_count = (
            conn.execute(
                text("SELECT COUNT(*) FROM brand_flavors WHERE domain = :domain"), params
            ).scalar()
            or 0
        )
        map_count = (
            conn.execute(
                text("SELECT COUNT(*) FROM bt_gk_map WHERE domain = :domain"), params
            ).scalar()
            or 0
        )
    return {
        "catalog": int(cat_count),
        "dict_counts": dict_counts,
        "brands": int(brand_count),
        "bt_gk_map": int(map_count),
    }


def search_catalog_items(
    *,
    domain: str,
    raw_q: str,
    offset: int,
    limit: int,
    sort_by: str | None = None,
    sort_order: str = "asc",
    min_price: float | None = None,
    max_price: float | None = None,
    region: str | None = None,
    category: str | None = None,
    gk_contains: str | None = None,
    brand: str | None = None,
    basictype: str | None = None,
) -> tuple[int, list[tuple]]:
    """Runs pg_trgm + ILIKE filtered search against `catalog_items`."""
    where_clauses = ["domain = :domain"]
    params: dict[str, Any] = {"domain": domain, "limit": limit, "offset": offset}

    if raw_q:
        params["q_like"] = f"%{raw_q}%"
        params["raw_q"] = raw_q
        where_clauses.append(
            "("
            "name ILIKE :q_like OR description ILIKE :q_like "
            "OR name % :raw_q OR description % :raw_q "
            "OR basictype ILIKE :q_like OR generic_keywords ILIKE :q_like "
            "OR brand ILIKE :q_like OR flavor ILIKE :q_like"
            ")"
        )
    if min_price is not None:
        where_clauses.append("price >= :min_price")
        params["min_price"] = min_price
    if max_price is not None:
        where_clauses.append("price <= :max_price")
        params["max_price"] = max_price
    if region:
        where_clauses.append("region ILIKE :region")
        params["region"] = f"%{region.strip()}%"
    if category:
        where_clauses.append("category ILIKE :category")
        params["category"] = f"%{category.strip()}%"
    if gk_contains:
        for idx, term in enumerate(t.strip() for t in gk_contains.split(",") if t.strip()):
            key = f"gk_{idx}"
            where_clauses.append(f"generic_keywords ILIKE :{key}")
            params[key] = f"%{term}%"
    if brand:
        col = "flavor" if domain == "food" else "brand"
        where_clauses.append(f"{col} ILIKE :brand")
        params["brand"] = f"%{brand.strip()}%"
    if basictype:
        where_clauses.append("basictype ILIKE :basictype")
        params["basictype"] = f"%{basictype.strip()}%"

    where_str = " WHERE " + " AND ".join(where_clauses)
    sort_map = {
        "name": "name",
        "price": "price",
        "brand": "brand",
        "flavor": "flavor",
        "basictype": "basictype",
        "bt": "basictype",
        "generic_keywords": "generic_keywords",
        "gk": "generic_keywords",
        "category": "category",
        "region": "region",
        "description": "description",
    }
    order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
    col_target = sort_map.get(sort_by) if sort_by else None
    if col_target:
        order_by_sql = f"{col_target} {order_dir} NULLS LAST, id ASC"
    elif raw_q:
        order_by_sql = (
            "GREATEST(similarity(COALESCE(name, ''), :raw_q), "
            "similarity(COALESCE(description, ''), :raw_q)) DESC, id ASC"
        )
    else:
        order_by_sql = "id ASC"

    conn = get_connection()
    try:
        total = int(
            conn.execute(text(f"SELECT COUNT(*) FROM catalog_items{where_str}"), params).scalar()
            or 0
        )
        rows = conn.execute(
            text(
                f"""
                SELECT id, name, brand, flavor, basictype, generic_keywords, price, category, region, description
                FROM catalog_items
                {where_str}
                ORDER BY {order_by_sql}
                LIMIT :limit OFFSET :offset
                """
            ),
            params,
        ).fetchall()
    finally:
        conn.close()
    return total, rows


def search_classifier_tags(
    *,
    domain: str,
    tag_type: str,
    raw_q: str,
    offset: int,
    limit: int,
    sort_by: str | None = None,
    sort_order: str = "asc",
) -> tuple[int, list[tuple]]:
    """Searches `classifier_dictionaries` tags using pg_trgm + ILIKE."""
    params: dict[str, Any] = {
        "domain": domain,
        "tag_type": tag_type,
        "limit": limit,
        "offset": offset,
    }
    where_clauses = ["domain = :domain", "tag_type = :tag_type"]
    if raw_q:
        params["q_like"] = f"%{raw_q}%"
        params["raw_q"] = raw_q
        where_clauses.append("(tag ILIKE :q_like OR tag % :raw_q)")

    where_str = " WHERE " + " AND ".join(where_clauses)
    if sort_by == "name":
        order_by = f"tag {'DESC' if sort_order.lower() == 'desc' else 'ASC'}"
    elif raw_q and not sort_by:
        order_by = "similarity(tag, :raw_q) DESC, catalog_count DESC"
    else:
        order_by = f"catalog_count {'ASC' if sort_by and sort_order.lower() == 'asc' else 'DESC'}"

    conn = get_connection()
    try:
        total = int(
            conn.execute(
                text(f"SELECT COUNT(*) FROM classifier_dictionaries{where_str}"), params
            ).scalar()
            or 0
        )
        rows = conn.execute(
            text(
                f"SELECT id, tag, catalog_count FROM classifier_dictionaries{where_str} "
                f"ORDER BY {order_by} LIMIT :limit OFFSET :offset"
            ),
            params,
        ).fetchall()
    finally:
        conn.close()
    return total, rows


def search_brand_flavors(
    *,
    domain: str,
    raw_q: str,
    offset: int,
    limit: int,
    sort_by: str | None = None,
    sort_order: str = "asc",
) -> tuple[int, list[tuple]]:
    """Searches `brand_flavors` entries using pg_trgm + ILIKE."""
    params: dict[str, Any] = {"domain": domain, "limit": limit, "offset": offset}
    where_clauses = ["domain = :domain"]
    if raw_q:
        params["q_like"] = f"%{raw_q}%"
        params["raw_q"] = raw_q
        where_clauses.append(
            "(name ILIKE :q_like OR COALESCE(aliases, '') ILIKE :q_like OR name % :raw_q)"
        )

    where_str = " WHERE " + " AND ".join(where_clauses)
    if sort_by == "name":
        order_by = f"name {'DESC' if sort_order.lower() == 'desc' else 'ASC'}"
    elif sort_by in ("is_weak", "is_meat", "is_vegetable", "is_seafood"):
        order_by = f"{sort_by} {'DESC' if sort_order.lower() == 'desc' else 'ASC'}"
    elif raw_q and not sort_by:
        order_by = "similarity(name, :raw_q) DESC, catalog_count DESC"
    else:
        order_by = f"catalog_count {'ASC' if sort_by and sort_order.lower() == 'asc' else 'DESC'}"

    conn = get_connection()
    try:
        total = int(
            conn.execute(text(f"SELECT COUNT(*) FROM brand_flavors{where_str}"), params).scalar()
            or 0
        )
        rows = conn.execute(
            text(
                f"SELECT id, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, catalog_count "
                f"FROM brand_flavors{where_str} ORDER BY {order_by} LIMIT :limit OFFSET :offset"
            ),
            params,
        ).fetchall()
    finally:
        conn.close()
    return total, rows


def search_bt_gk_mappings(
    *,
    domain: str,
    raw_q: str,
    offset: int,
    limit: int,
    sort_by: str | None = None,
    sort_order: str = "asc",
) -> tuple[int, list[tuple]]:
    """Searches `bt_gk_map` entries using pg_trgm + ILIKE."""
    params: dict[str, Any] = {"domain": domain, "limit": limit, "offset": offset}
    where_clauses = ["domain = :domain"]
    if raw_q:
        params["q_like"] = f"%{raw_q}%"
        params["raw_q"] = raw_q
        where_clauses.append(
            "(basictype ILIKE :q_like OR generic_keywords ILIKE :q_like OR basictype % :raw_q)"
        )

    where_str = " WHERE " + " AND ".join(where_clauses)
    if sort_by in ("name", "bt"):
        order_by = f"basictype {'DESC' if sort_order.lower() == 'desc' else 'ASC'}"
    elif sort_by == "gk_count":
        order_by = f"gk_count {'DESC' if sort_order.lower() == 'desc' else 'ASC'}"
    elif raw_q and not sort_by:
        order_by = "similarity(basictype, :raw_q) DESC, catalog_count DESC"
    else:
        order_by = f"catalog_count {'ASC' if sort_by and sort_order.lower() == 'asc' else 'DESC'}"

    conn = get_connection()
    try:
        total = int(
            conn.execute(text(f"SELECT COUNT(*) FROM bt_gk_map{where_str}"), params).scalar() or 0
        )
        rows = conn.execute(
            text(
                f"SELECT id, basictype, generic_keywords, gk_count, catalog_count "
                f"FROM bt_gk_map{where_str} ORDER BY {order_by} LIMIT :limit OFFSET :offset"
            ),
            params,
        ).fetchall()
    finally:
        conn.close()
    return total, rows


def fetch_catalog_items_for_diff(domain: str) -> pd.DataFrame:
    """Loads minimal `catalog_items` columns from PostgreSQL for change-preview diffing."""
    with get_connection() as conn:
        return pd.read_sql_query(
            text(
                "SELECT id, domain, name, brand, flavor, generic_keywords, price, description, row_hash "
                "FROM catalog_items WHERE domain = :domain"
            ),
            conn,
            params={"domain": domain},
        )


def fetch_catalog_and_brand_rows(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Queries raw `catalog_items` and `brand_flavors` DataFrames for a domain."""
    params = {"domain": domain}
    with get_connection() as conn:
        cat_df = pd.read_sql_query(
            text("SELECT * FROM catalog_items WHERE domain = :domain"), conn, params=params
        )
        brands_df = pd.read_sql_query(
            text("SELECT * FROM brand_flavors WHERE domain = :domain"), conn, params=params
        )
    return cat_df, brands_df


def fetch_classifier_dictionary_rows(domain: str) -> list[tuple[str, str]]:
    """Queries `(tag_type, tag)` rows from `classifier_dictionaries`."""
    with get_connection() as conn:
        return conn.execute(
            text(
                "SELECT tag_type, tag FROM classifier_dictionaries "
                "WHERE domain = :domain ORDER BY id"
            ),
            {"domain": domain},
        ).fetchall()


def fetch_bt_gk_map_rows(domain: str) -> list[tuple[str, str]]:
    """Queries `(basictype, generic_keywords)` rows from `bt_gk_map`."""
    with get_connection() as conn:
        return conn.execute(
            text("SELECT basictype, generic_keywords FROM bt_gk_map WHERE domain = :domain"),
            {"domain": domain},
        ).fetchall()


def fetch_catalog_tag_source_rows(domain: str) -> list[tuple[str, str]]:
    """Queries non-null `(basictype, generic_keywords)` pairs from `catalog_items`."""
    with get_connection() as conn:
        return conn.execute(
            text(
                "SELECT basictype, generic_keywords FROM catalog_items "
                "WHERE domain = :domain AND basictype IS NOT NULL AND generic_keywords IS NOT NULL"
            ),
            {"domain": domain},
        ).fetchall()


def fetch_catalog_and_brand_hashes(
    domain: str,
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Fetches existing `(id, name, row_hash)` rows for catalog items and brands in order."""
    with get_connection() as conn:
        old_cat = conn.execute(
            text(
                "SELECT id, name, row_hash FROM catalog_items WHERE domain = :domain ORDER BY id ASC"
            ),
            {"domain": domain},
        ).fetchall()
        old_brands = conn.execute(
            text(
                "SELECT id, name, row_hash FROM brand_flavors WHERE domain = :domain ORDER BY id ASC"
            ),
            {"domain": domain},
        ).fetchall()
    return old_cat, old_brands


def apply_catalog_incremental_diff(
    *,
    cat_inserts: list[dict[str, Any]],
    cat_updates: list[dict[str, Any]],
    deleted_cat_ids: list[int],
    brand_inserts: list[dict[str, Any]],
    brand_updates: list[dict[str, Any]],
    deleted_brand_ids: list[int],
) -> None:
    """Applies incremental catalog and brand inserts, updates, and deletes in one transaction."""
    conn = get_connection()
    try:
        if cat_inserts:
            conn.execute(
                text(
                    """
                    INSERT INTO catalog_items (
                        domain, name, brand, flavor, basictype, generic_keywords, price, category, region, description,
                        clean_text, clean_no_weights, weight_val, token_count, row_hash
                    ) VALUES (
                        :domain, :name, :brand, :flavor, :basictype, :generic_keywords, :price, :category, :region, :description,
                        :clean_text, :clean_no_weights, :weight_val, :token_count, :row_hash
                    )
                    """
                ),
                cat_inserts,
            )
        if cat_updates:
            conn.execute(
                text(
                    """
                    UPDATE catalog_items
                    SET name = :name, brand = :brand, flavor = :flavor, basictype = :basictype,
                        generic_keywords = :generic_keywords, price = :price, category = :category,
                        region = :region, description = :description, clean_text = :clean_text,
                        clean_no_weights = :clean_no_weights, weight_val = :weight_val,
                        token_count = :token_count, row_hash = :row_hash
                    WHERE id = :id
                    """
                ),
                cat_updates,
            )
        if deleted_cat_ids:
            conn.execute(
                text("DELETE FROM catalog_items WHERE id = ANY(:ids)"),
                {"ids": deleted_cat_ids},
            )

        if brand_inserts:
            conn.execute(
                text(
                    """
                    INSERT INTO brand_flavors (
                        domain, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, row_hash
                    ) VALUES (
                        :domain, :name, :aliases, :is_weak, :is_meat, :is_vegetable, :is_seafood, :row_hash
                    )
                    """
                ),
                brand_inserts,
            )
        if brand_updates:
            conn.execute(
                text(
                    """
                    UPDATE brand_flavors
                    SET name = :name, aliases = :aliases, is_weak = :is_weak, is_meat = :is_meat,
                        is_vegetable = :is_vegetable, is_seafood = :is_seafood, row_hash = :row_hash
                    WHERE id = :id
                    """
                ),
                brand_updates,
            )
        if deleted_brand_ids:
            conn.execute(
                text("DELETE FROM brand_flavors WHERE id = ANY(:ids)"),
                {"ids": deleted_brand_ids},
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def replace_classifier_dictionaries(domain: str, dicts: dict[str, list[str]]) -> None:
    """Replaces all `classifier_dictionaries` rows for `domain` atomically."""
    conn = get_connection()
    try:
        conn.execute(
            text("DELETE FROM classifier_dictionaries WHERE domain = :domain"),
            {"domain": domain},
        )
        rows = [
            {"domain": domain, "tag_type": key, "tag": kw}
            for key, keywords in dicts.items()
            for kw in keywords
        ]
        if rows:
            conn.execute(
                text(
                    "INSERT INTO classifier_dictionaries (domain, tag_type, tag) "
                    "VALUES (:domain, :tag_type, :tag)"
                ),
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def replace_bt_gk_mappings(domain: str, inserts: list[dict[str, Any]]) -> None:
    """Replaces all `bt_gk_map` rows for `domain` atomically."""
    conn = get_connection()
    try:
        conn.execute(text("DELETE FROM bt_gk_map WHERE domain = :domain"), {"domain": domain})
        if inserts:
            conn.execute(
                text(
                    "INSERT INTO bt_gk_map (domain, basictype, generic_keywords) "
                    "VALUES (:domain, :bt, :gks)"
                ),
                inserts,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_catalog_frequency_source_rows(
    domain: str, cat_col: str, brand_db_col: str
) -> list[tuple]:
    """Fetches tag columns from `catalog_items` to compute dictionary frequencies."""
    with get_connection() as conn:
        return conn.execute(
            text(
                f"SELECT generic_keywords, basictype, {cat_col}, {brand_db_col} "
                f"FROM catalog_items WHERE domain = :domain"
            ),
            {"domain": domain},
        ).fetchall()


def update_dictionary_catalog_counts(
    domain: str,
    gk_counts: Counter,
    bt_counts: Counter,
    cat_counts: Counter,
    brand_counts: Counter,
) -> None:
    """Updates precomputed catalog occurrence counts across dictionary tables."""
    conn = get_connection()
    params = {"domain": domain}
    try:
        dict_rows = conn.execute(
            text("SELECT id, tag_type, tag FROM classifier_dictionaries WHERE domain = :domain"),
            params,
        ).fetchall()
        dict_updates = []
        for row_id, tag_type, tag in dict_rows:
            t = tag.strip().lower()
            if tag_type == "gk":
                cnt = gk_counts.get(t, 0)
            elif tag_type == "bt":
                cnt = bt_counts.get(t, 0)
            else:
                cnt = cat_counts.get(t, 0)
            dict_updates.append({"cnt": cnt, "id": row_id})
        if dict_updates:
            conn.execute(
                text("UPDATE classifier_dictionaries SET catalog_count = :cnt WHERE id = :id"),
                dict_updates,
            )

        brand_rows = conn.execute(
            text("SELECT id, name FROM brand_flavors WHERE domain = :domain"), params
        ).fetchall()
        brand_updates = [
            {"cnt": brand_counts.get(name.strip().lower(), 0), "id": row_id}
            for row_id, name in brand_rows
        ]
        if brand_updates:
            conn.execute(
                text("UPDATE brand_flavors SET catalog_count = :cnt WHERE id = :id"),
                brand_updates,
            )

        map_rows = conn.execute(
            text("SELECT id, basictype, generic_keywords FROM bt_gk_map WHERE domain = :domain"),
            params,
        ).fetchall()
        map_updates = [
            {
                "cat_cnt": bt_counts.get(bt.strip().lower(), 0),
                "gk_cnt": len([k for k in str(gks).split(",") if k.strip()]),
                "id": row_id,
            }
            for row_id, bt, gks in map_rows
        ]
        if map_updates:
            conn.execute(
                text(
                    "UPDATE bt_gk_map SET catalog_count = :cat_cnt, gk_count = :gk_cnt WHERE id = :id"
                ),
                map_updates,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
