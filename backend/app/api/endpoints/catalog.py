import logging
import re
from typing import Dict, Optional
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from backend.app.services.catalog_service import (
    clean_record_nans,
    check_changes_for_domain,
    escape_meili_filter_value,
    get_bt_gk_cache,
    get_catalog_and_brands,
    get_classifier_dicts,
    get_build_task_callable,
    trigger_build_cache,
)
from backend.app.services.meilisearch_service import get_meili_client, is_meili_healthy
from engine import config as engine_config
from engine.core.db import ensure_db_initialized

logger = logging.getLogger("matchops.catalog_api")
router = APIRouter()

# Re-export data helpers for backward-compatible references
__all__ = [
    "router",
    "get_catalog_and_brands",
    "get_classifier_dicts",
    "get_bt_gk_cache",
]


@router.get("/catalog/summary-stats")
def get_catalog_summary_stats(domain: str = "market"):
    """
    Returns total document counts for all 6 catalog datasets (catalog, gk, bt, category, brands, bt_gk_map)
    in a single lightweight call (<5ms).
    """
    if domain not in ("market", "food"):
        raise HTTPException(status_code=400, detail="Invalid domain. Must be 'market' or 'food'.")

    # 1. Try Meilisearch
    if is_meili_healthy():
        try:
            client = get_meili_client()
            cat_index = engine_config.MEILI_INDEX_MARKET if domain == "market" else engine_config.MEILI_INDEX_FOOD
            dict_index = engine_config.MEILI_INDEX_MARKET_DICTS if domain == "market" else engine_config.MEILI_INDEX_FOOD_DICTS

            cat_stats = client.index(cat_index).get_stats()
            cat_count = getattr(cat_stats, "number_of_documents", 0) if not isinstance(cat_stats, dict) else cat_stats.get("numberOfDocuments", 0)

            dict_search = client.index(dict_index).search("", {"facets": ["dataset"], "limit": 0})
            facets = dict_search.get("facetDistribution", {}).get("dataset", {})

            return {
                "domain": domain,
                "catalog": cat_count,
                "gk": facets.get("gk", 0),
                "bt": facets.get("bt", 0),
                "category": facets.get("category", 0),
                "brands": facets.get("brands", 0),
                "bt_gk_map": facets.get("bt_gk_map", 0),
            }
        except Exception as meili_err:
            logger.warning(f"[STATS] Failed to fetch summary stats from Meilisearch: {meili_err}. Falling back to SQLite.")

    # 2. Fast SQLite query fallback
    conn = ensure_db_initialized()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM catalog_items WHERE domain = ?", (domain,))
        cat_count = cur.fetchone()[0]

        cur.execute("SELECT tag_type, COUNT(*) FROM classifier_dictionaries WHERE domain = ? GROUP BY tag_type", (domain,))
        dict_counts = dict(cur.fetchall())

        cur.execute("SELECT COUNT(*) FROM brand_flavors WHERE domain = ?", (domain,))
        brand_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM bt_gk_map WHERE domain = ?", (domain,))
        map_count = cur.fetchone()[0]

        return {
            "domain": domain,
            "catalog": cat_count,
            "gk": dict_counts.get("gk", 0),
            "bt": dict_counts.get("bt", 0),
            "category": dict_counts.get("region" if domain == "food" else "category", 0),
            "brands": brand_count,
            "bt_gk_map": map_count,
        }
    finally:
        conn.close()


@router.get("/catalog")
def search_catalog(
    dataset: str = "catalog",
    domain: str = "market",
    query: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    region: Optional[str] = None,
    category: Optional[str] = None,
    gk_contains: Optional[str] = None,
    brand: Optional[str] = None,
    basictype: Optional[str] = None,
):
    if domain not in ("market", "food"):
        raise HTTPException(status_code=400, detail="Invalid domain. Must be 'market' or 'food'.")
        
    dataset = dataset.lower().strip()
    valid_datasets = ("catalog", "gk", "bt", "category", "brands", "bt_gk_map")
    if dataset not in valid_datasets:
        raise HTTPException(status_code=400, detail=f"Invalid dataset. Must be one of {valid_datasets}")

    # =========================================================================
    # 1. CATALOG DATASET (Uses Meilisearch with fallback to Pandas/SQLite)
    # =========================================================================
    if dataset == "catalog":
        if is_meili_healthy():
            try:
                client = get_meili_client()
                index_name = engine_config.MEILI_INDEX_MARKET if domain == "market" else engine_config.MEILI_INDEX_FOOD
                index = client.index(index_name)
                
                # Build filter list
                meili_filters = []
                if min_price is not None:
                    meili_filters.append(f"price >= {min_price}")
                if max_price is not None:
                    meili_filters.append(f"price <= {max_price}")
                
                if region:
                    meili_filters.append(f"region = \"{escape_meili_filter_value(region)}\"")
                if category:
                    meili_filters.append(f"category = \"{escape_meili_filter_value(category)}\"")
                if gk_contains:
                    gk_terms = [t.strip() for t in gk_contains.split(",") if t.strip()]
                    for term in gk_terms:
                        meili_filters.append(f"gk = \"{escape_meili_filter_value(term)}\"")
                if brand:
                    meili_filters.append(f"brand = \"{escape_meili_filter_value(brand)}\"")
                if basictype:
                    meili_filters.append(f"bt = \"{escape_meili_filter_value(basictype)}\"")
                
                filter_str = " AND ".join(meili_filters) if meili_filters else None
                
                # Build sort mapping
                sort_map = {
                    "name": "name",
                    "brand": "brand",
                    "flavor": "flavor",
                    "price": "price",
                    "basictype": "bt",
                    "bt": "bt",
                    "category": "category",
                    "region": "region",
                    "description": "description"
                }
                sort_by_meili = sort_map.get(sort_by, sort_by)
                sort_list = []
                if sort_by_meili:
                    sort_list.append(f"{sort_by_meili}:{sort_order}")
                
                search_params = {
                    "limit": page_size,
                    "offset": (page - 1) * page_size,
                }
                if filter_str:
                    search_params["filter"] = filter_str
                if sort_list:
                    search_params["sort"] = sort_list
                
                q = query if query else ""
                res = index.search(q, search_params)
                
                hits = res.get("hits", [])
                total = res.get("totalHits") or res.get("estimatedTotalHits", 0)
                
                results_list = []
                for hit in hits:
                    doc = dict(hit)
                    for field in ("gk", "bt", "category", "region", "brand", "flavor"):
                        val = doc.get(field)
                        if isinstance(val, list):
                            doc[field] = ", ".join(val)
                    doc["count"] = 1
                    results_list.append(doc)
                
                return {
                    "results": results_list,
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total + page_size - 1) // page_size if total > 0 else 1
                }
            except Exception as e:
                logger.warning(f"[MEILI] Search failed: {e}. Falling back to Pandas in-memory search.")

        # Fallback to Pandas in-memory search for catalog
        try:
            catalog_df, brands_df = get_catalog_and_brands(domain)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load catalog data: {str(e)}")

        df = catalog_df.copy()
        if "id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].astype(str)

        if min_price is not None:
            df = df[df["Price"] >= min_price]
        if max_price is not None:
            df = df[df["Price"] <= max_price]
        if region and "region" in df.columns:
            df = df[df["region"].astype(str).str.contains(region, case=False, na=False)]
        if category and "category" in df.columns:
            df = df[df["category"].astype(str).str.contains(category, case=False, na=False)]
        if gk_contains and "Generic keywords" in df.columns:
            gk_terms = [t.strip() for t in gk_contains.split(",") if t.strip()]
            for term in gk_terms:
                df = df[df["Generic keywords"].astype(str).str.contains(re.escape(term), case=False, na=False)]
        if brand and "Brand" in df.columns:
            df = df[df["Brand"].astype(str).str.contains(brand, case=False, na=False)]
        if basictype and "basictype" in df.columns:
            df = df[df["basictype"].astype(str).str.contains(basictype, case=False, na=False)]

        if query:
            q = query.lower()
            mask = pd.Series(False, index=df.index)
            searchable_cols = ["Name", "Brand", "Flavor", "basictype", "category", "region", "Generic keywords"]
            for col in searchable_cols:
                if col in df.columns:
                    mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
            df = df[mask]

        if sort_by:
            col_map = {
                "name": "Name",
                "brand": "Brand",
                "flavor": "Flavor",
                "price": "Price",
                "basictype": "basictype",
                "bt": "basictype",
                "category": "category",
                "region": "region",
                "gk": "Generic keywords",
                "description": "Description"
            }
            target_col = col_map.get(sort_by, sort_by)
            if target_col in df.columns:
                df = df.sort_values(by=target_col, ascending=(sort_order == "asc"))

        total = len(df)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_df = df.iloc[start:end]

        records = paginated_df.to_dict(orient="records")
        for r in records:
            r["name"] = r.get("Name")
            r["brand"] = r.get("Brand")
            r["price"] = r.get("Price")
            r["sellercategory"] = r.get("SellerCategory")
            r["category"] = r.get("category")
            r["gk"] = r.get("Generic keywords")
            r["bt"] = r.get("basictype")
            r["merchant"] = r.get("Merchant")
            r["flavor"] = r.get("Flavor")
            r["description"] = r.get("Description")
            r["region"] = r.get("region")
            r["count"] = 1

        return {
            "results": clean_record_nans(records),
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 1
        }

    # =========================================================================
    # 2. DICTIONARIES DATASETS (gk, bt, category, brands, bt_gk_map)
    # Fast path: Meilisearch (<2ms) -> Fast fallback: Direct SQLite (<1ms)
    # NEVER loads the 80,000 catalog rows into memory!
    # =========================================================================
    if is_meili_healthy():
        try:
            client = get_meili_client()
            dict_index_name = engine_config.MEILI_INDEX_MARKET_DICTS if domain == "market" else engine_config.MEILI_INDEX_FOOD_DICTS
            index = client.index(dict_index_name)

            meili_filters = [f'dataset = "{dataset}"']
            filter_str = " AND ".join(meili_filters)

            # Sorting mapping
            if sort_by:
                sort_field = sort_by
                if sort_field in ("sku_count",):
                    sort_field = "count"
                elif sort_field in ("gks", "basictype", "brand_name", "flavor_name"):
                    sort_field = "name"
                sort_list = [f"{sort_field}:{sort_order}"]
            else:
                sort_list = ["count:desc"]

            search_params = {
                "limit": page_size,
                "offset": (page - 1) * page_size,
                "filter": filter_str,
                "sort": sort_list,
            }
            q = query if query else ""
            res = index.search(q, search_params)

            hits = res.get("hits", [])
            total = res.get("totalHits") or res.get("estimatedTotalHits", 0)

            results_list = []
            for hit in hits:
                doc = dict(hit)
                if dataset == "bt_gk_map":
                    doc["bt"] = doc.get("bt") or doc.get("name")
                elif dataset == "brands":
                    if domain == "food":
                        doc["flavor_name"] = doc.get("name")
                    else:
                        doc["brand_name"] = doc.get("name")
                results_list.append(doc)

            return {
                "results": results_list,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size if total > 0 else 1
            }
        except Exception as e:
            logger.warning(f"[MEILI] Dictionary search failed: {e}. Falling back to SQLite direct query.")

    # Direct SQLite Fallback (Zero catalog loading)
    conn = ensure_db_initialized()
    try:
        cur = conn.cursor()
        offset = (page - 1) * page_size
        q_param = f"%{query.strip().lower()}%" if query else None

        if dataset in ("gk", "bt", "category"):
            tag_type = "region" if (domain == "food" and dataset == "category") else dataset
            if sort_by == "name":
                sort_col = "tag"
                order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
            else:
                sort_col = "catalog_count"
                order_dir = "ASC" if sort_by and sort_order.lower() == "asc" else "DESC"
            
            if q_param:
                cur.execute(
                    f"SELECT COUNT(*) FROM classifier_dictionaries WHERE domain = ? AND tag_type = ? AND LOWER(tag) LIKE ?",
                    (domain, tag_type, q_param)
                )
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT id, tag, catalog_count FROM classifier_dictionaries WHERE domain = ? AND tag_type = ? AND LOWER(tag) LIKE ? ORDER BY {sort_col} {order_dir} LIMIT ? OFFSET ?",
                    (domain, tag_type, q_param, page_size, offset)
                )
            else:
                cur.execute(
                    f"SELECT COUNT(*) FROM classifier_dictionaries WHERE domain = ? AND tag_type = ?",
                    (domain, tag_type)
                )
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT id, tag, catalog_count FROM classifier_dictionaries WHERE domain = ? AND tag_type = ? ORDER BY {sort_col} {order_dir} LIMIT ? OFFSET ?",
                    (domain, tag_type, page_size, offset)
                )
            
            rows = cur.fetchall()
            results = [{"id": f"{dataset}_{r[0]}", "name": r[1], "count": r[2]} for r in rows]

        elif dataset == "brands":
            if sort_by == "name":
                sort_col = "name"
                order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
            elif sort_by in ("is_weak", "is_meat", "is_vegetable", "is_seafood"):
                sort_col = sort_by
                order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
            else:
                sort_col = "catalog_count"
                order_dir = "ASC" if sort_by and sort_order.lower() == "asc" else "DESC"

            if q_param:
                cur.execute(
                    f"SELECT COUNT(*) FROM brand_flavors WHERE domain = ? AND (LOWER(name) LIKE ? OR LOWER(COALESCE(aliases, '')) LIKE ?)",
                    (domain, q_param, q_param)
                )
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT id, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, catalog_count FROM brand_flavors WHERE domain = ? AND (LOWER(name) LIKE ? OR LOWER(COALESCE(aliases, '')) LIKE ?) ORDER BY {sort_col} {order_dir} LIMIT ? OFFSET ?",
                    (domain, q_param, q_param, page_size, offset)
                )
            else:
                cur.execute(f"SELECT COUNT(*) FROM brand_flavors WHERE domain = ?", (domain,))
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT id, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, catalog_count FROM brand_flavors WHERE domain = ? ORDER BY {sort_col} {order_dir} LIMIT ? OFFSET ?",
                    (domain, page_size, offset)
                )
            
            rows = cur.fetchall()
            results = []
            for r in rows:
                item = {
                    "id": str(r[0]),
                    "name": r[1],
                    "aliases": r[2],
                    "count": r[7],
                }
                if domain == "market":
                    item["brand_name"] = r[1]
                    item["is_weak"] = bool(r[3])
                else:
                    item["flavor_name"] = r[1]
                    item["is_meat"] = bool(r[4])
                    item["is_vegetable"] = bool(r[5])
                    item["is_seafood"] = bool(r[6])
                results.append(item)

        elif dataset == "bt_gk_map":
            if sort_by in ("name", "bt"):
                sort_col = "basictype"
                order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
            elif sort_by == "gk_count":
                sort_col = "gk_count"
                order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
            else:
                sort_col = "catalog_count"
                order_dir = "ASC" if sort_by and sort_order.lower() == "asc" else "DESC"

            if q_param:
                cur.execute(
                    f"SELECT COUNT(*) FROM bt_gk_map WHERE domain = ? AND (LOWER(basictype) LIKE ? OR LOWER(generic_keywords) LIKE ?)",
                    (domain, q_param, q_param)
                )
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT id, basictype, generic_keywords, gk_count, catalog_count FROM bt_gk_map WHERE domain = ? AND (LOWER(basictype) LIKE ? OR LOWER(generic_keywords) LIKE ?) ORDER BY {sort_col} {order_dir} LIMIT ? OFFSET ?",
                    (domain, q_param, q_param, page_size, offset)
                )
            else:
                cur.execute(f"SELECT COUNT(*) FROM bt_gk_map WHERE domain = ?", (domain,))
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT id, basictype, generic_keywords, gk_count, catalog_count FROM bt_gk_map WHERE domain = ? ORDER BY {sort_col} {order_dir} LIMIT ? OFFSET ?",
                    (domain, page_size, offset)
                )

            
            rows = cur.fetchall()
            results = [
                {
                    "id": f"map_{r[0]}",
                    "bt": r[1],
                    "name": r[1],
                    "gks": r[2],
                    "gk_count": r[3],
                    "count": r[4]
                }
                for r in rows
            ]

        return {
            "results": clean_record_nans(results),
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 1
        }
    finally:
        conn.close()



@router.get("/catalog/check-sync")
def check_catalog_sync(limit: int = Query(50, ge=0, le=500)):
    """Checks Google Sheets and returns summary counts & capped row-level preview changes."""
    try:
        results = {
            domain: check_changes_for_domain(domain, limit=limit)
            for domain in ("market", "food")
        }
        return {
            "status": "success",
            "has_changes": any(r["new_count"] > 0 or r["changed_count"] > 0 for r in results.values()),
            "details": results
        }
    except Exception as e:
        logger.error(f"Failed to check sync status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check sync status: {str(e)}")


@router.post("/catalog/build-cache")
def build_catalog_cache(background_tasks: BackgroundTasks):
    """Trigger background catalog cache build, Qdrant sync, and classifier training."""
    started, message = trigger_build_cache()
    if not started:
        status_code = "ignored"
        return {"status": status_code, "message": message}
        
    background_tasks.add_task(get_build_task_callable())
    return {"status": "started", "message": message}


@router.post("/catalog/refresh")
def refresh_catalog_cache(background_tasks: BackgroundTasks):
    """Forces cache invalidation and rebuilds cache in background (delegates to build_catalog_cache)."""
    return build_catalog_cache(background_tasks)
