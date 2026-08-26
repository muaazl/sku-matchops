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
    get_column_counter,
    get_build_task_callable,
    trigger_build_cache,
)
from backend.app.services.meilisearch_service import get_meili_client, is_meili_healthy
from engine import config as engine_config

logger = logging.getLogger("matchops.catalog_api")
router = APIRouter()

# Re-export data helpers for backward-compatible references
__all__ = [
    "router",
    "get_catalog_and_brands",
    "get_classifier_dicts",
    "get_bt_gk_cache",
]


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

    # 1. CATALOG DATASET - Try Meilisearch search first
    if dataset == "catalog":
        try:
            if is_meili_healthy():
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

    # Fallback to Pandas in-memory search or metadata dataset queries
    try:
        catalog_df, brands_df = get_catalog_and_brands(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load catalog data: {str(e)}")

    cat_col = "category" if domain == "market" else "region"
    brand_col = "Brand" if domain == "market" else "Flavor"

    gk_counter = get_column_counter(catalog_df, "Generic keywords", split_comma=True) if dataset == "gk" else None
    bt_counter = get_column_counter(catalog_df, "basictype", split_comma=True) if dataset in ("bt", "bt_gk_map") else None
    cat_counter = get_column_counter(catalog_df, cat_col, split_comma=False) if dataset == "category" else None
    brand_counter = get_column_counter(catalog_df, brand_col, split_comma=True) if dataset == "brands" else None

    results = []

    # 1b. Fallback Pandas implementation
    if dataset == "catalog":
        df = catalog_df.copy()
        
        if "id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].astype(str)

        # Filters
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

        # Search Query
        if query:
            q = query.lower()
            mask = pd.Series(False, index=df.index)
            searchable_cols = ["Name", "Brand", "Flavor", "basictype", "category", "region", "Generic keywords"]
            for col in searchable_cols:
                if col in df.columns:
                    mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
            df = df[mask]

        # Sorting
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
                "genericKeywords": "Generic keywords",
                "sellercategory": "SellerCategory",
                "sellerCategory": "SellerCategory",
                "merchant": "Merchant",
                "description": "Description"
            }
            target_col = col_map.get(sort_by, sort_by)
            if target_col in df.columns:
                df = df.sort_values(by=target_col, ascending=(sort_order == "asc"))

        records = df.to_dict(orient="records")
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
            
        results = clean_record_nans(records)

    # 2. GK DATASET
    elif dataset == "gk":
        dicts = get_classifier_dicts(domain)
        gk_list = dicts.get("gk", [])
        
        raw_results = [
            {"id": f"gk_{i}", "name": tag, "count": gk_counter[tag.lower()]}
            for i, tag in enumerate(gk_list)
        ]
        if query:
            q = query.lower()
            raw_results = [r for r in raw_results if q in r["name"].lower()]
            
        reverse = (sort_order == "desc")
        key_func = (lambda x: x["count"]) if sort_by == "count" else (lambda x: x["name"].lower())
        raw_results.sort(key=key_func, reverse=reverse)
        results = raw_results

    # 3. BT DATASET
    elif dataset == "bt":
        dicts = get_classifier_dicts(domain)
        bt_list = dicts.get("bt", [])
        
        raw_results = [
            {"id": f"bt_{i}", "name": tag, "count": bt_counter[tag.lower()]}
            for i, tag in enumerate(bt_list)
        ]
        if query:
            q = query.lower()
            raw_results = [r for r in raw_results if q in r["name"].lower()]
            
        reverse = (sort_order == "desc")
        key_func = (lambda x: x["count"]) if sort_by == "count" else (lambda x: x["name"].lower())
        raw_results.sort(key=key_func, reverse=reverse)
        results = raw_results

    # 4. CATEGORY / REGION DATASET
    elif dataset == "category":
        dicts = get_classifier_dicts(domain)
        tag_key = "region" if domain == "food" else "category"
        tags_list = dicts.get(tag_key, [])
        
        raw_results = [
            {"id": f"cat_{i}", "name": tag, "count": cat_counter[tag.lower()]}
            for i, tag in enumerate(tags_list)
        ]
        if query:
            q = query.lower()
            raw_results = [r for r in raw_results if q in r["name"].lower()]
            
        reverse = (sort_order == "desc")
        key_func = (lambda x: x["count"]) if sort_by == "count" else (lambda x: x["name"].lower())
        raw_results.sort(key=key_func, reverse=reverse)
        results = raw_results

    # 5. BRANDS / FLAVORS DATASET
    elif dataset == "brands":
        df = brands_df.copy()
        
        if "id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].astype(str)
        records = df.to_dict(orient="records")
        raw_results = []

        if domain == "market":
            for r in records:
                b_name = str(r.get("Brand Name", ""))
                raw_results.append({
                    "id": r["id"],
                    "name": b_name,
                    "brand_name": b_name,
                    "aliases": r.get("Aliases"),
                    "is_weak": r.get("Is_Weak"),
                    "count": brand_counter[b_name.lower()]
                })
        else:
            for r in records:
                fl_name = str(r.get("Flavor Name", ""))
                raw_results.append({
                    "id": r["id"],
                    "name": fl_name,
                    "flavor_name": fl_name,
                    "aliases": r.get("Aliases"),
                    "is_meat": r.get("Is_Meat"),
                    "is_vegetable": r.get("Is_Vegetable"),
                    "is_seafood": r.get("Is_Seafood"),
                    "count": brand_counter[fl_name.lower()]
                })

        if query:
            q = query.lower()
            raw_results = [
                r for r in raw_results 
                if q in r["name"].lower() or (r["aliases"] and q in str(r["aliases"]).lower())
            ]

        reverse = (sort_order == "desc")
        if sort_by == "count":
            raw_results.sort(key=lambda x: x["count"], reverse=reverse)
        elif sort_by in ("is_weak", "is_meat", "is_vegetable", "is_seafood"):
            raw_results.sort(key=lambda x: bool(x.get(sort_by)), reverse=reverse)
        else:
            raw_results.sort(key=lambda x: x["name"].lower(), reverse=reverse)

        results = clean_record_nans(raw_results)

    # 6. BT-GK MAP DATASET
    elif dataset == "bt_gk_map":
        bt_gk_cache = get_bt_gk_cache(domain)
        bt_gk_map = bt_gk_cache.get("bt_gk_map", {})
        
        raw_results = [
            {
                "id": f"map_{i}",
                "bt": bt,
                "name": bt,
                "gks": ", ".join(gks) if isinstance(gks, list) else str(gks),
                "gk_count": len(gks) if isinstance(gks, list) else 0,
                "count": bt_counter[bt.lower()]
            }
            for i, (bt, gks) in enumerate(bt_gk_map.items())
        ]

        if query:
            q = query.lower()
            raw_results = [
                r for r in raw_results 
                if q in r["bt"].lower() or q in r["gks"].lower()
            ]

        reverse = (sort_order == "desc")
        if sort_by in ("count", "sku_count"):
            raw_results.sort(key=lambda x: x["count"], reverse=reverse)
        elif sort_by == "gk_count":
            raw_results.sort(key=lambda x: x["gk_count"], reverse=reverse)
        else:
            raw_results.sort(key=lambda x: x["bt"].lower(), reverse=reverse)

        results = raw_results

    # Pagination
    total = len(results)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_results = results[start:end]

    return {
        "results": paginated_results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 1
    }


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
