import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import text

from backend.app.core.security import require_platform_admin
from backend.app.db.session import get_connection
from backend.app.services.catalog_service import (
    check_changes_for_domain,
    clean_record_nans,
    get_bt_gk_cache,
    get_build_task_callable,
    get_catalog_and_brands,
    get_classifier_dicts,
    trigger_build_cache,
)

logger = logging.getLogger("matchops.catalog_api")
router = APIRouter()

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
    directly from PostgreSQL in a single lightweight call (<5ms).
    """
    if domain not in ("market", "food"):
        raise HTTPException(status_code=400, detail="Invalid domain. Must be 'market' or 'food'.")

    conn = get_connection()
    try:
        p = {"domain": domain}
        cat_count = (
            conn.execute(
                text("SELECT COUNT(*) FROM catalog_items WHERE domain = :domain"), p
            ).scalar()
            or 0
        )

        dict_counts = dict(
            conn.execute(
                text(
                    "SELECT tag_type, COUNT(*) FROM classifier_dictionaries WHERE domain = :domain GROUP BY tag_type"
                ),
                p,
            ).fetchall()
        )

        brand_count = (
            conn.execute(
                text("SELECT COUNT(*) FROM brand_flavors WHERE domain = :domain"), p
            ).scalar()
            or 0
        )

        map_count = (
            conn.execute(text("SELECT COUNT(*) FROM bt_gk_map WHERE domain = :domain"), p).scalar()
            or 0
        )

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
):
    """
    Fast catalog and dictionary search backed by PostgreSQL pg_trgm GIN indexing and ILIKE.
    """
    if domain not in ("market", "food"):
        raise HTTPException(status_code=400, detail="Invalid domain. Must be 'market' or 'food'.")

    dataset = dataset.lower().strip()
    valid_datasets = ("catalog", "gk", "bt", "category", "brands", "bt_gk_map")
    if dataset not in valid_datasets:
        raise HTTPException(
            status_code=400, detail=f"Invalid dataset. Must be one of {valid_datasets}"
        )

    conn = get_connection()
    try:
        offset = max(0, (page - 1) * page_size)
        limit = max(1, page_size)
        raw_q = (query or "").strip()

        # =========================================================================
        # 1. CATALOG DATASET (PostgreSQL pg_trgm + ILIKE with GIN Index)
        # =========================================================================
        if dataset == "catalog":
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

            # Sorting
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
                order_by_sql = "GREATEST(similarity(COALESCE(name, ''), :raw_q), similarity(COALESCE(description, ''), :raw_q)) DESC, id ASC"
            else:
                order_by_sql = "id ASC"

            count_sql = f"SELECT COUNT(*) FROM catalog_items{where_str}"
            total = conn.execute(text(count_sql), params).scalar() or 0

            select_sql = f"""
                SELECT id, name, brand, flavor, basictype, generic_keywords, price, category, region, description
                FROM catalog_items
                {where_str}
                ORDER BY {order_by_sql}
                LIMIT :limit OFFSET :offset
            """
            rows = conn.execute(text(select_sql), params).fetchall()

            results_list = []
            for r in rows:
                item = {
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
                results_list.append(item)

            return {
                "results": clean_record_nans(results_list),
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size if total > 0 else 1,
            }

        # =========================================================================
        # 2. DICTIONARIES DATASETS (gk, bt, category, brands, bt_gk_map)
        # =========================================================================
        p = {"domain": domain, "limit": limit, "offset": offset}
        if raw_q:
            p["q_like"] = f"%{raw_q}%"
            p["raw_q"] = raw_q

        if dataset in ("gk", "bt", "category"):
            tag_type = "region" if (domain == "food" and dataset == "category") else dataset
            p["tag_type"] = tag_type
            where_clauses = ["domain = :domain", "tag_type = :tag_type"]

            if raw_q:
                where_clauses.append("(tag ILIKE :q_like OR tag % :raw_q)")

            where_str = " WHERE " + " AND ".join(where_clauses)
            if sort_by == "name":
                order_by = f"tag {'DESC' if sort_order.lower() == 'desc' else 'ASC'}"
            elif raw_q and not sort_by:
                order_by = "similarity(tag, :raw_q) DESC, catalog_count DESC"
            else:
                order_by = (
                    f"catalog_count {'ASC' if sort_by and sort_order.lower() == 'asc' else 'DESC'}"
                )

            total = (
                conn.execute(
                    text(f"SELECT COUNT(*) FROM classifier_dictionaries{where_str}"), p
                ).scalar()
                or 0
            )
            rows = conn.execute(
                text(
                    f"SELECT id, tag, catalog_count FROM classifier_dictionaries{where_str} ORDER BY {order_by} LIMIT :limit OFFSET :offset"
                ),
                p,
            ).fetchall()
            results = [{"id": f"{dataset}_{r[0]}", "name": r[1], "count": r[2]} for r in rows]

        elif dataset == "brands":
            where_clauses = ["domain = :domain"]
            if raw_q:
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
                order_by = (
                    f"catalog_count {'ASC' if sort_by and sort_order.lower() == 'asc' else 'DESC'}"
                )

            total = (
                conn.execute(text(f"SELECT COUNT(*) FROM brand_flavors{where_str}"), p).scalar()
                or 0
            )
            rows = conn.execute(
                text(
                    f"SELECT id, name, aliases, is_weak, is_meat, is_vegetable, is_seafood, catalog_count FROM brand_flavors{where_str} ORDER BY {order_by} LIMIT :limit OFFSET :offset"
                ),
                p,
            ).fetchall()
            results = []
            for r in rows:
                item = {
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

        elif dataset == "bt_gk_map":
            where_clauses = ["domain = :domain"]
            if raw_q:
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
                order_by = (
                    f"catalog_count {'ASC' if sort_by and sort_order.lower() == 'asc' else 'DESC'}"
                )

            total = (
                conn.execute(text(f"SELECT COUNT(*) FROM bt_gk_map{where_str}"), p).scalar() or 0
            )
            rows = conn.execute(
                text(
                    f"SELECT id, basictype, generic_keywords, gk_count, catalog_count FROM bt_gk_map{where_str} ORDER BY {order_by} LIMIT :limit OFFSET :offset"
                ),
                p,
            ).fetchall()
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
    finally:
        conn.close()


@router.get("/catalog/check-sync")
def check_catalog_sync(limit: int = Query(50, ge=0, le=500)):
    """Checks Google Sheets and returns summary counts & capped row-level preview changes."""
    try:
        results = {
            domain: check_changes_for_domain(domain, limit=limit) for domain in ("market", "food")
        }
        return {
            "status": "success",
            "has_changes": any(
                r["new_count"] > 0 or r["changed_count"] > 0 for r in results.values()
            ),
            "details": results,
        }
    except Exception as e:
        logger.error(f"Failed to check sync status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check sync status: {str(e)}")


@router.post("/catalog/build-cache", dependencies=[Depends(require_platform_admin)])
def build_catalog_cache(background_tasks: BackgroundTasks):
    """Trigger background catalog cache build, Qdrant sync, and classifier training."""
    started, message = trigger_build_cache()
    if not started:
        status_code = "ignored"
        return {"status": status_code, "message": message}

    background_tasks.add_task(get_build_task_callable())
    return {"status": "started", "message": message}


@router.post("/catalog/refresh", dependencies=[Depends(require_platform_admin)])
def refresh_catalog_cache(background_tasks: BackgroundTasks):
    """Forces cache invalidation and rebuilds cache in background (delegates to build_catalog_cache)."""
    return build_catalog_cache(background_tasks)
