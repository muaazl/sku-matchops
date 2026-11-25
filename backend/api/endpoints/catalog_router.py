"""Catalog exploration, summary statistics, Feather snapshot export, and sync endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response

from backend.core.exceptions import ResourceNotFoundError, ServiceValidationError
from backend.services import catalog_service

router = APIRouter(tags=["Catalog"])


@router.get("/catalog/summary-stats")
def get_catalog_summary_stats(domain: str = "market") -> dict:
    """Returns counts across catalog items and taxonomy dictionaries for a domain."""
    try:
        return catalog_service.fetch_catalog_summary_stats(domain)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/catalog")
def search_catalog(
    dataset: str = "catalog",
    domain: str = "market",
    q: str | None = None,
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
) -> dict:
    """Searches catalog items, brands, and taxonomy dictionaries with optional filters."""
    search_term = q if q is not None else query
    try:
        return catalog_service.search_catalog(
            dataset=dataset,
            domain=domain,
            query=search_term,
            page=page,
            page_size=page_size,
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
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/catalog/check-sync")
def check_catalog_sync(limit: int = 50) -> dict:
    """Compares live Google Sheets catalog rows against PostgreSQL cached rows."""
    try:
        return catalog_service.check_all_domain_changes(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _initiate_cache_build(background_tasks: BackgroundTasks) -> dict[str, str]:
    started, message = catalog_service.trigger_build_cache()
    if started:
        background_tasks.add_task(catalog_service.get_build_task_callable())
    return {"status": "started" if started else "already_running", "message": message}


@router.post("/catalog/build-cache")
def build_catalog_cache(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Starts an asynchronous Google Sheets sync and ML Engine model cache rebuild."""
    return _initiate_cache_build(background_tasks)


@router.post("/catalog/refresh")
def refresh_catalog_cache(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Triggers a background catalog sync and cache refresh."""
    return _initiate_cache_build(background_tasks)


@router.get("/catalog/{domain}/snapshot/catalog.feather")
def get_catalog_feather_snapshot(domain: str) -> Response:
    """Exports the domain catalog DataFrame as LZ4-compressed Apache Feather bytes."""
    try:
        payload = catalog_service.export_catalog_feather(domain)
        return Response(content=payload, media_type="application/octet-stream")
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/catalog/{domain}/snapshot/brands.feather")
def get_brands_feather_snapshot(domain: str) -> Response:
    """Exports domain brands/flavors as LZ4-compressed Apache Feather bytes."""
    try:
        payload = catalog_service.export_brands_feather(domain)
        return Response(content=payload, media_type="application/octet-stream")
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/catalog/{domain}/snapshot/dictionaries.json")
def get_dictionaries_snapshot(domain: str) -> dict:
    """Exports classifier taxonomy dictionaries for a domain."""
    try:
        return catalog_service.export_dictionaries_json(domain)
    except (ServiceValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
