"""Food Portal client service for SSRF-safe CSV retrieval and SKU parsing."""

import csv
import io
import logging
import time
from typing import Any

import requests

from backend.core import config
from backend.core.exceptions import (
    ServiceValidationError,
    UnauthorizedPortalError,
    UpstreamGatewayError,
)
from backend.core.security import validate_outbound_url
from backend.schemas.sku_schema import SKUItem
from backend.services import audit_service

logger = logging.getLogger("matchops.services.portal")

# Portal error codes/messages indicating an invalid, blacklisted, or expired token.
AUTH_ERROR_CODES = {"MER-4007", "MER-4006"}
AUTH_ERROR_KEYWORDS = (
    "token blacklisted",
    "token expired",
    "invalid token",
    "unauthorized",
    "unauthenticated",
    "signature verification failed",
)


def parse_csv_text_to_skus(text: str) -> list[SKUItem]:
    """Parses raw CSV text into SKUItem rows, auto-detecting name/price/description/category."""
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    name_col = next(
        (f for f in fieldnames if f.lower().strip() in ("name", "sku_name", "sku", "title")), None
    )
    price_col = next((f for f in fieldnames if f.lower().strip() in ("price", "cost", "mrp")), None)
    desc_col = next((f for f in fieldnames if f.lower().strip() in ("description", "desc")), None)
    cat_col = next(
        (f for f in fieldnames if f.lower().strip() in ("category", "cat", "type")), None
    )

    if not name_col and fieldnames:
        name_col = fieldnames[0]

    skus: list[SKUItem] = []
    for row in reader:
        if not name_col:
            continue
        name_val = (row.get(name_col) or "").strip()
        if not name_val:
            continue

        price_val = 0.0
        if price_col:
            try:
                price_val = float(row.get(price_col) or 0)
            except ValueError:
                pass
        desc_val = (row.get(desc_col) or "").strip() if desc_col else ""
        cat_val = (row.get(cat_col) or "").strip() if cat_col else ""

        skus.append(SKUItem(name=name_val, price=price_val, description=desc_val, category=cat_val))
    return skus


def fetch_merchant_csv(
    merchant_id: str, bearer_token: str, portal_url: str | None = None
) -> dict[str, Any]:
    """Fetches a merchant's SKU CSV from the Food Portal API with SSRF protection."""
    base_url = (portal_url or "").strip() or config.PORTAL_URL
    try:
        validate_outbound_url(base_url, allowed_domains=("pickme.lk", "google.com"))
    except ValueError as e:
        logger.warning(f"[PORTAL] Blocked SSRF attempt for merchant '{merchant_id}': {e}")
        return {"skus": [], "auth_failed": False, "error": f"Invalid portal URL: {e}"}

    url = base_url.replace("{merchantid}", merchant_id.strip())
    headers = {
        "Authorization": f"Bearer {bearer_token.strip()}",
        "user-action": "view_restaurant/view",
    }

    t0 = time.time()
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        logger.warning(f"[PORTAL] Failed to reach merchant portal for '{merchant_id}': {e}")
        return {
            "skus": [],
            "auth_failed": False,
            "error": f"Could not reach the merchant portal: {e}",
        }

    duration_ms = int((time.time() - t0) * 1000)
    text = response.text

    try:
        payload = response.json()
    except ValueError:
        payload = None

    audit_service.record_outbound_request(
        url=url,
        method="GET",
        payload={"merchant_id": merchant_id},
        response_status=response.status_code,
        response_text=(text[:5000] if payload is not None else f"<{len(text)} bytes CSV>"),
        duration_ms=duration_ms,
        path="/merchant-fetch",
    )

    if isinstance(payload, dict) and payload.get("errors"):
        errors = payload["errors"]
        is_auth_failure = any(
            (e.get("code") in AUTH_ERROR_CODES)
            or any(kw in str(e.get("message", "")).lower() for kw in AUTH_ERROR_KEYWORDS)
            for e in errors
        )
        error_msg = ", ".join(
            e.get("message") or e.get("code") or "Unknown portal error" for e in errors
        )
        return {"skus": [], "auth_failed": is_auth_failure, "error": error_msg}

    if response.status_code in (401, 403):
        return {
            "skus": [],
            "auth_failed": True,
            "error": "Portal authentication failed (invalid or expired token).",
        }

    if not response.ok:
        return {
            "skus": [],
            "auth_failed": False,
            "error": f"Portal returned status {response.status_code}: {response.reason}",
        }

    skus = parse_csv_text_to_skus(text)
    if not skus:
        return {"skus": [], "auth_failed": False, "error": "Fetched CSV has no valid SKU rows."}

    return {"skus": skus, "auth_failed": False, "error": None}


def fetch_merchant_preview(
    *, merchant_id: str, bearer_token: str, portal_url: str
) -> dict[str, Any]:
    """Validates request credentials, fetches merchant CSV rows, and formats preview response."""
    if not merchant_id or not merchant_id.strip():
        raise ServiceValidationError("Merchant ID is required.")
    if not bearer_token or not bearer_token.strip():
        raise ServiceValidationError("Bearer token is required.")

    result = fetch_merchant_csv(merchant_id, bearer_token, portal_url)
    if result["auth_failed"]:
        raise UnauthorizedPortalError(
            result["error"] or "Portal token is invalid, blacklisted, or expired."
        )
    if result["error"]:
        raise UpstreamGatewayError(result["error"])

    return {
        "merchant_id": merchant_id,
        "rows": [s.model_dump() for s in result["skus"]],
    }
