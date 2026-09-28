"""
SKU MatchOps Backend - Food Portal Client
Fetches merchant SKU CSVs from the PickMe Food Portal API and parses them into SKU rows.
"""
import csv
import io
import logging
import os
import time
from typing import List, Optional

import requests

from backend.app.schemas.models import SKUItem
from engine.core.db import log_outbound_request

logger = logging.getLogger("matchops.portal_service")

DEFAULT_PORTAL_URL = "https://uni-portal-api.pickme.lk/food/v1/t0001/food/place/skus/csv/{merchantid}"
PORTAL_URL = os.getenv("PORTAL_URL", DEFAULT_PORTAL_URL)

# Portal error codes/messages indicating an invalid, blacklisted, or expired token.
AUTH_ERROR_CODES = {"MER-4007", "MER-4006"}
AUTH_ERROR_KEYWORDS = ("token blacklisted", "token expired", "invalid token", "unauthorized", "unauthenticated", "signature verification failed")


def parse_csv_text_to_skus(text: str) -> List[SKUItem]:
    """Parses raw CSV text into SKUItem rows, auto-detecting name/price/description/category columns.

    Shared by the direct CSV/TSV upload path and the merchant-portal fetch path so both
    produce identically-shaped SKU rows from the same column-detection rules.
    """
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    name_col = next((f for f in fieldnames if f.lower().strip() in ('name', 'sku_name', 'sku', 'title')), None)
    price_col = next((f for f in fieldnames if f.lower().strip() in ('price', 'cost', 'mrp')), None)
    desc_col = next((f for f in fieldnames if f.lower().strip() in ('description', 'desc')), None)
    cat_col = next((f for f in fieldnames if f.lower().strip() in ('category', 'cat', 'type')), None)

    if not name_col and fieldnames:
        name_col = fieldnames[0]

    skus: List[SKUItem] = []
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


def fetch_merchant_csv(merchant_id: str, bearer_token: str, portal_url: Optional[str] = None) -> dict:
    """
    Fetches a merchant's SKU CSV from the Food Portal API (server-side) and parses it.

    Returns {"skus": List[SKUItem], "auth_failed": bool, "error": Optional[str]}.
    `auth_failed` signals the caller should clear the stored bearer token and re-prompt
    the user for a fresh one (covers the portal's own error codes for a blacklisted/expired
    token, as well as a generic 401/403 from the portal).
    """
    base_url = (portal_url or "").strip() or PORTAL_URL
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
        return {"skus": [], "auth_failed": False, "error": f"Could not reach the merchant portal: {e}"}

    duration_ms = int((time.time() - t0) * 1000)
    text = response.text

    try:
        payload = response.json()
    except ValueError:
        payload = None

    log_outbound_request(
        url=url,
        method="GET",
        payload={"merchant_id": merchant_id},
        response_status=response.status_code,
        response_text=(text[:5000] if payload is not None else f"<{len(text)} bytes CSV>"),
        duration_ms=duration_ms,
        path="/merchant-fetch",
    )

    # The portal returns a JSON error body (including auth failures) on failure, raw CSV on success.
    if isinstance(payload, dict) and payload.get("errors"):
        errors = payload["errors"]
        is_auth_failure = any(
            (e.get("code") in AUTH_ERROR_CODES)
            or any(kw in str(e.get("message", "")).lower() for kw in AUTH_ERROR_KEYWORDS)
            for e in errors
        )
        error_msg = ", ".join(e.get("message") or e.get("code") or "Unknown portal error" for e in errors)
        return {"skus": [], "auth_failed": is_auth_failure, "error": error_msg}

    if response.status_code in (401, 403):
        return {"skus": [], "auth_failed": True, "error": "Portal authentication failed (invalid or expired token)."}

    if not response.ok:
        return {"skus": [], "auth_failed": False, "error": f"Portal returned status {response.status_code}: {response.reason}"}

    skus = parse_csv_text_to_skus(text)
    if not skus:
        return {"skus": [], "auth_failed": False, "error": "Fetched CSV has no valid SKU rows."}

    return {"skus": skus, "auth_failed": False, "error": None}
