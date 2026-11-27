"""Tabular row normalization, price parsing, stable row/DataFrame hashing, and weight/BT consensus helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

import pandas as pd

from engine.core import config
from engine.preprocessing.text_normalizer import (
    extract_weight_feature,
    normalize_sku_text,
    standardize_sku_units,
    strip_weight_tokens,
)

logger = logging.getLogger("matchops.preprocessing.weight")


def get_catalog_row(raw_catalog: pd.DataFrame | list[dict], idx: int):
    """Fetches a catalog row by integer position regardless of list vs DataFrame storage."""
    return raw_catalog[idx] if isinstance(raw_catalog, list) else raw_catalog.iloc[idx]


def clean_price_value(val: Any) -> float | None:
    """Safely cleans and converts a scalar price value to a float."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_str = val.strip().replace("$", "").replace(",", "")
        if not val_str:
            return None
        try:
            return float(val_str)
        except ValueError:
            logger.warning(f"Could not convert price string '{val}' to float.")
            return None
    return None


def is_missing_scalar(val: Any) -> bool:
    """Type-safe missing check handling dicts, lists, None, pd.NA, and NaN."""
    return val is None or val is pd.NA or (isinstance(val, float) and math.isnan(val))


def calculate_row_hash(row: Any, domain: str = config.DOMAIN_MARKET) -> str:
    """Generates a stable SHA-256 hash for a single catalog row."""
    row_dict = row if isinstance(row, dict) else row.to_dict()
    cols = ["Name", "basictype", "Generic keywords", "Price", "Description", "SellerCategory"]
    if domain == config.DOMAIN_FOOD:
        cols.extend(["region", "Flavor"])
    else:
        cols.extend(["category", "Brand"])

    cleaned: dict[str, str] = {}
    for col in cols:
        val = row_dict.get(col, "")
        if pd.isna(val) or val is None:
            val = ""
        elif col == "Price":
            parsed = clean_price_value(val)
            val = f"{parsed:.2f}" if parsed is not None else ""
        elif isinstance(val, (int, float)):
            val = f"{float(val):.2f}"
        else:
            val = str(val).strip()
        cleaned[col] = val

    row_str = json.dumps(cleaned, sort_keys=True) + config.CACHE_SALT
    return hashlib.sha256(row_str.encode("utf-8")).hexdigest()


def calculate_dataframe_hash(df: pd.DataFrame, domain: str = config.DOMAIN_MARKET) -> str | None:
    """Stable, cross-session deterministic hash of catalog DataFrame content."""
    if df is None or df.empty:
        return None

    cols = ["Name", "basictype", "Generic keywords", "Price", "Description", "SellerCategory"]
    if domain == config.DOMAIN_FOOD:
        cols.extend(["region", "Flavor"])
    else:
        cols.extend(["category", "Brand"])

    existing_cols = [c for c in cols if c in df.columns]
    temp_df = df[existing_cols].copy()
    temp_df = temp_df[temp_df["Name"].astype(str).str.strip() != ""]
    temp_df = temp_df[temp_df["Name"].notna()]

    for col in temp_df.columns:
        if col == "Price":
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce").fillna(0.0).round(2)
        else:
            temp_df[col] = temp_df[col].astype(str).str.strip().str.lower()

    temp_df = temp_df.sort_values(by="Name").reset_index(drop=True)
    return hashlib.sha256(temp_df.to_json(orient="records").encode("utf-8")).hexdigest()


def populate_basic_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes clean_text, weight_val, token_count, and clean_no_weights on a DataFrame."""
    df["clean_text"] = (
        df["Name"]
        .fillna("")
        .astype(str)
        .apply(lambda x: normalize_sku_text(standardize_sku_units(x)))
    )
    df["weight_val"] = df["clean_text"].apply(extract_weight_feature)
    df["token_count"] = df["clean_text"].fillna("").astype(str).apply(lambda s: len(s.split()))
    df["clean_no_weights"] = df["clean_text"].apply(strip_weight_tokens)
    return df


def resolve_bt_consensus_indices(
    raw_catalog: pd.DataFrame | list[dict],
    cand_indices: list[int],
    conf_bt: str | None = None,
    cat_bts_lower: list[str] | None = None,
) -> list[int]:
    """
    Filters catalog candidate indices tied on exact or token-sorted text to a consistent BasicType.
    """
    if not cand_indices or len(cand_indices) <= 1:
        return list(cand_indices)

    bts: list[str] = []
    bt_counts: dict[str, int] = {}
    for idx in cand_indices:
        if cat_bts_lower is not None and idx < len(cat_bts_lower):
            bt = cat_bts_lower[idx]
        else:
            row = get_catalog_row(raw_catalog, idx)
            bt = str(row.get("basictype", row.get("BasicType", "")) or "").strip().lower()
        bts.append(bt)
        if bt:
            bt_counts[bt] = bt_counts.get(bt, 0) + 1

    if not bt_counts:
        return list(cand_indices)

    majority_bt = max(bt_counts.items(), key=lambda item: item[1])[0]
    majority_count = bt_counts[majority_bt]

    if conf_bt:
        conf_bt_norm = conf_bt.strip().lower()
        bt_matched = [idx for idx, bt in zip(cand_indices, bts) if bt == conf_bt_norm]
        if bt_matched and majority_count < 3 * len(bt_matched):
            return bt_matched

    majority_indices = [idx for idx, bt in zip(cand_indices, bts) if bt == majority_bt]
    return majority_indices or list(cand_indices)


def resolve_weight_bypass_candidate(
    raw_catalog: pd.DataFrame | list[dict],
    cand_indices: list[int],
    input_w_data: tuple[float | None, str | None, str | None],
) -> tuple[int, str]:
    """
    Given catalog row indices tied on token-sorted text, selects the one whose weight
    is numerically closest to `input_w_data` within the same physical-form type.
    """
    selected_idx = cand_indices[0]
    weight_reason = ""

    if input_w_data[0] is None:
        return selected_idx, weight_reason

    in_val, _, in_type = input_w_data
    best_match_idx = None
    min_diff_pct = float("inf")

    for c_idx in cand_indices:
        cat_row = get_catalog_row(raw_catalog, c_idx)
        catalog_w_data = cat_row.get("weight_val")
        if (
            catalog_w_data is not None
            and isinstance(catalog_w_data, (tuple, list))
            and catalog_w_data[0] is not None
        ):
            cat_val, _, cat_type = catalog_w_data
            if in_type == cat_type:
                max_val = max(in_val, cat_val)
                diff_pct = abs(in_val - cat_val) / max_val * 100 if max_val > 0 else 0
                if diff_pct < min_diff_pct:
                    min_diff_pct = diff_pct
                    best_match_idx = c_idx

    if best_match_idx is not None:
        selected_idx = best_match_idx
        cat_row = get_catalog_row(raw_catalog, selected_idx)
        catalog_w_data = cat_row.get("weight_val")
        cat_val = catalog_w_data[0]
        if min_diff_pct < 1.0:
            weight_reason = f" | Weight Match ({int(in_val)})"
        else:
            weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(cat_val)})"
    else:
        cat_row = get_catalog_row(raw_catalog, selected_idx)
        catalog_w_data = cat_row.get("weight_val")
        if (
            catalog_w_data is not None
            and isinstance(catalog_w_data, (tuple, list))
            and catalog_w_data[0] is not None
        ):
            weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(catalog_w_data[0])})"

    return selected_idx, weight_reason
