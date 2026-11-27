"""Pure data cleaning, catalog hashing, unit normalization, and confidence helpers."""

import hashlib
import json
import logging
import math
import re
from typing import Any

import pandas as pd

from backend.core.constants import CACHE_SALT, DOMAIN_FOOD, DOMAIN_MARKET

logger = logging.getLogger("matchops.helpers")

UNIT_MAP: dict[str, tuple[float, str, str]] = {
    "g": (1, "g", "solid"),
    "gram": (1, "g", "solid"),
    "mg": (0.001, "g", "solid"),
    "milligram": (0.001, "g", "solid"),
    "kg": (1000, "g", "solid"),
    "kilogram": (1000, "g", "solid"),
    "tonne": (1000000, "g", "solid"),
    "ml": (1, "ml", "liquid"),
    "millilitre": (1, "ml", "liquid"),
    "l": (1000, "ml", "liquid"),
    "liter": (1000, "ml", "liquid"),
    "cl": (10, "ml", "liquid"),
}

FAST_WEIGHT_REGEX = re.compile(
    r"(?<![a-z0-9])(\d{1,12}(?:\.\d{1,6})?)\s{0,6}(mg|milligrams?|kg|kilograms?|g|grams?|ml|millilitres?|milliliters?|l|liters?|litres?|cl|tonnes?)\b",
    re.IGNORECASE,
)


def clean_price(val: Any) -> float | None:
    """Safely cleans and converts a price value to a float."""
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


def parse_optional_float(val: Any) -> float | None:
    """Converts a value to float when non-empty, otherwise returning None."""
    try:
        return float(val) if val not in (None, "") else None
    except (TypeError, ValueError):
        return None


def replace_nan_with_none(row: dict[str, Any]) -> dict[str, Any]:
    """Converts pandas NaN floats to None so SQL binds store NULL cleanly."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def coerce_int_flag(val: Any) -> int:
    """Coerces a nullable flag value into an integer 0/1 value."""
    return int(val) if val is not None and not pd.isna(val) else 0


def format_catalog_field(col: str, val: Any) -> str:
    """Normalizes a single catalog cell value for deterministic hashing and diffing."""
    if pd.isna(val) or val is None:
        return ""
    if col == "Price":
        parsed = clean_price(val)
        return f"{parsed:.2f}" if parsed is not None else ""
    if isinstance(val, (int, float)):
        return f"{float(val):.2f}"
    return str(val).strip()


def calculate_row_hash(row: Any, domain: str = DOMAIN_MARKET) -> str:
    """Generates a stable SHA-256 hash for a single catalog row."""
    row_dict = row if isinstance(row, dict) else row.to_dict()

    cols = ["Name", "basictype", "Generic keywords", "Price", "Description"]
    if domain == DOMAIN_FOOD:
        cols.extend(["region", "Flavor"])
    else:
        cols.extend(["category", "Brand"])

    cleaned = {c: format_catalog_field(c, row_dict.get(c, "")) for c in cols}
    row_str = json.dumps(cleaned, sort_keys=True) + CACHE_SALT
    return hashlib.sha256(row_str.encode("utf-8")).hexdigest()


def build_occurrence_uids(df: pd.DataFrame, name_col: str = "Name") -> tuple[list[str], list[str]]:
    """Builds vectorized per-name occurrence UIDs ('{raw_name}#occ_{n}') aligned to df rows."""
    if len(df) == 0:
        return [], []
    if name_col in df.columns:
        # Use .map(str) rather than .astype(str) so NaN becomes "nan" consistently without groupby drops.
        raw_names = df[name_col].astype(object).map(str).str.strip().str.lower()
    else:
        raw_names = pd.Series([""] * len(df), index=df.index)
    occ = raw_names.groupby(raw_names).cumcount() + 1
    uids = (raw_names + "#occ_" + occ.astype(str)).tolist()
    return raw_names.tolist(), uids


def normalize_unit_value(value: float, unit_str: str) -> tuple[float, str, str] | None:
    """Converts a value/unit pair to its canonical base unit ('g' or 'ml')."""
    if not unit_str:
        return None
    unit_info = UNIT_MAP.get(unit_str.lower().rstrip("s"))
    if unit_info:
        multiplier, base_unit, u_type = unit_info
        return value * multiplier, base_unit, u_type
    return None


def extract_normalized_weight(text: str) -> tuple[float | None, str | None, str | None]:
    """Extracts and normalizes the first measurement found in a text string."""
    if not isinstance(text, str) or not any(c.isdigit() for c in text):
        return None, None, None

    try:
        match = FAST_WEIGHT_REGEX.search(text)
        if match:
            result = normalize_unit_value(float(match.group(1)), match.group(2))
            if result:
                return result
    except Exception:
        pass

    try:
        match = re.search(r"(?<!\d)(\d{1,12}(\.\d{1,6})?)\s{0,6}([a-zA-Z]{1,24})\b", text.lower())
        if match:
            result = normalize_unit_value(float(match.group(1)), match.group(3))
            if result:
                return result
    except Exception:
        pass

    return None, None, None


def standardize_sku_units(text: str) -> str:
    """Normalizes multipacks, weights, volumes, and pack counts in SKU text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(?<!\d)(\d{1,12})([a-z]{1,24})", r"\1 \2", text)
    text = re.sub(
        r"(?<!\d)(\d{1,12}(?:\.\d{1,6})?)\s{0,6}[a-z]{1,24}\s{0,6}-\s{0,6}(\d{1,12}(?:\.\d{1,6})?)\s{0,6}[a-z]{1,24}",
        "",
        text,
    )

    def _fmt_num(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{round(x, 4):.4f}".rstrip("0").rstrip(".")

    def _calc_multipack(match: re.Match, reverse: bool = False) -> str:
        count_grp, size_grp, unit_grp = (1, 2, 4) if not reverse else (4, 1, 3)
        count = float(match.group(count_grp))
        size = float(match.group(size_grp))
        unit = match.group(unit_grp)
        total = count * size
        normalized = normalize_unit_value(total, unit)
        if normalized:
            val, base_unit, _ = normalized
            return f"{_fmt_num(val)}{base_unit} "
        return f"{_fmt_num(total)}{unit} "

    text = re.sub(
        r"(?<!\d)(\d{1,12})\s{0,6}[xX]\s{0,6}(\d{1,12}(\.\d{1,6})?)\s{0,6}([a-z]{1,24})",
        _calc_multipack,
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d{1,12}(\.\d{1,6})?)\s{0,6}([a-z]{1,24})\s{0,6}[xX]\s{0,6}(\d{1,12})",
        lambda m: _calc_multipack(m, reverse=True),
        text,
    )

    def _convert_match(match: re.Match) -> str:
        val = float(match.group(1))
        unit = match.group(3)
        normalized = normalize_unit_value(val, unit)
        if normalized:
            norm_val, base_unit, _ = normalized
            return f"{_fmt_num(norm_val)}{base_unit}"
        return match.group(0)

    text = re.sub(r"\b(\d+(\.\d+)?)\s?([a-z]+)\b", _convert_match, text)
    text = re.sub(r"\bpack of (\d+)\b", r"\1pc", text)
    text = re.sub(r"(?<!\d)(\d{1,12})\s{0,6}(?:pieces|pcs|units)\b", r"\1pc", text)
    return text.strip()


def normalize_sku_text(text: str) -> str:
    """Strips SKU codes and punctuation while preserving English, Sinhala, and Tamil characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(?:[-|]\s{0,6}|(?<!\s)\s{0,6})sku\s{0,6}\d{1,12}", "", text)
    text = re.sub(r"\b\d{4,6}\b", "", text)
    text = re.sub(r"[^\w\s\.\u0D80-\u0DFF\u0B80-\u0BFF]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_weight_tokens(text: str) -> str:
    """Removes measurement and bare numeric tokens from normalized SKU text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\b\d+(\.\d+)?\s?[a-z]+\b", "", text.lower())
    text = re.sub(r"\b\d+\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_weighted_confidence(
    bt_conf: float | None = None,
    gk_conf: float | None = None,
    region_conf: float | None = None,
    has_bt: bool = True,
    has_gk: bool = True,
    has_region: bool = True,
    w_bt: float = 0.5,
    w_gk: float = 0.3,
    w_reg: float = 0.2,
) -> float:
    """Computes dynamically normalized weighted confidence across available tags."""
    weights = []
    confs = []
    if has_bt and bt_conf is not None:
        try:
            confs.append(float(bt_conf))
            weights.append(w_bt)
        except (ValueError, TypeError):
            pass
    if has_gk and gk_conf is not None:
        try:
            confs.append(float(gk_conf))
            weights.append(w_gk)
        except (ValueError, TypeError):
            pass
    if has_region and region_conf is not None:
        try:
            confs.append(float(region_conf))
            weights.append(w_reg)
        except (ValueError, TypeError):
            pass

    if not weights or sum(weights) == 0:
        return 0.0
    return float(sum(c * w for c, w in zip(confs, weights)) / sum(weights))
