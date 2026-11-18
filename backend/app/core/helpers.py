"""
SKU MatchOps Backend - Helpers
Core data cleaning, hashing, and calculation utilities for the API Gateway.
"""

import hashlib
import json
import logging
import re
from typing import Any

import pandas as pd

from backend.app.core.constants import CACHE_SALT, DOMAIN_FOOD, DOMAIN_MARKET

logger = logging.getLogger("matchops.helpers")


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


def calculate_row_hash(row: Any, domain: str = DOMAIN_MARKET) -> str:
    """Generates a stable MD5 hash for a single catalog row."""
    row_dict = row if isinstance(row, dict) else row.to_dict()

    cols = ["Name", "basictype", "Generic keywords", "Price", "Description"]
    if domain == DOMAIN_FOOD:
        cols.extend(["region", "Flavor"])
    else:
        cols.extend(["category", "Brand"])

    cleaned = {}
    for c in cols:
        val = row_dict.get(c, "")
        if pd.isna(val) or val is None:
            val = ""
        elif c == "Price":
            p = clean_price(val)
            val = f"{p:.2f}" if p is not None else ""
        elif isinstance(val, (int, float)):
            val = f"{float(val):.2f}"
        else:
            val = str(val).strip()
        cleaned[c] = val

    row_str = json.dumps(cleaned, sort_keys=True) + CACHE_SALT
    return hashlib.md5(row_str.encode("utf-8")).hexdigest()


class UnitConfig:
    UNIT_MAP = {
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


_FAST_WEIGHT_REGEX = re.compile(
    r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(mg|milligram|milligrams|kg|kilogram|kilograms|g|gram|grams|ml|millilitre|millilitres|milliliter|milliliters|l|liter|liters|litre|litres|cl|tonne|tonnes)\b",
    re.IGNORECASE,
)


class UnitUtils:
    @staticmethod
    def normalize_value(value: float, unit_str: str) -> tuple[float, str, str] | None:
        if not unit_str:
            return None
        unit_info = UnitConfig.UNIT_MAP.get(unit_str.lower().rstrip("s"))
        if unit_info:
            multiplier, base_unit, u_type = unit_info
            return value * multiplier, base_unit, u_type
        return None

    @staticmethod
    def get_normalized_weight(text: str) -> tuple[float | None, str | None, str | None]:
        if not isinstance(text, str) or not any(c.isdigit() for c in text):
            return None, None, None

        try:
            match = _FAST_WEIGHT_REGEX.search(text)
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                result = UnitUtils.normalize_value(val, unit)
                if result:
                    return result
        except Exception:
            pass

        try:
            match = re.search(r"(\d+(\.\d+)?)\s*([a-zA-Z]+)\b", text.lower())
            if match:
                val = float(match.group(1))
                unit = match.group(3)
                result = UnitUtils.normalize_value(val, unit)
                if result:
                    return result
        except Exception:
            pass

        return None, None, None


class TextPipeline:
    """SKU text cleaning and normalization for catalog ingestion."""

    @staticmethod
    def extract_weight_feature(text: str) -> tuple[float | None, str | None, str | None]:
        return UnitUtils.get_normalized_weight(text)

    @staticmethod
    def standardize_units(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"(\d+)([a-z]+)", r"\1 \2", text)
        text = re.sub(r"(\d+(\.\d+)?)\s*[a-z]+\s*-\s*(\d+(\.\d+)?)\s*[a-z]+", "", text)

        def _fmt_num(x: float) -> str:
            if abs(x - round(x)) < 1e-9:
                return str(int(round(x)))
            return f"{round(x, 4):.4f}".rstrip("0").rstrip(".")

        def calc_multipack(match: re.Match, reverse: bool = False) -> str:
            count_grp, size_grp, unit_grp = (1, 2, 4) if not reverse else (4, 1, 3)
            count = float(match.group(count_grp))
            size = float(match.group(size_grp))
            unit = match.group(unit_grp)
            total = count * size
            normalized = UnitUtils.normalize_value(total, unit)
            if normalized:
                val, base_unit, _ = normalized
                return f"{_fmt_num(val)}{base_unit} "
            return f"{_fmt_num(total)}{unit} "

        text = re.sub(r"(\d+)\s*[xX]\s*(\d+(\.\d+)?)\s*([a-z]+)", calc_multipack, text)
        text = re.sub(
            r"(\d+(\.\d+)?)\s*([a-z]+)\s*[xX]\s*(\d+)",
            lambda m: calc_multipack(m, reverse=True),
            text,
        )

        def convert_match(match: re.Match) -> str:
            val = float(match.group(1))
            unit = match.group(3)
            normalized = UnitUtils.normalize_value(val, unit)
            if normalized:
                norm_val, base_unit, _ = normalized
                return f"{_fmt_num(norm_val)}{base_unit}"
            return match.group(0)

        text = re.sub(r"\b(\d+(\.\d+)?)\s?([a-z]+)\b", convert_match, text)
        text = re.sub(r"\bpack of (\d+)\b", r"\1pc", text)
        text = re.sub(r"(\d+)\s*(pcs|pieces|units)\b", r"\1pc", text)
        return text.strip()

    @staticmethod
    def normalize_final(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"(-|\|)?\s*sku\s*\d+", "", text)
        text = re.sub(r"\b\d{4,6}\b", "", text)
        text = re.sub(r"[^\w\s\.\u0D80-\u0DFF\u0B80-\u0BFF]|_", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def strip_weights(text: str) -> str:
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
