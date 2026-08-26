import re
from typing import Optional, Tuple
from engine import config

class UnitConfig:
    """Configuration for unit normalization and physical form mapping."""
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
    """Utility functions for parsing and normalizing measurements."""

    @staticmethod
    def normalize_value(value: float, unit_str: str) -> Optional[Tuple[float, str, str]]:
        """Converts a value/unit pair to its base unit (g or ml)."""
        if not unit_str:
            return None
        unit_info = UnitConfig.UNIT_MAP.get(unit_str.lower().rstrip("s"))
        if unit_info:
            multiplier, base_unit, u_type = unit_info
            return value * multiplier, base_unit, u_type
        return None

    @staticmethod
    def get_normalized_weight(text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Extracts and normalizes the first measurement found in a text string."""
        if not isinstance(text, str):
            return None, None, None

        # Fast-path regex for standard unit patterns (avoids heavy quantulum3 NLP parse)
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

        # Fallback to robust parsing with quantulum3 for complex expressions
        try:
            from quantulum3 import parser as q_parser
            # Ensure space between digits and letters for better parsing
            clean_text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
            quants = q_parser.parse(clean_text)
            for q in quants:
                result = UnitUtils.normalize_value(q.value, q.unit.name.lower())
                if result:
                    return result
        except Exception:
            pass

        # Fallback to simple regex parsing
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
    """Pipeline for SKU text cleaning, normalization, and feature extraction."""

    @staticmethod
    def extract_weight_feature(text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Feature extraction wrapper for normalized weights."""
        return UnitUtils.get_normalized_weight(text)

    @staticmethod
    def standardize_units(text: str) -> str:
        """Standardizes unit formatting and resolves multipack math (e.g., '2 x 100ml' -> '200ml')."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Add space between numbers and units
        text = re.sub(r"(\d+)([a-z]+)", r"\1 \2", text)
        # Remove ranges (e.g., '100g - 200g') as they confuse matching
        text = re.sub(r"(\d+(\.\d+)?)\s*[a-z]+\s*-\s*(\d+(\.\d+)?)\s*[a-z]+", "", text)

        def calc_multipack(match: re.Match, reverse: bool = False) -> str:
            count_grp, size_grp, unit_grp = (1, 2, 4) if not reverse else (4, 1, 3)
            count = float(match.group(count_grp))
            size = float(match.group(size_grp))
            unit = match.group(unit_grp)
            total = count * size
            normalized = UnitUtils.normalize_value(total, unit)
            if normalized:
                val, base_unit, _ = normalized
                return f"{int(val)}{base_unit} "
            return f"{int(total)}{unit} "

        # Resolve '4 x 100g' and '100g x 4'
        text = re.sub(r"(\d+)\s*[xX]\s*(\d+(\.\d+)?)\s*([a-z]+)", calc_multipack, text)
        text = re.sub(r"(\d+(\.\d+)?)\s*([a-z]+)\s*[xX]\s*(\d+)", lambda m: calc_multipack(m, reverse=True), text)

        def convert_match(match: re.Match) -> str:
            val = float(match.group(1))
            unit = match.group(3)
            normalized = UnitUtils.normalize_value(val, unit)
            if normalized:
                norm_val, base_unit, _ = normalized
                return f"{int(norm_val)}{base_unit}"
            return match.group(0)

        # Normalize all remaining units to base units
        text = re.sub(r"\b(\d+(\.\d+)?)\s?([a-z]+)\b", convert_match, text)
        # Standardize piece counts
        text = re.sub(r"\bpack of (\d+)\b", r"\1pc", text)
        text = re.sub(r"(\d+)\s*(pcs|pieces|units)\b", r"\1pc", text)
        return text.strip()

    @staticmethod
    def normalize_final(text: str) -> str:
        """Final text normalization: lowercase, remove special chars, and noise."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Remove common SKU noise
        text = re.sub(r"(-|\|)?\s*sku\s*\d+", "", text)
        text = re.sub(r"\b\d{4,6}\b", "", text)
        # Keep only alphanumeric and dots
        text = re.sub(r"[^a-z0-9\s\.]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def get_ice_cream_type(weight_str: str) -> Optional[str]:
        """Maps a weight/volume value to an ice cream packaging type based on config buckets."""
        if not weight_str or not isinstance(weight_str, str):
            return None
        match = re.search(r"(\d+(\.\d+)?)", weight_str)
        if not match:
            return None
        val = float(match.group(1))
        for low, high, label in config.IC_BUCKETS:
            if low <= val <= high:
                return label
        return None

    @staticmethod
    def strip_weights(text: str) -> str:
        """Removes extracted units/weights to improve fuzzy matching and prevent NER hallucinations."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\b\d+(\.\d+)?\s?[a-z]+\b", "", text.lower())
        text = re.sub(r"\b\d+\b", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def prep_for_ner(raw_text: str) -> str:
        """Formats text for GLiNER by preserving casing while removing numeric noise."""
        if not isinstance(raw_text, str):
            return ""
        # Remove weights/units while preserving case of other words
        text = re.sub(r"\b\d+(\.\d+)?\s?[a-zA-Z]+\b", "", raw_text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d+\b", "", text)
        cleaned = re.sub(r"\s+", " ", text).strip()
        return f"Product: {cleaned}"