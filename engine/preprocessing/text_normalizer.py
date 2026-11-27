"""Pure SKU text normalization, unit conversion, weight parsing, and reranker prompt formatting."""

from __future__ import annotations

import re
from typing import Any

from engine.core import config

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

MINOR_TITLE_WORDS: frozenset[str] = frozenset(
    {
        "with",
        "of",
        "and",
        "in",
        "the",
        "for",
        "to",
        "at",
        "by",
        "on",
        "a",
        "an",
        "or",
        "as",
        "but",
        "nor",
        "yet",
        "so",
    }
)


def normalize_unit_value(value: float, unit_str: str) -> tuple[float, str, str] | None:
    """Converts a value/unit pair to its canonical base unit (g or ml)."""
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

    # Fast-path regex for standard unit patterns (avoids heavy quantulum3 NLP parse)
    try:
        match = FAST_WEIGHT_REGEX.search(text)
        if match:
            result = normalize_unit_value(float(match.group(1)), match.group(2))
            if result:
                return result
    except Exception:
        pass

    # Fallback to quantulum3 for complex textual expressions
    try:
        from quantulum3 import parser as q_parser

        clean_text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
        for q in q_parser.parse(clean_text):
            result = normalize_unit_value(q.value, q.unit.name.lower())
            if result:
                return result
    except Exception:
        pass

    # Fallback to basic numeric + unit regex
    try:
        match = re.search(r"(?<!\d)(\d{1,12}(\.\d{1,6})?)\s{0,6}([a-zA-Z]{1,24})\b", text.lower())
        if match:
            result = normalize_unit_value(float(match.group(1)), match.group(3))
            if result:
                return result
    except Exception:
        pass

    return None, None, None


def extract_weight_feature(text: str) -> tuple[float | None, str | None, str | None]:
    """Feature extraction wrapper returning normalized (value, base_unit, form_type)."""
    return extract_normalized_weight(text)


def standardize_sku_units(text: str) -> str:
    """Standardizes unit formatting and resolves multipack math (e.g., '2 x 100ml' -> '200ml')."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(?<!\d)(\d{1,12})([a-z]{1,24})", r"\1 \2", text)
    # Remove measurement ranges (e.g., '100g - 200g') that confuse matching
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
        total = float(match.group(count_grp)) * float(match.group(size_grp))
        unit = match.group(unit_grp)
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
        normalized = normalize_unit_value(float(match.group(1)), match.group(3))
        if normalized:
            norm_val, base_unit, _ = normalized
            return f"{_fmt_num(norm_val)}{base_unit}"
        return match.group(0)

    text = re.sub(r"\b(\d+(\.\d+)?)\s?([a-z]+)\b", _convert_match, text)
    text = re.sub(r"\bpack of (\d+)\b", r"\1pc", text)
    text = re.sub(r"(?<!\d)(\d{1,12})\s{0,6}(?:pieces|pcs|units)\b", r"\1pc", text)
    return text.strip()


def normalize_sku_text(text: str) -> str:
    """Performs final text cleanup: lowercase, strip SKU noise, and preserve Indic/Latin word chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(?:[-|]\s{0,6}|(?<!\s)\s{0,6})sku\s{0,6}\d{1,12}", "", text)
    text = re.sub(r"\b\d{4,6}\b", "", text)
    text = re.sub(r"[^\w\s\.\u0D80-\u0DFF\u0B80-\u0BFF]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_ice_cream_packaging(weight_str: str) -> str | None:
    """Maps a weight/volume value to an ice cream packaging bucket configured in settings."""
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


def strip_weight_tokens(text: str) -> str:
    """Removes weight tokens and standalone numbers to improve fuzzy matching and NER recall."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\b\d+(\.\d+)?\s?[a-z]+\b", "", text.lower())
    text = re.sub(r"\b\d+\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_ner_text(raw_text: str) -> str:
    """Formats SKU text for GLiNER extraction while preserving casing and stripping weights."""
    if not isinstance(raw_text, str):
        return ""
    text = re.sub(r"\b\d+(\.\d+)?\s?[a-zA-Z]+\b", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b", "", text)
    cleaned = re.sub(r"\s+", " ", text).strip()
    return f"Product: {cleaned}"


def build_ner_input_text(name: str, description: str = "", category: str = "") -> str:
    """Concatenates name, description, and category, skipping null-placeholder values."""
    full_text = name or ""
    if description and str(description).lower() not in ("nan", "none", "<na>"):
        full_text += f" {description}"
    if category and str(category).lower() not in ("nan", "none", "<na>"):
        full_text += f" {category}"
    return full_text


def clean_prompt_value(val: Any) -> str:
    """Sanitizes optional scalar values for reranker prompt serialization."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("", "nan", "none", "<na>", "null") else s


def format_matcher_rerank_text(
    name: str,
    seller_category: str = "",
    brand_or_flavor: str = "",
    description: str = "",
    domain: str = config.DOMAIN_MARKET,
) -> str:
    """Formats SKU or catalog metadata for Stage-2 reranking, omitting empty segments."""
    parts = [f"Name: {clean_prompt_value(name)}"]
    cat_clean = clean_prompt_value(seller_category)
    if cat_clean:
        parts.append(f"SellerCategory: {cat_clean}")

    attr_clean = clean_prompt_value(brand_or_flavor)
    if attr_clean:
        attr_label = "Flavor" if domain == config.DOMAIN_FOOD else "Brand"
        parts.append(f"{attr_label}: {attr_clean}")

    desc_clean = clean_prompt_value(description)
    if desc_clean:
        parts.append(f"Description: {desc_clean}")

    return " | ".join(parts)


def format_catalog_row_for_reranker(row: dict, domain: str = config.DOMAIN_MARKET) -> str:
    """Extracts catalog row fields and formats them for Stage-2 SKU-to-Catalog reranking."""
    cand_name = (
        clean_prompt_value(row.get("clean_no_weights"))
        or clean_prompt_value(strip_weight_tokens(clean_prompt_value(row.get("clean_text"))))
        or clean_prompt_value(row.get("Name"))
    )
    cand_cat = (
        clean_prompt_value(row.get("SellerCategory"))
        or clean_prompt_value(row.get("Seller_Category"))
        or clean_prompt_value(row.get("seller_category"))
    )
    cand_ents = row.get("entities") if isinstance(row.get("entities"), dict) else {}
    attr_key = "flavor" if domain == config.DOMAIN_FOOD else "brand"
    raw_attr = clean_prompt_value(row.get(attr_key.capitalize())) or clean_prompt_value(
        row.get(attr_key)
    )
    cand_bf = raw_attr or ", ".join(sorted(str(x) for x in cand_ents.get(attr_key, []) if x))
    cand_desc = clean_prompt_value(row.get("Description")) or clean_prompt_value(
        row.get("description")
    )
    return format_matcher_rerank_text(
        name=cand_name,
        seller_category=cand_cat,
        brand_or_flavor=cand_bf,
        description=cand_desc,
        domain=domain,
    )


def _split_word_affixes(word: str) -> tuple[str, str, str]:
    """Splits leading and trailing non-word punctuation from a token in linear time."""
    start = 0
    end = len(word)
    while start < end and not (word[start].isalnum() or word[start] == "_"):
        start += 1
    while end > start and not (word[end - 1].isalnum() or word[end - 1] == "_"):
        end -= 1
    return word[:start], word[start:end].lower(), word[end:]


def format_title_case(text: str) -> str:
    """Formats tags to title case while keeping minor conjunctions/prepositions lowercase."""
    if not text:
        return ""

    capitalized_words = []
    for idx, word in enumerate(text.split()):
        prefix, clean_word, suffix = _split_word_affixes(word)
        if clean_word in MINOR_TITLE_WORDS:
            inner = clean_word.capitalize() if idx == 0 else clean_word
            capitalized_words.append(prefix + inner + suffix)
        else:
            capitalized_words.append("-".join(p.capitalize() for p in word.split("-")))

    return " ".join(capitalized_words)
