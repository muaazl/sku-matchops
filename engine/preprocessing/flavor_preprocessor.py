"""Unified flavor alias parsing, regex compilation, protein/veg/seafood triage, and food mixed-suffix rules."""

from __future__ import annotations

import functools
import re
from typing import Any

import pandas as pd


@functools.lru_cache(maxsize=32)
def _compile_sorted_terms_pattern(sorted_terms: tuple[str, ...]) -> re.Pattern | None:
    if not sorted_terms:
        return None
    return re.compile(
        r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in sorted_terms) + r")(?![a-z0-9])"
    )


def compile_flavor_pattern(flavors_dict: dict[str, str] | None) -> re.Pattern | None:
    """Compiles a longest-first word-boundary regex for all flavor keys in `flavors_dict`."""
    if not flavors_dict:
        return None
    sorted_terms = tuple(sorted(flavors_dict.keys(), key=len, reverse=True))
    return _compile_sorted_terms_pattern(sorted_terms)


def extract_canonical_flavors(
    text: str,
    flavors_dict: dict[str, str],
    pattern: re.Pattern | None = None,
) -> set[str]:
    """Returns canonical flavor names matched inside `text`."""
    if not text or not flavors_dict:
        return set()
    active_pattern = pattern if pattern is not None else compile_flavor_pattern(flavors_dict)
    if active_pattern is None:
        return set()

    text_lower = text.lower()
    result: set[str] = set()
    for match in active_pattern.finditer(text_lower):
        canonical = flavors_dict.get(match.group(1))
        if canonical:
            result.add(canonical)
    return result


def resolve_flavor_set(
    flavor_set: set[str],
    flavors_dict: dict[str, str],
    pattern: re.Pattern | None = None,
) -> set[str]:
    """Maps a set of flavor strings (including multi-word phrases) to canonical flavor names."""
    result: set[str] = set()
    if not flavor_set or not flavors_dict:
        return result
    active_pattern = pattern if pattern is not None else compile_flavor_pattern(flavors_dict)
    for raw in flavor_set:
        f_lower = str(raw).lower()
        if f_lower in flavors_dict:
            result.add(flavors_dict[f_lower])
        elif active_pattern is not None:
            match = active_pattern.search(f_lower)
            if match:
                canonical = flavors_dict.get(match.group(1))
                if canonical:
                    result.add(canonical)
    return result


def build_food_flavors_info(
    brands_df: pd.DataFrame,
) -> tuple[dict[str, str], set[str], set[str], set[str], dict[str, tuple[bool, bool, bool]]]:
    """
    Parses the brands/flavors DataFrame to extract flavor lookup mapping and category sets.

    Returns:
      (flavors_dict, meat_flavors, vegetable_flavors, seafood_flavors, flavor_categories)
    """
    if brands_df is None or brands_df.empty:
        return {}, set(), set(), set(), {}

    cols = list(brands_df.columns)
    name_col = (
        "Flavor Name" if "Flavor Name" in cols else ("Brand Name" if "Brand Name" in cols else "")
    )
    if not name_col:
        return {}, set(), set(), set(), {}

    flavors_dict: dict[str, str] = {}
    meat_flavors: set[str] = set()
    vegetable_flavors: set[str] = set()
    seafood_flavors: set[str] = set()
    flavor_categories: dict[str, tuple[bool, bool, bool]] = {}

    names = brands_df[name_col].tolist()
    is_meats = brands_df["Is_Meat"].tolist() if "Is_Meat" in cols else [False] * len(brands_df)
    is_vegs = (
        brands_df["Is_Vegetable"].tolist() if "Is_Vegetable" in cols else [False] * len(brands_df)
    )
    is_seafoods = (
        brands_df["Is_Seafood"].tolist() if "Is_Seafood" in cols else [False] * len(brands_df)
    )
    aliases_list = brands_df["Aliases"].tolist() if "Aliases" in cols else [""] * len(brands_df)

    for raw_name, raw_meat, raw_veg, raw_seafood, raw_alias in zip(
        names, is_meats, is_vegs, is_seafoods, aliases_list
    ):
        flavor_name = str(raw_name or "").strip()
        if not flavor_name or flavor_name.lower() == "nan":
            continue

        canonical = flavor_name.lower()
        is_meat = str(raw_meat or "").strip().lower() in ("true", "1", "yes", "y") or bool(raw_meat)
        is_veg = str(raw_veg or "").strip().lower() in ("true", "1", "yes", "y") or bool(raw_veg)
        is_seafood = str(raw_seafood or "").strip().lower() in ("true", "1", "yes", "y") or bool(
            raw_seafood
        )
        flags = (is_meat, is_veg, is_seafood)

        if is_meat:
            meat_flavors.add(canonical)
        if is_veg:
            vegetable_flavors.add(canonical)
        if is_seafood:
            seafood_flavors.add(canonical)

        flavors_dict[canonical] = canonical
        flavor_categories[canonical] = flags

        aliases_str = str(raw_alias or "")
        if aliases_str and aliases_str.lower() not in ("none", "nan"):
            aliases = [x.strip().lower() for x in aliases_str.split(",") if x.strip()]
            for alias in aliases:
                flavors_dict[alias] = canonical
                flavor_categories[alias] = flags

    if "shrimp" in flavors_dict and "prawn" in flavors_dict:
        flavors_dict["shrimp"] = "prawn"
        flavors_dict["prawn"] = "prawn"
        flavor_categories["shrimp"] = flavor_categories["prawn"]

    return flavors_dict, meat_flavors, vegetable_flavors, seafood_flavors, flavor_categories


def triage_input_flavors(
    flavor_resolver: Any,
    flavor_categories: dict[str, tuple[bool, bool, bool]],
    extracted_entities: dict | None,
    matched_row: Any,
) -> tuple[set[str], set[str], set[str]]:
    """
    Resolves input flavor entities (falling back to matched catalog row entities), then splits
    them into (input_meats, input_seafoods, input_vegs).
    """
    input_flavors: set[str] = set()
    if extracted_entities and isinstance(extracted_entities, dict):
        input_flavors.update(x.lower() for x in extracted_entities.get("flavor", set()) if x)
    if not input_flavors:
        cat_ents = matched_row.get("entities")
        if isinstance(cat_ents, dict):
            input_flavors.update(x.lower() for x in cat_ents.get("flavor", set()) if x)

    if callable(flavor_resolver):
        resolved_input_flavors = flavor_resolver(input_flavors)
    elif hasattr(flavor_resolver, "resolve_flavors"):
        resolved_input_flavors = flavor_resolver.resolve_flavors(input_flavors)
    elif hasattr(flavor_resolver, "_resolve_flavors"):
        resolved_input_flavors = flavor_resolver._resolve_flavors(input_flavors)
    else:
        resolved_input_flavors = input_flavors

    input_meats = {
        f
        for f in resolved_input_flavors
        if flavor_categories.get(f, (False, False, False))[0]
        and not flavor_categories.get(f, (False, False, False))[2]
    }
    input_seafoods = {
        f for f in resolved_input_flavors if flavor_categories.get(f, (False, False, False))[2]
    }
    input_vegs = {
        f for f in resolved_input_flavors if flavor_categories.get(f, (False, False, False))[1]
    }
    return input_meats, input_seafoods, input_vegs


def resolve_food_mixed_suffix(
    sku_name: str,
    flavor_set: set[str],
    meat_flavors: set[str] | None = None,
    vegetable_flavors: set[str] | None = None,
    seafood_flavors: set[str] | None = None,
) -> str:
    """Returns 'mixed' when multiple land meats or meat+seafood appear and name lacks 'mix'."""
    if not flavor_set:
        return ""
    meat_set = meat_flavors or set()
    veg_set = vegetable_flavors or set()
    seafood_set = seafood_flavors or set()
    if not (meat_set or veg_set or seafood_set):
        return ""

    has_seafood = any(f in seafood_set for f in flavor_set)
    # Exclude meta-categories ('mixed') and special protein ('egg') from land meats
    land_meats = [
        f
        for f in flavor_set
        if f in meat_set and f not in seafood_set and f not in ("egg", "mixed")
    ]
    meat_count = len(land_meats)
    has_other_protein = meat_count > 0 or has_seafood
    has_egg = ("egg" in flavor_set) and ("egg" in meat_set)
    has_veg = any(f in veg_set for f in flavor_set)

    # Egg priority: lone egg (or egg+veg) counts as 1 protein; with other meat/seafood it does not
    if has_egg and not has_other_protein:
        meat_count = 1
        has_meat = True
    else:
        has_meat = meat_count > 0

    if ((has_meat and has_seafood) or meat_count >= 2) and "mix" not in sku_name.lower():
        return "mixed"
    if not has_seafood and not has_meat and has_veg:
        return ""
    return ""
