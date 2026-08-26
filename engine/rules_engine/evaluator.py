import functools
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from engine import config
from engine.utils.flavor_utils import build_food_flavors_info

logger = logging.getLogger("matchops.rules_evaluator")

def evaluate_conditions(rule, record: dict) -> bool:
    """Evaluates all conditions for a rule against a given record."""
    if not rule.conditions:
        return True # If no conditions, it always fires
        
    # Group conditions by condition_group
    groups = {}
    for cond in rule.conditions:
        cg = cond.get("condition_group", 1)
        if cg not in groups:
            groups[cg] = []
        groups[cg].append(cond)
        
    group_results = []
    
    for group_id, group_conditions in groups.items():
        # Within a group, we have AND logic, but multiple values for the SAME type in the SAME group = OR logic.
        # To handle this cleanly, we can group by condition_type within the group.
        type_groups = {}
        for cond in group_conditions:
            ctype = cond["condition_type"]
            if ctype not in type_groups:
                type_groups[ctype] = []
            type_groups[ctype].append(cond)
            
        group_passed = True
        
        for ctype, conditions_of_type in type_groups.items():
            # OR logic within the same condition_type in the same group
            type_passed = False
            for cond in conditions_of_type:
                try:
                    res = _evaluate_single_condition(cond, record)
                    if res:
                        type_passed = True
                        break
                except Exception as e:
                    logger.warning(f"Error evaluating condition {cond} for rule {rule.rule_id}: {e}")
                    # If it fails to evaluate, treat as false to prevent unwanted firing
                    pass
            
            # AND logic across different condition_types in the same group
            if not type_passed:
                group_passed = False
                break
                
        group_results.append(group_passed)
        
    # Across groups: OR logic or AND logic based on rule.condition_logic (default is OR in standard engines, but SRS says check condition_logic if we want)
    # The SRS specifically says "Conditions across different groups are evaluated with OR logic" in 4.2
    # So we ignore rule.condition_logic and use ANY for across groups, but wait, the SRS says "top-level logic between condition groups: AND | OR" in 4.1 rule table.
    # We will use rule.condition_logic.
    
    logic = rule.condition_logic.upper()
    if logic == 'AND':
        return all(group_results)
    else: # OR
        return any(group_results)

def _get_combined_flavor_pattern(flavors_dict: dict) -> Optional[re.Pattern]:
    if not flavors_dict:
        return None
    sorted_terms = sorted(flavors_dict.keys(), key=len, reverse=True)
    if not sorted_terms:
        return None
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in sorted_terms) + r")(?![a-z0-9])")

@functools.lru_cache(maxsize=10000)
def _extract_flavors_from_sku_cached(sku_lower: str) -> frozenset:
    fdata = _load_flavor_data()
    flavors_dict = fdata.get("flavors_dict", {})
    if not sku_lower or not flavors_dict:
        return frozenset()
    
    combined_pattern = fdata.get("combined_pattern")
    if combined_pattern is None:
        combined_pattern = _get_combined_flavor_pattern(flavors_dict)
        fdata["combined_pattern"] = combined_pattern

    result = set()
    if combined_pattern:
        for match in combined_pattern.finditer(sku_lower):
            canonical = flavors_dict.get(match.group(1))
            if canonical:
                result.add(canonical)
    return frozenset(result)

_FLAVOR_CACHE = {}

def _load_flavor_data():
    global _FLAVOR_CACHE
    if _FLAVOR_CACHE:
        if "combined_pattern" not in _FLAVOR_CACHE and "flavors_dict" in _FLAVOR_CACHE:
            _FLAVOR_CACHE["combined_pattern"] = _get_combined_flavor_pattern(_FLAVOR_CACHE["flavors_dict"])
        return _FLAVOR_CACHE
    
    brands_cache = os.path.join(config.CACHE_DIR, "food_brands_mmap.feather")
    brands_df = pd.DataFrame()
    if os.path.exists(brands_cache):
        try:
            brands_df = pd.read_feather(brands_cache)
        except Exception as e:
            logger.warning(f"Failed to read Feather cache for flavor rules: {e}")
            
    if brands_df.empty:
        try:
            from engine.data_pipeline.ingestion import DataIngestion
            _, brands_df = DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, "food")
        except Exception as e:
            logger.warning(f"Failed to load flavor sheet for flavor rules: {e}")
            brands_df = pd.DataFrame()

    flavors_dict, meat_flavors, vegetable_flavors, seafood_flavors, _ = build_food_flavors_info(brands_df)
    sorted_terms = sorted(flavors_dict.keys(), key=len, reverse=True)
    combined_pattern = _get_combined_flavor_pattern(flavors_dict)
            
    _FLAVOR_CACHE = {
        "flavors_dict": flavors_dict,
        "meat_flavors": meat_flavors,
        "vegetable_flavors": vegetable_flavors,
        "seafood_flavors": seafood_flavors,
        "sorted_flavor_terms": sorted_terms,
        "combined_pattern": combined_pattern
    }
    _extract_flavors_from_sku_cached.cache_clear()
    return _FLAVOR_CACHE

def _extract_flavors_from_sku(sku_name: str, flavors_dict: dict = None) -> set:
    if not sku_name:
        return set()
    fdata = _load_flavor_data()
    actual_dict = flavors_dict if flavors_dict is not None else fdata.get("flavors_dict", {})
    if not actual_dict:
        return set()

    # Fast-path: use the LRU cached helper when standard cache is active
    if flavors_dict is None or flavors_dict is fdata.get("flavors_dict"):
        return set(_extract_flavors_from_sku_cached(sku_name.lower().strip()))

    combined_pattern = _get_combined_flavor_pattern(actual_dict)
    if not combined_pattern:
        return set()
    result = set()
    for match in combined_pattern.finditer(sku_name.lower()):
        canonical = actual_dict.get(match.group(1))
        if canonical:
            result.add(canonical)
    return result

def _evaluate_single_condition(cond: dict, record: dict) -> bool:
    ctype = cond["condition_type"]
    val = cond["value"]
    negate = cond.get("negate", False)
    
    result = False
    
    if ctype == "sku_contains":
        sku = str(record.get("sku_name", "")).lower()
        result = val.lower() in sku
        
    elif ctype == "bt_is":
        bt = str(record.get("bt") or "").lower()
        result = bt == val.lower()
        
    elif ctype == "gk_contains":
        gks = [str(g).lower() for g in record.get("gk", [])]
        result = val.lower() in gks
        
    elif ctype == "category_contains":
        cat = str(record.get("category") or "").lower()
        result = val.lower() in cat
        
    elif ctype == "region_is":
        reg = str(record.get("region") or "").lower()
        result = reg == val.lower()
        
    elif ctype == "price_below":
        price = record.get("price")
        if price is not None:
            result = float(price) < float(val)
            
    elif ctype == "price_above":
        price = record.get("price")
        if price is not None:
            result = float(price) > float(val)

    elif ctype == "flavor_contains":
        sku_name = str(record.get("sku_name", ""))
        fdata = _load_flavor_data()
        extracted = _extract_flavors_from_sku(sku_name, fdata["flavors_dict"])
        target_val = val.lower().strip()
        canonical_target = fdata["flavors_dict"].get(target_val, target_val)
        result = canonical_target in extracted

    elif ctype == "flavor_is":
        sku_name = str(record.get("sku_name", ""))
        fdata = _load_flavor_data()
        extracted = _extract_flavors_from_sku(sku_name, fdata["flavors_dict"])
        category = val.lower().strip()
        if category == "meat":
            result = any(f in fdata["meat_flavors"] for f in extracted)
        elif category in ("vegetable", "veg"):
            result = any(f in fdata["vegetable_flavors"] for f in extracted)
        elif category == "seafood":
            result = any(f in fdata["seafood_flavors"] for f in extracted)
        else:
            result = False
            
    return not result if negate else result
