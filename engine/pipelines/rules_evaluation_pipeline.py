import functools
import json
import logging
from pathlib import Path

import pandas as pd

from engine.core import config
from engine.pipelines.rules_loader_pipeline import Rule, get_rules
from engine.preprocessing.flavor_preprocessor import (
    build_food_flavors_info,
    compile_flavor_pattern,
    extract_canonical_flavors,
)

logger = logging.getLogger("matchops.rules_evaluator")
_FLAVOR_CACHE: dict = {}


def apply_actions(rule, record: dict) -> str:
    """Applies all actions of a rule to the record. Returns a summary of changes."""
    changes = []

    for action in rule.actions:
        atype = action["action_type"]
        val = action["value"]

        try:
            if atype == "set_bt":
                old_bt = record.get("bt")
                if old_bt != val:
                    record["bt"] = val
                    changes.append(f"BT changed to '{val}'")

            elif atype == "add_gk":
                raw_gk = record.get("gk")
                gk_list = (
                    list(raw_gk)
                    if isinstance(raw_gk, (list, set, tuple))
                    else ([] if raw_gk is None else [raw_gk])
                )
                if val not in gk_list:
                    gk_list.append(val)
                    record["gk"] = gk_list
                    changes.append(f"Added GK '{val}'")
                elif "gk" not in record or record["gk"] is None:
                    record["gk"] = gk_list

            elif atype == "remove_gk":
                raw_gk = record.get("gk")
                gk_list = (
                    list(raw_gk)
                    if isinstance(raw_gk, (list, set, tuple))
                    else ([] if raw_gk is None else [raw_gk])
                )
                if val in gk_list:
                    gk_list.remove(val)
                    record["gk"] = gk_list
                    changes.append(f"Removed GK '{val}'")
                elif "gk" not in record or record["gk"] is None:
                    record["gk"] = gk_list

            elif atype == "set_region":
                old_reg = record.get("region")
                if old_reg != val:
                    record["region"] = val
                    changes.append(f"Region changed to '{val}'")

            elif atype == "set_category":
                old_cat = record.get("category")
                if old_cat != val:
                    record["category"] = val
                    changes.append(f"Category changed to '{val}'")

        except Exception as e:
            logger.warning(f"Error applying action {action} for rule {rule.rule_id}: {e}")

    return "; ".join(changes)


@functools.lru_cache(maxsize=10000)
def _extract_flavors_from_sku_cached(sku_lower: str) -> frozenset:
    fdata = _load_flavor_data()
    flavors_dict = fdata.get("flavors_dict", {})
    if not sku_lower or not flavors_dict:
        return frozenset()

    combined_pattern = fdata.get("combined_pattern")
    if combined_pattern is None:
        combined_pattern = compile_flavor_pattern(flavors_dict)
        fdata["combined_pattern"] = combined_pattern

    return frozenset(extract_canonical_flavors(sku_lower, flavors_dict, pattern=combined_pattern))


def clear_flavor_cache() -> None:
    """Clears the cached food flavor dictionary so the next flavor_contains/flavor_is rule
    evaluation reloads it from the current brands/flavors catalog."""
    global _FLAVOR_CACHE
    _FLAVOR_CACHE = {}
    _extract_flavors_from_sku_cached.cache_clear()


def _load_flavor_data() -> dict:
    global _FLAVOR_CACHE
    if _FLAVOR_CACHE:
        if "combined_pattern" not in _FLAVOR_CACHE and "flavors_dict" in _FLAVOR_CACHE:
            _FLAVOR_CACHE["combined_pattern"] = compile_flavor_pattern(
                _FLAVOR_CACHE["flavors_dict"]
            )
        return _FLAVOR_CACHE

    brands_cache = Path(config.CACHE_DIR) / "food_brands_mmap.feather"
    brands_df = pd.DataFrame()
    if brands_cache.exists():
        try:
            brands_df = pd.read_feather(brands_cache)
        except Exception as e:
            logger.warning(f"Failed to read Feather cache for flavor rules: {e}")

    flavors_dict, meat_flavors, vegetable_flavors, seafood_flavors, _ = build_food_flavors_info(
        brands_df
    )
    sorted_terms = sorted(flavors_dict.keys(), key=len, reverse=True)
    combined_pattern = compile_flavor_pattern(flavors_dict)

    _FLAVOR_CACHE = {
        "flavors_dict": flavors_dict,
        "meat_flavors": meat_flavors,
        "vegetable_flavors": vegetable_flavors,
        "seafood_flavors": seafood_flavors,
        "sorted_flavor_terms": sorted_terms,
        "combined_pattern": combined_pattern,
    }
    _extract_flavors_from_sku_cached.cache_clear()
    return _FLAVOR_CACHE


def _extract_flavors_from_sku(sku_name: str, flavors_dict: dict | None = None) -> set:
    if not sku_name:
        return set()
    fdata = _load_flavor_data()
    actual_dict = flavors_dict if flavors_dict is not None else fdata.get("flavors_dict", {})
    if not actual_dict:
        return set()

    if flavors_dict is None or flavors_dict is fdata.get("flavors_dict"):
        return set(_extract_flavors_from_sku_cached(sku_name.lower().strip()))

    return extract_canonical_flavors(sku_name, actual_dict)


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
            seafood_set = fdata.get("seafood_flavors", set())
            result = any(
                f in fdata["meat_flavors"] and f not in seafood_set and f not in ("egg", "mixed")
                for f in extracted
            )
        elif category in ("vegetable", "veg"):
            result = any(f in fdata["vegetable_flavors"] for f in extracted)
        elif category == "seafood":
            result = any(f in fdata["seafood_flavors"] for f in extracted)
        else:
            result = False

    return not result if negate else result


def evaluate_conditions(rule, record: dict) -> bool:
    """Evaluates all conditions for a rule against a given record."""
    if not rule.conditions:
        return True

    if any(cond.get("operator") is not None for cond in rule.conditions):
        rule_logic = str(getattr(rule, "condition_logic", "AND") or "AND").upper()
        or_groups: list[list[dict]] = []
        for idx, cond in enumerate(rule.conditions):
            op = str(cond.get("operator") or "AND").upper()
            group_changed = (
                idx > 0
                and rule_logic == "OR"
                and cond.get("condition_group", 1)
                != rule.conditions[idx - 1].get("condition_group", 1)
            )
            if idx == 0 or op == "OR" or group_changed:
                or_groups.append([cond])
            else:
                or_groups[-1].append(cond)

        for group in or_groups:
            group_passed = True
            for cond in group:
                try:
                    if not _evaluate_single_condition(cond, record):
                        group_passed = False
                        break
                except Exception as e:
                    logger.warning(
                        f"Error evaluating condition {cond} for rule {rule.rule_id}: {e}"
                    )
                    group_passed = False
                    break
            if group_passed:
                return True
        return False

    # Legacy fallback: group conditions by condition_group with rule-level condition_logic
    groups = {}
    for cond in rule.conditions:
        cg = cond.get("condition_group", 1)
        if cg not in groups:
            groups[cg] = []
        groups[cg].append(cond)

    group_results = []

    for group_conditions in groups.values():
        type_groups = {}
        for cond in group_conditions:
            ctype = cond["condition_type"]
            if ctype not in type_groups:
                type_groups[ctype] = []
            type_groups[ctype].append(cond)

        group_passed = True

        for conditions_of_type in type_groups.values():
            type_passed = False
            for cond in conditions_of_type:
                try:
                    res = _evaluate_single_condition(cond, record)
                    if res:
                        type_passed = True
                        break
                except Exception as e:
                    logger.warning(
                        f"Error evaluating condition {cond} for rule {rule.rule_id}: {e}"
                    )

            if not type_passed:
                group_passed = False
                break

        group_results.append(group_passed)

    logic = rule.condition_logic.upper()
    if logic == "AND":
        return all(group_results)
    return any(group_results)


def run_rules_engine(record: dict) -> dict:
    """
    Main orchestrator for the Post-Processing Rules Engine.
    Receives a structured record, applies rules, and returns augmented record.
    """
    domain = record.get("domain")
    if not domain:
        logger.warning("No domain found in record, skipping rules engine.")
        return record

    rules_applied = list(record.get("rules_applied") or [])
    rules = get_rules(domain=domain)

    for rule in rules:
        if evaluate_conditions(rule, record):
            changes = apply_actions(rule, record)
            if changes:
                rules_applied.append(
                    {
                        "rule_id": rule.rule_id,
                        "description": rule.description,
                        "change": changes,
                        "reasoning": rule.reasoning,
                    }
                )

    record["rules_applied"] = rules_applied
    return record


def rerun_rules_for_sku(
    sku_name: str,
    domain: str,
    bt: str,
    gk_csv: str,
    region: str,
    category: str,
    price: float,
    confidence: float,
    match_source: str,
    matched_sku: str,
    reasoning: str,
) -> dict[str, str]:
    """Re-evaluates active rules for one SKU record and returns formatted tag fields."""
    gk_list = [x.strip() for x in (gk_csv or "").split(",") if x.strip()]
    record = {
        "sku_name": sku_name,
        "domain": domain,
        "bt": bt,
        "gk": gk_list,
        "region": region if domain == "food" else None,
        "category": category if domain == "market" else None,
        "price": price,
        "confidence": confidence,
        "match_source": match_source,
        "matched_sku": matched_sku,
        "reasoning": reasoning,
        "rules_applied": [],
    }
    aug_record = run_rules_engine(record)
    rules_applied = aug_record.get("rules_applied", [])
    return {
        "suggested_bt": str(aug_record.get("bt") or ""),
        "suggested_gk": ", ".join(aug_record.get("gk", [])),
        "suggested_region": str(aug_record.get("region") or aug_record.get("category") or ""),
        "rules_applied": json.dumps(rules_applied) if rules_applied else "",
    }


def test_single_rule(rule_dict: dict, sample_record: dict) -> dict:
    """Evaluates an unsaved rule definition against a sample record for live preview."""
    row_values = (
        rule_dict.get("id", 0),
        rule_dict["rule_id"],
        rule_dict["domain"],
        rule_dict["priority"],
        rule_dict["description"],
        rule_dict["reasoning"],
        rule_dict["condition_logic"],
        rule_dict.get("is_active", 1),
    )
    engine_rule = Rule(row_values)
    engine_rule.conditions = rule_dict.get("conditions", [])
    engine_rule.actions = rule_dict.get("actions", [])

    record = dict(sample_record)
    fires = bool(evaluate_conditions(engine_rule, record))
    changes = apply_actions(engine_rule, record) if fires else ""
    return {
        "rule_id": rule_dict["rule_id"],
        "fires": fires,
        "matched_conditions": rule_dict.get("conditions", []) if fires else [],
        "actions_applied": rule_dict.get("actions", []) if fires else [],
        "changes": str(changes),
        "sample_record_after": record,
    }
