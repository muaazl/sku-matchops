"""
SKU MatchOps Backend - Local Rule Evaluation Fallback
Provides standalone rule condition evaluation and action application within the API Gateway
when the remote ML Engine gRPC server is offline, unreachable, or in unit test mode.
"""

import logging
from typing import Any

logger = logging.getLogger("matchops.backend.rules_evaluator")


def _evaluate_single_condition(cond: dict[str, Any], record: dict[str, Any]) -> bool:
    ctype = cond.get("condition_type", "")
    val = str(cond.get("value", ""))
    negate = bool(cond.get("negate", False))
    result = False

    try:
        if ctype == "sku_contains":
            sku = str(record.get("sku_name", "")).lower()
            result = val.lower() in sku

        elif ctype == "bt_is":
            bt = str(record.get("bt") or "").lower()
            result = bt == val.lower()

        elif ctype == "gk_contains":
            raw_gk = record.get("gk", [])
            gks = [
                str(g).lower()
                for g in (raw_gk if isinstance(raw_gk, list) else str(raw_gk).split(","))
            ]
            result = val.lower() in [g.strip() for g in gks]

        elif ctype == "category_contains":
            cat = str(record.get("category") or "").lower()
            result = val.lower() in cat

        elif ctype == "region_is":
            reg = str(record.get("region") or "").lower()
            result = reg == val.lower()

        elif ctype == "price_below":
            price = record.get("price")
            if price is not None and price != "":
                result = float(price) < float(val)

        elif ctype == "price_above":
            price = record.get("price")
            if price is not None and price != "":
                result = float(price) > float(val)

        elif ctype in ("flavor_contains", "flavor_is"):
            sku = str(record.get("sku_name", "")).lower()
            result = val.lower() in sku
    except Exception as e:
        logger.debug(f"Condition evaluation error for {cond}: {e}")
        result = False

    return not result if negate else result


def evaluate_rule_conditions(rule_dict: dict[str, Any], record: dict[str, Any]) -> bool:
    conditions = rule_dict.get("conditions", [])
    if not conditions:
        return True

    groups: dict[int, list[dict[str, Any]]] = {}
    for c in conditions:
        cg = c.get("condition_group", 1)
        groups.setdefault(cg, []).append(c)

    group_results = []
    for group_conditions in groups.values():
        type_groups: dict[str, list[dict[str, Any]]] = {}
        for c in group_conditions:
            type_groups.setdefault(c.get("condition_type", ""), []).append(c)

        group_passed = True
        for conds_of_type in type_groups.values():
            type_passed = any(_evaluate_single_condition(c, record) for c in conds_of_type)
            if not type_passed:
                group_passed = False
                break
        group_results.append(group_passed)

    logic = str(rule_dict.get("condition_logic", "AND")).upper()
    return all(group_results) if logic == "AND" else any(group_results)


def apply_rule_actions(rule_dict: dict[str, Any], record: dict[str, Any]) -> list[str]:
    changes = []
    for action in rule_dict.get("actions", []):
        atype = action.get("action_type", "")
        val = action.get("value", "")
        try:
            if atype == "set_bt":
                if record.get("bt") != val:
                    record["bt"] = val
                    changes.append(f"BT changed to '{val}'")
            elif atype == "add_gk":
                raw_gk = record.get("gk")
                gk_list = (
                    list(raw_gk)
                    if isinstance(raw_gk, list)
                    else [g.strip() for g in str(raw_gk or "").split(",") if g.strip()]
                )
                if val not in gk_list:
                    gk_list.append(val)
                    record["gk"] = gk_list
                    changes.append(f"Added GK '{val}'")
            elif atype == "remove_gk":
                raw_gk = record.get("gk")
                gk_list = (
                    list(raw_gk)
                    if isinstance(raw_gk, list)
                    else [g.strip() for g in str(raw_gk or "").split(",") if g.strip()]
                )
                if val in gk_list:
                    gk_list.remove(val)
                    record["gk"] = gk_list
                    changes.append(f"Removed GK '{val}'")
            elif atype == "set_region":
                if record.get("region") != val:
                    record["region"] = val
                    changes.append(f"Region changed to '{val}'")
            elif atype == "set_category":
                if record.get("category") != val:
                    record["category"] = val
                    changes.append(f"Category changed to '{val}'")
            elif atype == "normalize_sku":
                sku = str(record.get("sku_name") or "")
                if "|" in val:
                    old_s, new_s = val.split("|", 1)
                    if old_s in sku:
                        record["sku_name"] = sku.replace(old_s, new_s)
                        changes.append(f"Normalized SKU substring '{old_s}' to '{new_s}'")
        except Exception as e:
            logger.debug(f"Action application error for {action}: {e}")
    return changes


def evaluate_rule_locally(
    rule_dict: dict[str, Any], sample_record: dict[str, Any]
) -> dict[str, Any]:
    record = dict(sample_record)
    fires = evaluate_rule_conditions(rule_dict, record)
    changes = []
    if fires:
        changes = apply_rule_actions(rule_dict, record)
    return {
        "rule_id": rule_dict.get("rule_id", ""),
        "fires": fires,
        "matched_conditions": rule_dict.get("conditions", []) if fires else [],
        "actions_applied": rule_dict.get("actions", []) if fires else [],
        "changes": "; ".join(changes),
        "sample_record_after": record,
    }


def rerun_rules_locally(
    sku_name: str,
    domain: str = "market",
    bt: str | None = None,
    gk: str | None = None,
    region: str | None = None,
    category: str | None = None,
    price: float | None = 0.0,
    confidence: float | None = 1.0,
    match_source: str | None = "Classifier",
    matched_sku: str | None = "",
    reasoning: str | None = "",
) -> dict[str, Any]:
    from backend.app.services.rules_service import load_active_rules_from_db

    record = {
        "sku_name": sku_name,
        "bt": bt,
        "gk": [g.strip() for g in (gk or "").split(",") if g.strip()],
        "region": region,
        "category": category,
        "price": price,
        "confidence": confidence,
        "match_source": match_source,
        "matched_sku": matched_sku,
        "reasoning": reasoning,
    }
    rules = load_active_rules_from_db()
    applied_rules = []
    for r in sorted(rules, key=lambda x: x.get("priority", 100)):
        if evaluate_rule_conditions(r, record):
            apply_rule_actions(r, record)
            applied_rules.append(r.get("rule_id", ""))

    raw_gk = record.get("gk", [])
    gk_str = ", ".join(raw_gk) if isinstance(raw_gk, list) else str(raw_gk or "")
    return {
        "suggested_bt": str(record.get("bt") or ""),
        "suggested_gk": gk_str,
        "suggested_region": str(record.get("region") or record.get("category") or ""),
        "rules_applied": ", ".join(applied_rules),
    }
