import logging

logger = logging.getLogger("matchops.rules_actions")


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
