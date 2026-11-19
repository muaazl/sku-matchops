import pandas as pd


def _get_catalog_row(raw_catalog: pd.DataFrame | list[dict], idx: int):
    return raw_catalog[idx] if isinstance(raw_catalog, list) else raw_catalog.iloc[idx]


def resolve_bt_consensus_indices(
    raw_catalog: pd.DataFrame | list[dict],
    cand_indices: list[int],
    conf_bt: str | None = None,
    cat_bts_lower: list[str] | None = None,
) -> list[int]:
    """
    Filters catalog candidate indices tied on exact or token-sorted text to a consistent BasicType.
    If conf_bt matches a subset of cand_indices and is not outnumbered >=3:1 by the catalog
    consensus BasicType, returns the conf_bt-matching indices; otherwise returns the indices
    matching the majority-vote BasicType across cand_indices.
    """
    if not cand_indices or len(cand_indices) <= 1:
        return list(cand_indices)

    bts: list[str] = []
    bt_counts: dict[str, int] = {}
    for idx in cand_indices:
        if cat_bts_lower is not None and idx < len(cat_bts_lower):
            bt = cat_bts_lower[idx]
        else:
            row = _get_catalog_row(raw_catalog, idx)
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
    Given catalog row indices tied on token-sorted text, picks the one whose
    weight is numerically closest to the input's (same physical-form type only).

    Returns (selected_idx, weight_reason), where weight_reason is an empty
    string, a " | Weight Match (...)" suffix, or a " | Weight Mismatch (...)" suffix.
    """
    selected_idx = cand_indices[0]
    weight_reason = ""

    if input_w_data[0] is None:
        return selected_idx, weight_reason

    in_val, _, in_type = input_w_data
    best_match_idx = None
    min_diff_pct = float("inf")

    for c_idx in cand_indices:
        cat_row = _get_catalog_row(raw_catalog, c_idx)
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
        cat_row = _get_catalog_row(raw_catalog, selected_idx)
        catalog_w_data = cat_row.get("weight_val")
        cat_val = catalog_w_data[0]
        if min_diff_pct < 1.0:
            weight_reason = f" | Weight Match ({int(in_val)})"
        else:
            weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(cat_val)})"
    else:
        cat_row = _get_catalog_row(raw_catalog, selected_idx)
        catalog_w_data = cat_row.get("weight_val")
        if (
            catalog_w_data is not None
            and isinstance(catalog_w_data, (tuple, list))
            and catalog_w_data[0] is not None
        ):
            weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(catalog_w_data[0])})"

    return selected_idx, weight_reason
