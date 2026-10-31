import pandas as pd


def resolve_weight_bypass_candidate(
    raw_catalog: pd.DataFrame,
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
        cat_row = raw_catalog.iloc[c_idx]
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
        cat_row = raw_catalog.iloc[selected_idx]
        catalog_w_data = cat_row.get("weight_val")
        cat_val = catalog_w_data[0]
        if min_diff_pct < 1.0:
            weight_reason = f" | Weight Match ({int(in_val)})"
        else:
            weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(cat_val)})"
    else:
        cat_row = raw_catalog.iloc[selected_idx]
        catalog_w_data = cat_row.get("weight_val")
        if (
            catalog_w_data is not None
            and isinstance(catalog_w_data, (tuple, list))
            and catalog_w_data[0] is not None
        ):
            weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(catalog_w_data[0])})"

    return selected_idx, weight_reason
