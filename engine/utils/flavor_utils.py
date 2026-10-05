import pandas as pd


def triage_input_flavors(
    rules,
    flavor_categories: dict[str, tuple[bool, bool, bool]],
    extracted_entities: dict | None,
    matched_row,
) -> tuple[set[str], set[str], set[str]]:
    """
    Resolves the input's flavor entities (falling back to the matched catalog row's
    entities if the input has none), then splits them into meat/seafood/vegetable
    canonical-name sets using flavor_categories.

    Returns (input_meats, input_seafoods, input_vegs).
    """
    input_flavors = set()
    if extracted_entities and isinstance(extracted_entities, dict):
        input_flavors.update(x.lower() for x in extracted_entities.get("flavor", set()) if x)
    if not input_flavors:
        cat_ents = matched_row.get("entities")
        if isinstance(cat_ents, dict):
            input_flavors.update(x.lower() for x in cat_ents.get("flavor", set()) if x)
    resolved_input_flavors = (
        rules._resolve_flavors(input_flavors)
        if hasattr(rules, "_resolve_flavors")
        else input_flavors
    )

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


def build_food_flavors_info(
    brands_df: pd.DataFrame,
) -> tuple[dict[str, str], set[str], set[str], set[str], dict[str, tuple[bool, bool, bool]]]:
    """
    Parses the brands/flavors DataFrame to extract flavor lookup mapping and category sets.

    Returns a tuple of:
      - flavors_dict: Maps flavor aliases and canonical names to canonical flavor name.
      - meat_flavors: Set of canonical meat flavor names.
      - vegetable_flavors: Set of canonical vegetable flavor names.
      - seafood_flavors: Set of canonical seafood flavor names.
      - flavor_categories: Maps flavor aliases and canonical names to (is_meat, is_veg, is_seafood) flags.
    """
    if brands_df is None or brands_df.empty:
        return {}, set(), set(), set(), {}

    cols = list(brands_df.columns)
    name_col = (
        "Flavor Name" if "Flavor Name" in cols else ("Brand Name" if "Brand Name" in cols else "")
    )
    if not name_col:
        return {}, set(), set(), set(), {}

    flavors_dict = {}
    meat_flavors = set()
    vegetable_flavors = set()
    seafood_flavors = set()
    flavor_categories = {}

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

    # Direct overrides for shrimp / prawn alignment
    if "shrimp" in flavors_dict and "prawn" in flavors_dict:
        flavors_dict["shrimp"] = "prawn"
        flavors_dict["prawn"] = "prawn"
        flavor_categories["shrimp"] = flavor_categories["prawn"]

    return flavors_dict, meat_flavors, vegetable_flavors, seafood_flavors, flavor_categories
