import pandas as pd
from typing import Dict, Set, Tuple

def build_food_flavors_info(
    brands_df: pd.DataFrame
) -> Tuple[Dict[str, str], Set[str], Set[str], Set[str], Dict[str, Tuple[bool, bool, bool]]]:
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
    name_col = "Flavor Name" if "Flavor Name" in cols else ("Brand Name" if "Brand Name" in cols else "")
    if not name_col:
        return {}, set(), set(), set(), {}
        
    flavors_dict = {}
    meat_flavors = set()
    vegetable_flavors = set()
    seafood_flavors = set()
    flavor_categories = {}
    
    for _, row in brands_df.iterrows():
        flavor_name = str(row.get(name_col, "")).strip()
        if not flavor_name or flavor_name.lower() == "nan":
            continue
            
        canonical = flavor_name.lower()
        
        is_meat = str(row.get("Is_Meat", "")).strip().lower() in ("true", "1", "yes", "y") or bool(row.get("Is_Meat", False))
        is_veg = str(row.get("Is_Vegetable", "")).strip().lower() in ("true", "1", "yes", "y") or bool(row.get("Is_Vegetable", False))
        is_seafood = str(row.get("Is_Seafood", "")).strip().lower() in ("true", "1", "yes", "y") or bool(row.get("Is_Seafood", False))
        
        flags = (is_meat, is_veg, is_seafood)
        
        if is_meat:
            meat_flavors.add(canonical)
        if is_veg:
            vegetable_flavors.add(canonical)
        if is_seafood:
            seafood_flavors.add(canonical)
            
        flavors_dict[canonical] = canonical
        flavor_categories[canonical] = flags
        
        aliases_str = str(row.get("Aliases", ""))
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
