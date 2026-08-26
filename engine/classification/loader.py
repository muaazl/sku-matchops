import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("matchops.loader")

# Single-word names that carry almost no food signal
_DEGENERATE_NAMES = {
    "small", "medium", "large", "extra large", "xl",
    "chicken", "fish", "beef", "mutton", "prawn", "prawns",
    "vegetable", "veg", "egg", "seafood", "mixed", "mix",
    "regular", "special", "normal", "half", "full", "quarter",
}

def is_degenerate_name(name: str) -> bool:
    """Returns True if the SKU name is a single generic word with minimal context."""
    return name.strip().lower() in _DEGENERATE_NAMES

def build_sku_query(row: pd.Series, col_name: str, col_desc: str, col_cat: str) -> str:
    """Combines SKU metadata into a single enriched query string for encoding."""
    parts = []
    for col in [col_name, col_desc, col_cat]:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            parts.append(val)
    return " | ".join(parts)

def build_descriptions(domain: str, dicts: Dict[str, List[str]], bt_gk_map: Optional[Dict[str, List[str]]] = None, bt_third_tag_map: Optional[Dict[str, str]] = None) -> Dict[str, dict]:
    """Builds semantic descriptions for BT and third-tag classes."""
    bt_descriptions = {}
    for bt in dicts.get("bt", []):
        if bt_gk_map:
            keywords = bt_gk_map.get(bt, [])[:8]
            bt_descriptions[bt] = f"{bt}: {', '.join(keywords)}" if keywords else bt
        else:
            bt_descriptions[bt] = bt

    # Enriched region descriptions: gives the zero-shot classifier real semantic
    # signal instead of matching on the bare country/cuisine name alone.
    # Coverage matches the 16 live labels returned by the Food_Region sheet tab.
    _REGION_DESCRIPTIONS = {
        "Sri Lankan": (
            "Sri Lankan traditional food including short eats, fish patty, chicken roll, "
            "kottu roti, rice and curry, string hoppers, pol sambol, hoppers, lamprais, "
            "pittu, wade, seeni sambol bun"
        ),
        "North Indian": (
            "North Indian cuisine including naan, butter chicken, biryani, paneer tikka, "
            "dal makhani, roti, tandoori chicken, chole bhature, aloo paratha, lassi, "
            "samosa, kebab"
        ),
        "South Indian": (
            "South Indian cuisine including dosa, idli, sambar, rasam, uttapam, vada, "
            "coconut chutney, pongal, tamarind rice, fish curry, appam, prawn masala"
        ),
        "Sri Lankan Chinese": (
            "Sri Lankan-style Chinese food including fried rice, chow mein, noodles, "
            "devilled chicken, devilled pork, sweet and sour, kottu, egg fried rice, "
            "wonton soup, spring rolls, devilled prawns"
        ),
        "Italian": (
            "Italian cuisine including pasta, pizza, risotto, lasagna, carbonara, "
            "bruschetta, tiramisu, gnocchi, minestrone, fettuccine, pesto, calzone"
        ),
        "Mediterranean": (
            "Mediterranean food including hummus, falafel, grilled fish, tabbouleh, "
            "pita bread, shawarma, moussaka, baklava, olive oil dishes, mezze, "
            "stuffed grape leaves, tzatziki"
        ),
        "French": (
            "French cuisine including croissant, baguette, quiche, crepe, eclair, "
            "french onion soup, coq au vin, ratatouille, profiterole, brioche, "
            "boeuf bourguignon, souffle"
        ),
        "Western": (
            "Western food including burger, sandwich, steak, grilled chicken, pasta, "
            "fish and chips, club sandwich, hot dog, caesar salad, mac and cheese, "
            "bacon egg roll, mashed potato"
        ),
        "Mongolian": (
            "Mongolian cuisine including Mongolian beef, Mongolian lamb, stir-fried noodles, "
            "tsuivan, buuz dumplings, khuushuur, grilled meat, meat stew, mutton soup, "
            "Mongolian barbecue, steamed buns"
        ),
        "Thai": (
            "Thai cuisine including pad thai, green curry, tom yum soup, massaman curry, "
            "thai fried rice, papaya salad, spring rolls, satay, mango sticky rice, "
            "red curry, thai basil chicken, coconut soup"
        ),
        "Mexican": (
            "Mexican food including tacos, burrito, nachos, quesadilla, enchilada, "
            "guacamole, salsa, fajita, churros, tamale, chilli con carne, tortilla"
        ),
        "Japanese": (
            "Japanese cuisine including sushi, sashimi, ramen, udon, miso soup, "
            "tempura, gyoza, teriyaki chicken, onigiri, katsu curry, edamame, yakitori"
        ),
        "Vietnamese": (
            "Vietnamese food including pho, banh mi, spring rolls, bun bo hue, "
            "vietnamese iced coffee, goi cuon, bun cha, com tam, banh xeo, "
            "lemongrass chicken, vermicelli noodles"
        ),
        "German": (
            "German food including bratwurst, sauerkraut, schnitzel, pretzels, "
            "black forest cake, strudel, rye bread, beef rouladen, potato dumpling, "
            "currywurst, sausage platter, sauerbraten"
        ),
        "Middle Eastern": (
            "Middle Eastern cuisine including shawarma, kebab, hummus, falafel, "
            "pita bread, baklava, moutabal, fattoush, manakeesh, kibbeh, "
            "lentil soup, grilled halloumi"
        ),
        "Chinese": (
            "Chinese cuisine including fried rice, noodles, dim sum, spring rolls, "
            "wonton, chow mein, peking duck, steamed buns, kung pao chicken, "
            "mapo tofu, hot pot, dumplings"
        ),
    }

    third_tag_descriptions = {}
    third_tag_key = "region" if domain == "food" else "category"
    for label in dicts.get(third_tag_key, []):
        if label:
            third_tag_descriptions[label] = _REGION_DESCRIPTIONS.get(label, label)

    return {
        "bt_descriptions": bt_descriptions,
        "third_tag_descriptions": third_tag_descriptions,
        "third_tag_overrides": bt_third_tag_map or {},
        "bt_to_gk_umbrella": {},
        "bt_gk_map": bt_gk_map or {},
    }

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds the first matching column name from candidates, ignoring case and whitespace."""
    if df is None or df.empty:
        return None
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        cand_key = str(cand).strip().lower()
        if cand_key in col_map:
            return col_map[cand_key]
    return None

def build_bt_third_tag_map_from_catalog(cat_df: pd.DataFrame, domain: str) -> Dict[str, str]:
    """Mines the catalog to discover the most frequent Region/Category for each Basic Type."""
    if cat_df.empty:
        return {}
        
    bt_col = _find_column(cat_df, ["basictype", "BasicType", "basic_type", "BT"])
    if not bt_col:
        return {}
        
    from engine.config import get_third_tag_col
    target_name = get_third_tag_col(domain)
    target_col = _find_column(cat_df, [target_name, "region", "Region", "category", "Categories", "Categories / generic keywords"])
    
    if not target_col:
        return {}

    try:
        df = cat_df.fillna("")
        df = df[(df[bt_col].astype(str).str.strip() != "") & (df[target_col].astype(str).str.strip() != "")].copy()

        bt_map = {}
        for bt, group in df.groupby(bt_col):
            bt = str(bt).strip()
            if not bt:
                continue
            
            # Find the most common third tag for this BT
            most_common = str(group[target_col].value_counts().idxmax()).strip()
            if most_common:
                bt_map[bt] = most_common

        logger.info(f"[Loader] Mined {len(bt_map)} BT->{target_col.capitalize()} overrides from catalog.")
        return bt_map
    except Exception as e:
        logger.error(f"build_bt_third_tag_map_from_catalog failed: {e}")
        return {}

def build_umbrella_from_training(cat_df: pd.DataFrame, threshold: float = 0.60) -> Dict[str, List[str]]:
    """Mines the catalog to discover GK tags that reliably co-occur with specific BTs."""
    if cat_df.empty:
        return {}

    bt_col = _find_column(cat_df, ["basictype", "BasicType", "basic_type", "BT"])
    gk_col = _find_column(cat_df, ["Generic keywords", "GenericKeywords", "generic_keywords", "GK"])
    if not bt_col or not gk_col:
        return {}

    try:
        df = cat_df.fillna("")
        df = df[(df[bt_col].astype(str).str.strip() != "") & (df[gk_col].astype(str).str.strip() != "")].copy()

        umbrella = {}
        for bt, group in df.groupby(bt_col):
            bt = str(bt).strip()
            n = len(group)
            if n < 5:
                continue

            tag_counts = Counter()
            for tags_str in group[gk_col]:
                for tag in str(tags_str).split(","):
                    tag = tag.strip()
                    if tag:
                        tag_counts[tag] += 1

            common = [tag for tag, cnt in tag_counts.items() if cnt / n >= threshold]
            common.sort(key=lambda t: -tag_counts[t])
            if common:
                umbrella[bt] = common

        logger.info(f"[Umbrella] Mined {len(umbrella)} umbrella mappings from {len(df)} rows.")
        return umbrella
    except Exception as e:
        logger.error(f"build_umbrella_from_training failed: {e}")
        return {}

def augment_bt_gk_map_with_training(cat_df: pd.DataFrame, bt_gk_map: Dict[str, List[str]], gk_dict_list: List[str]) -> Dict[str, List[str]]:
    """Augments the BT->GK map using co-occurrences found in the catalog."""
    if cat_df.empty:
        return bt_gk_map

    bt_col = _find_column(cat_df, ["basictype", "BasicType", "basic_type", "BT"])
    gk_col = _find_column(cat_df, ["Generic keywords", "GenericKeywords", "generic_keywords", "GK"])
    if not bt_col or not gk_col:
        return bt_gk_map

    gk_set = {k.strip().lower(): k.strip() for k in gk_dict_list}
    augmented = {k: list(v) for k, v in bt_gk_map.items()}

    try:
        df = cat_df.fillna("")
        for bt, group in df.groupby(bt_col):
            bt_stripped = str(bt).strip()
            if not bt_stripped:
                continue

            current_kws_lower = {kw.lower() for kw in augmented.get(bt_stripped, [])}
            current_list = augmented.setdefault(bt_stripped, [])

            for tags_str in group[gk_col]:
                for tag in str(tags_str).split(","):
                    tag_stripped = tag.strip()
                    if not tag_stripped:
                        continue
                    tag_lower = tag_stripped.lower()
                    if tag_lower in gk_set and tag_lower not in current_kws_lower:
                        current_list.append(gk_set[tag_lower])
                        current_kws_lower.add(tag_lower)

        logger.info("[Loader] Augmented BT->GK map with training data.")
    except Exception as e:
        logger.error(f"Failed to augment BT->GK map: {e}")

    return augmented

def mine_umbrella_words_from_training(cat_df: pd.DataFrame, min_bts: int = 10, max_match_frac: float = 0.30) -> Set[str]:
    """Discovers generic category words that appear across many BTs but rarely in SKU names."""
    if cat_df.empty:
        return set()

    try:
        df = cat_df.fillna("")
        if not all(col in df.columns for col in ["basictype", "Generic keywords", "Name"]):
            return set()

        word_bt_count = defaultdict(set)
        word_total = defaultdict(int)
        word_sku_match = defaultdict(int)

        for _, row in df.iterrows():
            name_words = {w.strip().translate(str.maketrans("", "", "-&'")).lower() for w in str(row["Name"]).split()}
            gks = [t.strip().lower() for t in str(row["Generic keywords"]).split(",") if t.strip()]
            bt = str(row["basictype"]).strip()

            for gk in gks:
                gk_words = {w.strip().translate(str.maketrans("", "", "-&'")).lower() for w in gk.split()}
                for w in gk_words:
                    if not w: continue
                    word_total[w] += 1
                    if bt: word_bt_count[w].add(bt)
                    if w in name_words: word_sku_match[w] += 1

        umbrella_words = {w for w, bts in word_bt_count.items() if len(bts) >= min_bts and (word_sku_match[w] / word_total[w]) < max_match_frac}
        logger.info(f"[Loader] Mined {len(umbrella_words)} generic umbrella words.")
        return umbrella_words
    except Exception as e:
        logger.error(f"mine_umbrella_words_from_training failed: {e}")
        return set()
