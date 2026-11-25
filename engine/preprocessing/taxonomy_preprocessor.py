"""Taxonomy description builders and catalog metadata co-occurrence mining helpers."""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from engine.core.config import (
    PRIMARY_DISH_TYPES,
    UMBRELLA_MINING_THRESHOLD,
    get_third_tag_col,
)

logger = logging.getLogger("matchops.preprocessing.taxonomy")

_REGION_DESCRIPTIONS: dict[str, str] = {
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


def build_taxonomy_descriptions(
    domain: str,
    dicts: dict[str, list[str]],
    bt_gk_map: dict[str, list[str]] | None = None,
    bt_third_tag_map: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Builds semantic descriptions and overrides for BasicType and Third-Tag classes."""
    bt_descriptions: dict[str, str] = {}
    for bt in dicts.get("bt", []):
        if bt_gk_map:
            keywords = bt_gk_map.get(bt, [])[:8]
            bt_descriptions[bt] = f"{bt}: {', '.join(keywords)}" if keywords else bt
        else:
            bt_descriptions[bt] = bt

    third_tag_descriptions: dict[str, str] = {}
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


def find_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Finds the first matching column name from candidates, ignoring case and whitespace."""
    if df is None or df.empty:
        return None
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        cand_key = str(cand).strip().lower()
        if cand_key in col_map:
            return col_map[cand_key]
    return None


def build_bt_third_tag_map(cat_df: pd.DataFrame, domain: str) -> dict[str, str]:
    """Mines the catalog to discover the most frequent Region/Category for each Basic Type."""
    if cat_df.empty:
        return {}

    bt_col = find_matching_column(cat_df, ["basictype", "BasicType", "basic_type", "BT"])
    if not bt_col:
        return {}

    target_name = get_third_tag_col(domain)
    target_col = find_matching_column(
        cat_df,
        [
            target_name,
            "region",
            "Region",
            "category",
            "Categories",
            "Categories / generic keywords",
        ],
    )
    if not target_col:
        return {}

    try:
        df = cat_df.fillna("")
        df = df[
            (df[bt_col].astype(str).str.strip() != "")
            & (df[target_col].astype(str).str.strip() != "")
        ].copy()

        bt_map: dict[str, str] = {}
        for bt, group in df.groupby(bt_col):
            bt_clean = str(bt).strip()
            if not bt_clean:
                continue
            most_common = str(group[target_col].value_counts().idxmax()).strip()
            if most_common:
                bt_map[bt_clean] = most_common

        logger.info(
            f"[TAXONOMY] Mined {len(bt_map)} BT->{target_col.capitalize()} overrides from catalog."
        )
        return bt_map
    except Exception as e:
        logger.error(f"build_bt_third_tag_map failed: {e}")
        return {}


def build_umbrella_gk_map(
    cat_df: pd.DataFrame, threshold: float = UMBRELLA_MINING_THRESHOLD
) -> dict[str, list[str]]:
    """Mines the catalog to discover GK tags that reliably co-occur with specific BTs."""
    if cat_df.empty:
        return {}

    bt_col = find_matching_column(cat_df, ["basictype", "BasicType", "basic_type", "BT"])
    gk_col = find_matching_column(
        cat_df, ["Generic keywords", "GenericKeywords", "generic_keywords", "GK"]
    )
    if not bt_col or not gk_col:
        return {}

    try:
        df = cat_df.fillna("")
        df = df[
            (df[bt_col].astype(str).str.strip() != "") & (df[gk_col].astype(str).str.strip() != "")
        ].copy()

        umbrella: dict[str, list[str]] = {}
        for bt, group in df.groupby(bt_col):
            bt_clean = str(bt).strip()
            n = len(group)
            if n < 5:
                continue

            tag_counts: Counter[str] = Counter()
            for tags_str in group[gk_col]:
                for tag in str(tags_str).split(","):
                    tag_clean = tag.strip()
                    if tag_clean:
                        tag_counts[tag_clean] += 1

            common = [tag for tag, cnt in tag_counts.items() if cnt / n >= threshold]
            common.sort(key=lambda t: -tag_counts[t])
            if common:
                umbrella[bt_clean] = common

        logger.info(f"[TAXONOMY] Mined {len(umbrella)} umbrella mappings from {len(df)} rows.")
        return umbrella
    except Exception as e:
        logger.error(f"build_umbrella_gk_map failed: {e}")
        return {}


def check_conflicting_dish_tag(tag: str, bt: str) -> bool:
    """Returns True if `tag` represents a conflicting primary dish type for `bt`."""
    tag_lower = str(tag or "").lower().strip()
    bt_lower = str(bt or "").lower().strip()
    if not bt_lower or not tag_lower:
        return False
    if tag_lower == bt_lower or tag_lower.endswith(bt_lower):
        return False

    curr_dish = None
    for dish in PRIMARY_DISH_TYPES:
        if bt_lower == dish or bt_lower.endswith(dish):
            curr_dish = dish
            break
    if not curr_dish:
        return False

    for other_dish in PRIMARY_DISH_TYPES:
        if other_dish != curr_dish and not bt_lower.endswith(other_dish):
            if tag_lower == other_dish or tag_lower.endswith(" " + other_dish):
                return True
    return False


def augment_bt_gk_map(
    cat_df: pd.DataFrame, bt_gk_map: dict[str, list[str]], gk_dict_list: list[str]
) -> dict[str, list[str]]:
    """Augments the BT->GK map using catalog co-occurrences, filtering conflicting dish types."""
    if cat_df.empty:
        return bt_gk_map

    bt_col = find_matching_column(cat_df, ["basictype", "BasicType", "basic_type", "BT"])
    gk_col = find_matching_column(
        cat_df, ["Generic keywords", "GenericKeywords", "generic_keywords", "GK"]
    )
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

            bt_lower = bt_stripped.lower()
            current_kws_lower = {kw.lower() for kw in augmented.get(bt_stripped, [])}
            current_list = augmented.setdefault(bt_stripped, [])

            for tags_str in group[gk_col]:
                for tag in str(tags_str).split(","):
                    tag_stripped = tag.strip()
                    if not tag_stripped:
                        continue
                    tag_lower = tag_stripped.lower()
                    if check_conflicting_dish_tag(tag_lower, bt_lower):
                        continue
                    if tag_lower in gk_set and tag_lower not in current_kws_lower:
                        current_list.append(gk_set[tag_lower])
                        current_kws_lower.add(tag_lower)

        logger.info("[TAXONOMY] Augmented BT->GK map with catalog co-occurrences.")
    except Exception as e:
        logger.error(f"augment_bt_gk_map failed: {e}")

    return augmented
