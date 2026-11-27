"""
SKU MatchOps Engine - Template-Aware Tag Suggestion & Enrichment
Extracts entity variations (brand, flavor) and matches against structural catalog templates.
"""

import logging
import re
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from engine.core import config
from engine.pipelines import runtime_pipeline
from engine.pipelines.ingestion_pipeline import DataIngestion
from engine.preprocessing.text_normalizer import (
    format_title_case,
    normalize_sku_text,
    standardize_sku_units,
)

logger = logging.getLogger("matchops.engine.template_suggest")

# Live reference cache keyed by domain so stale object IDs cannot collide across reloads.
_catalog_records_cache: dict[str, tuple] = {}


def get_classifier(domain: str):
    """Loads domain classifier through runtime_pipeline so monkeypatching works uniformly."""
    return runtime_pipeline.get_classifier(domain)


def get_catalog_and_brands(domain: str):
    """Load catalog and brands/flavors from local Feather cache or PostgreSQL database."""
    return DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, domain, force_fetch=False)


def get_catalog_records(domain: str, cat_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """Returns cached list of catalog row dictionaries for template suggestion and fast lookup."""
    if cat_df is None:
        cat_df, _ = get_catalog_and_brands(domain)
    cached = _catalog_records_cache.get(domain)
    if cached is not None and cached[0] is cat_df:
        return cached[1]
    records = cat_df.to_dict("records")
    _catalog_records_cache[domain] = (cat_df, records)
    return records


def get_classifier_dicts(domain: str) -> dict[str, Any]:
    """Load classifier dictionaries (GK, BT, category/region tags) from local cache or PostgreSQL."""
    return DataIngestion.load_classifier_dictionaries(
        config.GOOGLE_SHEET_ID, domain, force_fetch=False
    )


def suggest_tags_from_template(
    sku_name: str,
    domain: str,
    current_bt: str | None = "",
    exclude_bt: str | None = "",
    exclude_gk: str | None = "",
) -> dict[str, Any]:
    """
    Suggests BasicType (BT) and GenericKeywords (GKs) for a new SKU name
    by matching it against a structured template of a base SKU in the catalog,
    swapping the flavor/brand attribute, and snapping to dictionary terms.
    """
    if not sku_name or not sku_name.strip():
        return {"error": "Empty SKU name"}

    sku_name_clean = sku_name.strip()
    clean_input = normalize_sku_text(standardize_sku_units(sku_name_clean))
    if not clean_input:
        return {"error": "Clean SKU name is empty"}

    sku_tokens = [w for w in clean_input.split() if w]

    try:
        cat_df, _brands_df = get_catalog_and_brands(domain)
        classifier = get_classifier(domain)
        try:
            dicts = get_classifier_dicts(domain)
        except Exception as e:
            logger.warning(
                f"Could not load classifier dicts for {domain}: {e}. Falling back to catalog extraction."
            )
            dicts = {}
    except Exception as e:
        logger.error(f"Failed to load resources for template suggestion: {e}")
        return {"error": f"Failed to load catalog/classifier: {str(e)}"}

    ner_engine = getattr(classifier, "ner_engine", None)
    if not ner_engine:
        return {"error": "NER Engine not available on classifier"}

    raw_entities = []
    if domain == config.DOMAIN_FOOD:
        extracted = ner_engine.extract_entities(clean_input)
        for flavor in extracted.get("flavor", set()):
            raw_entities.append((flavor, "flavor", ner_engine))
    else:
        extracted = ner_engine.extract_entities(clean_input)
        for brand in extracted.get("brand", set()):
            raw_entities.append((brand, "brand", ner_engine))

        try:
            food_classifier = get_classifier(config.DOMAIN_FOOD)
            food_ner = getattr(food_classifier, "ner_engine", None)
            if food_ner:
                extracted_food = food_ner.extract_entities(clean_input)
                for flavor in extracted_food.get("flavor", set()):
                    raw_entities.append((flavor, "flavor", food_ner))
        except Exception as e:
            logger.warning(f"Could not load food flavors for market suggestions: {e}")

    entities_to_try = []
    for ent, ent_type, ner_inst in raw_entities:
        ent_clean = str(ent).strip().lower()
        if not ent_clean:
            continue
        ent_words = [w for w in ent_clean.split() if w]
        if len(ent_words) == 0 or len(ent_words) > 2:
            continue
        if len(ent_words) >= len(sku_tokens) or len(ent_clean) >= 0.65 * len(clean_input):
            continue
        pattern = r"\b" + re.escape(ent_clean) + r"\b"
        if not re.search(pattern, clean_input, re.IGNORECASE):
            continue
        entities_to_try.append((ent_clean, ent_type, ner_inst))

    if not entities_to_try:
        return {
            "matched": False,
            "reason": f"No valid brand or flavor modifier entities extracted from SKU: '{clean_input}'",
        }

    def find_best_template_in_records(
        records: list[dict[str, Any]], current_bt_constraint: str = ""
    ):
        best_row = None
        best_ratio = 0.0
        best_base_entity = None
        best_new_entity = None
        best_entity_type = None

        curr_bt_lower = current_bt_constraint.strip().lower() if current_bt_constraint else ""

        for new_entity, entity_type, target_ner in entities_to_try:
            target_anchor = re.sub(
                r"\b" + re.escape(new_entity.lower()) + r"\b", "", clean_input.lower()
            ).strip()
            target_anchor = " ".join(target_anchor.split())
            if not target_anchor or len(target_anchor) < 2:
                continue

            for row in records:
                base_sku = str(row.get("Name", "")).strip()
                if (
                    not base_sku
                    or base_sku.lower() == clean_input.lower()
                    or base_sku.lower() == sku_name_clean.lower()
                ):
                    continue

                base_bt = str(row.get("basictype", "")).strip()
                if curr_bt_lower and base_bt:
                    if (
                        base_bt.lower() != curr_bt_lower
                        and fuzz.ratio(base_bt.lower(), curr_bt_lower) < 70.0
                    ):
                        continue

                base_entities = set()
                if entity_type == "flavor":
                    base_entity_val = row.get("Flavor")
                    if (
                        domain == config.DOMAIN_FOOD
                        and pd.notna(base_entity_val)
                        and str(base_entity_val).strip()
                    ):
                        base_entities.add(str(base_entity_val).strip().lower())
                    else:
                        dict_strong, dict_weak = target_ner._get_dict_entities(base_sku)
                        base_entities.update(dict_strong)
                        base_entities.update(dict_weak)
                else:
                    base_entity_val = row.get("Brand")
                    if (
                        domain == config.DOMAIN_MARKET
                        and pd.notna(base_entity_val)
                        and str(base_entity_val).strip()
                    ):
                        base_entities.add(str(base_entity_val).strip().lower())
                    else:
                        dict_strong, dict_weak = target_ner._get_dict_entities(base_sku)
                        base_entities.update(dict_strong)
                        base_entities.update(dict_weak)

                if not base_entities:
                    continue

                base_sku_words = [w for w in base_sku.split() if w]

                for base_entity in base_entities:
                    base_ent_clean = str(base_entity).strip().lower()
                    if not base_ent_clean or base_ent_clean == new_entity.lower():
                        continue
                    base_ent_words = [w for w in base_ent_clean.split() if w]
                    if len(base_ent_words) > 2:
                        continue
                    if (
                        len(base_ent_words) >= len(base_sku_words)
                        or base_ent_clean == base_sku.lower()
                    ):
                        continue

                    base_anchor = re.sub(
                        r"\b" + re.escape(base_ent_clean) + r"\b", "", base_sku.lower()
                    ).strip()
                    base_anchor = " ".join(base_anchor.split())
                    if not base_anchor or len(base_anchor) < 2:
                        continue

                    anchor_sim = fuzz.ratio(base_anchor, target_anchor)
                    if anchor_sim < 70.0:
                        continue

                    base_entity_pattern = r"\b" + re.escape(base_ent_clean) + r"\b"
                    substituted_name = re.sub(
                        base_entity_pattern, new_entity.lower(), base_sku.lower()
                    )

                    ratio = fuzz.ratio(substituted_name, clean_input.lower())
                    if ratio >= 85.0 and ratio > best_ratio:
                        best_ratio = ratio
                        best_row = row
                        best_base_entity = base_ent_clean
                        best_new_entity = new_entity
                        best_entity_type = entity_type

        return best_row, best_ratio, best_base_entity, best_new_entity, best_entity_type

    catalog_records = get_catalog_records(domain, cat_df=cat_df)
    best_match, _best_match_ratio, matched_base_entity, matched_entity_val, matched_entity_type = (
        find_best_template_in_records(catalog_records, current_bt_constraint=current_bt or "")
    )

    if best_match is None:
        tried_strs = [f"{v} ({t})" for v, t, _ in entities_to_try]
        return {
            "matched": False,
            "reason": f"No template match found in catalog for '{clean_input}' with entities: {', '.join(tried_strs)}",
        }

    base_sku_name = best_match.get("Name")
    base_bt = str(best_match.get("basictype", ""))
    base_gk_str = str(best_match.get("Generic keywords", ""))
    base_gk = [x.strip() for x in base_gk_str.split(",") if x.strip()]

    base_ent_lower = (matched_base_entity or "").lower()
    new_ent_lower = (matched_entity_val or "").lower()

    def swap_entity(text: str) -> str:
        pattern = re.compile(r"\b" + re.escape(base_ent_lower) + r"\b", re.IGNORECASE)
        return pattern.sub(new_ent_lower, text)

    swapped_bt_raw = swap_entity(base_bt) if base_bt else ""
    swapped_gk_raw = [swap_entity(gk) if base_ent_lower in gk.lower() else gk for gk in base_gk]

    gk_dict = dicts.get("gk", [])
    bt_dict = dicts.get("bt", [])

    if not gk_dict and not cat_df.empty and "Generic keywords" in cat_df.columns:
        all_gks = set()
        for val in cat_df["Generic keywords"].dropna():
            for item in str(val).split(","):
                item_clean = item.strip()
                if item_clean:
                    all_gks.add(item_clean)
        gk_dict = list(all_gks)

    if not bt_dict and not cat_df.empty and "basictype" in cat_df.columns:
        bt_dict = list(set(cat_df["basictype"].dropna().astype(str)))
    if not bt_dict and hasattr(classifier, "bt_labels"):
        bt_dict = getattr(classifier, "bt_labels", [])

    gk_dict_map = {g.lower().strip(): g for g in gk_dict}
    bt_dict_map = {b.lower().strip(): b for b in bt_dict}

    def process_and_snap_tag(tag: str, is_bt: bool) -> dict[str, Any]:
        tag_clean = tag.strip()
        tag_lower = tag_clean.lower()
        dict_map = bt_dict_map if is_bt else gk_dict_map
        new_entity_lower = matched_entity_val.lower().strip() if matched_entity_val else ""

        if tag_lower in dict_map:
            return {
                "original": tag_clean,
                "suggested": dict_map[tag_lower],
                "status": "exact_dictionary_match",
            }

        keys_list = list(dict_map.keys())
        if keys_list:
            filtered_keys = (
                [k for k in keys_list if new_entity_lower in k]
                if new_entity_lower and new_entity_lower in tag_lower
                else keys_list
            )
            if filtered_keys:
                best_match_item = process.extractOne(
                    tag_lower, filtered_keys, scorer=fuzz.ratio, score_cutoff=90.0
                )
                if best_match_item:
                    return {
                        "original": tag_clean,
                        "suggested": dict_map[best_match_item[0]],
                        "status": "fuzzy_snapped",
                        "snap_score": best_match_item[1],
                        "snapped_from": tag_clean,
                    }

        return {
            "original": tag_clean,
            "suggested": format_title_case(tag_clean),
            "status": "new_unregistered",
        }

    suggested_bt_info = process_and_snap_tag(swapped_bt_raw, is_bt=True) if swapped_bt_raw else None
    suggested_gks_info_raw = [process_and_snap_tag(gk, is_bt=False) for gk in swapped_gk_raw]

    allow_unregistered = getattr(config, "ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS", True)
    if not allow_unregistered:
        if suggested_bt_info and suggested_bt_info.get("status") == "new_unregistered":
            suggested_bt_info = None
        suggested_gks_info_raw = [
            item for item in suggested_gks_info_raw if item.get("status") != "new_unregistered"
        ]

    seen_gk = set()
    suggested_gks_info = []
    for item in suggested_gks_info_raw:
        sug_lower = item["suggested"].lower().strip()
        if sug_lower not in seen_gk:
            seen_gk.add(sug_lower)
            suggested_gks_info.append(item)

    final_bt = suggested_bt_info["suggested"] if suggested_bt_info else ""
    final_gk = [item["suggested"] for item in suggested_gks_info]

    if not final_bt and not final_gk:
        return {"matched": False, "reason": "No new template-based tag suggestions were generated."}

    return {
        "matched": True,
        "base_sku": base_sku_name,
        "base_entity": matched_base_entity,
        "new_entity": matched_entity_val,
        "entity_type": matched_entity_type,
        "suggested_bt": final_bt,
        "suggested_gk": final_gk,
        "bt_info": suggested_bt_info,
        "gk_info": suggested_gks_info,
    }


def resolve_template_enrichment_for_sku(
    sku_name: str,
    domain: str,
    curr_bt: str,
    curr_gk: str,
    status_to_check: str,
    suggest_fn=None,
) -> tuple[str, str, bool, str, str, dict[str, Any]]:
    """Shared template enrichment resolver for batch processing and single-SKU audits."""
    if not getattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", True):
        return (
            curr_bt,
            curr_gk,
            False,
            "",
            "",
            {
                "matched_template": False,
                "note": "Template tag enrichment is disabled in configuration",
            },
        )
    if status_to_check in ["Exact Text Match", "High Confidence", "HIGH", "AUTO"]:
        return (
            curr_bt,
            curr_gk,
            False,
            "",
            "",
            {
                "matched_template": False,
                "note": f"Skipped template enrichment because status was {status_to_check}",
            },
        )
    try:
        fn = suggest_fn or suggest_tags_from_template
        sug_res = fn(sku_name, domain=domain, current_bt=curr_bt)
        if not sug_res.get("matched"):
            return (
                curr_bt,
                curr_gk,
                False,
                "",
                "",
                {
                    "matched_template": False,
                    "note": sug_res.get("reason", "No template match found in catalog"),
                },
            )

        s_bt = sug_res.get("suggested_bt", "")
        s_gk_list = sug_res.get("suggested_gk", [])
        s_gk = ", ".join(s_gk_list) if isinstance(s_gk_list, list) else str(s_gk_list)

        new_bt = s_bt or curr_bt
        new_gk = curr_gk
        if s_gk:
            existing_gks = [x.strip() for x in str(curr_gk).split(",") if x.strip()]
            template_gks = (
                s_gk_list
                if isinstance(s_gk_list, list)
                else [x.strip() for x in s_gk.split(",") if x.strip()]
            )
            new_gk = ", ".join(list(dict.fromkeys(existing_gks + template_gks)))

        details = {
            "matched_template": True,
            "base_sku": sug_res.get("base_sku", ""),
            "template_bt": s_bt,
            "template_gk": s_gk,
            "notes": f"Template tag enrichment applied from base SKU '{sug_res.get('base_sku', '')}'",
        }
        return new_bt, new_gk, True, s_bt, s_gk, details
    except Exception as e:
        return curr_bt, curr_gk, False, "", "", {"matched_template": False, "error": str(e)}


def apply_template_tag_enrichment(
    skus: list[dict[str, Any]],
    results: list[dict[str, Any]],
    domain: str,
    mode: str = "classifier",
    suggest_fn=None,
) -> None:
    """Enriches predicted tags using template-based keyword substitution if enabled."""
    if not getattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", True):
        return

    try:
        for i, sku in enumerate(skus):
            if i >= len(results):
                break
            status_val = results[i].get("status", "") or results[i].get("bt_status", "")
            if status_val in ["Exact Text Match", "High Confidence", "HIGH", "AUTO"] or (
                results[i].get("bt_status", "")
                in ["Exact Text Match", "High Confidence", "HIGH", "AUTO"]
            ):
                continue

            new_bt, new_gk, matched, s_bt, s_gk, _ = resolve_template_enrichment_for_sku(
                sku.get("name", ""),
                domain,
                str(results[i].get("suggested_bt") or ""),
                str(results[i].get("suggested_gk") or ""),
                status_val,
                suggest_fn=suggest_fn,
            )
            if not matched:
                continue

            if s_bt:
                results[i]["suggested_bt"] = new_bt
                if mode == "classifier":
                    results[i]["bt_status"] = "AUTO"
                    results[i]["bt_confidence"] = max(results[i].get("bt_confidence") or 0.0, 0.95)
                elif mode == "pipeline":
                    results[i]["bt_status"] = "High Confidence"
                    results[i]["bt_confidence"] = max(results[i].get("bt_confidence") or 0.0, 0.95)
            if s_gk:
                results[i]["suggested_gk"] = new_gk
                if mode == "classifier":
                    results[i]["gk_status"] = "AUTO"
                    results[i]["gk_confidence"] = max(results[i].get("gk_confidence") or 0.0, 0.95)
                elif mode == "pipeline":
                    results[i]["gk_status"] = "High Confidence"
                    results[i]["gk_confidence"] = max(results[i].get("gk_confidence") or 0.0, 0.95)
    except Exception as e:
        logger.warning(f"Template tag enrichment failed for {mode}: {e}")
