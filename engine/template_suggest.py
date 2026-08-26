"""
SKU MatchOps Engine - Template-Aware Tag Suggestion
Extracts entity variations (brand, flavor) and matches against structural catalog templates.
"""

import re
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from rapidfuzz import fuzz, process

from engine import config
from engine.data_pipeline.ingestion import DataIngestion
from engine.nlp.text_cleaner import TextPipeline
from engine.resource_loader import _pipelines, get_pipeline, get_classifier

logger = logging.getLogger("matchops.engine.template_suggest")

_catalog_records_cache: Dict[tuple, List[Dict[str, Any]]] = {}

def get_catalog_and_brands(domain: str):
    """Load catalog and brands/flavors from local Feather cache if possible, else fetch from sheets."""
    return DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, domain)

def get_catalog_records(domain: str, cat_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """Returns cached list of catalog row dictionaries for template suggestion and fast lookup."""
    if cat_df is None:
        cat_df, _ = get_catalog_and_brands(domain)
    cache_key = (id(cat_df), domain)
    if cache_key in _catalog_records_cache:
        return _catalog_records_cache[cache_key]
    records = cat_df.to_dict('records')
    _catalog_records_cache[cache_key] = records
    return records

def get_classifier_dicts(domain: str) -> Dict[str, Any]:
    """Load classifier dictionaries (GK, BT, category/region tags)."""
    return DataIngestion.load_classifier_dictionaries(config.GOOGLE_SHEET_ID, domain)


def title_case_with_exceptions(text: str) -> str:
    """
    Convert string to title case but keep minor prepositions/conjunctions lowercase,
    unless they are the first word.
    """
    if not text:
        return ""
    
    minor_words = {
        "with", "of", "and", "in", "the", "for", "to", "at", "by", "on", 
        "a", "an", "or", "as", "but", "nor", "yet", "so"
    }
    
    words = text.split()
    capitalized_words = []
    
    for idx, word in enumerate(words):
        clean_word = re.sub(r'^\W+|\W+$', '', word).lower()
        if idx == 0:
            if clean_word in minor_words:
                prefix = re.match(r'^\W*', word).group(0)
                suffix = re.search(r'\W*$', word).group(0)
                capitalized_words.append(prefix + clean_word.capitalize() + suffix)
            else:
                parts = word.split('-')
                capitalized_parts = [p.capitalize() for p in parts]
                capitalized_words.append('-'.join(capitalized_parts))
        elif clean_word in minor_words:
            prefix = re.match(r'^\W*', word).group(0)
            suffix = re.search(r'\W*$', word).group(0)
            capitalized_words.append(prefix + clean_word + suffix)
        else:
            parts = word.split('-')
            capitalized_parts = [p.capitalize() for p in parts]
            capitalized_words.append('-'.join(capitalized_parts))
            
    return " ".join(capitalized_words)


def suggest_tags_from_template(
    sku_name: str,
    domain: str,
    current_bt: str | None = "",
    exclude_bt: str | None = "",
    exclude_gk: str | None = ""
) -> Dict[str, Any]:
    """
    Suggests BasicType (BT) and GenericKeywords (GKs) for a new SKU name
    by matching it against a structured template of a base SKU in the catalog,
    swapping the flavor/brand attribute, and snapping to dictionary terms.
    """
    if not sku_name or not sku_name.strip():
        return {"error": "Empty SKU name"}
        
    sku_name_clean = sku_name.strip()
    clean_input = TextPipeline.normalize_final(TextPipeline.standardize_units(sku_name_clean))
    if not clean_input:
        return {"error": "Clean SKU name is empty"}
        
    sku_tokens = [w for w in clean_input.split() if w]
    
    # 1. Load catalog and dictionaries
    try:
        cat_df, brands_df = get_catalog_and_brands(domain)
        classifier = get_classifier(domain)
        try:
            dicts = get_classifier_dicts(domain)
        except Exception as e:
            logger.warning(f"Could not load classifier dicts for {domain}: {e}. Falling back to catalog extraction.")
            dicts = {}
    except Exception as e:
        logger.error(f"Failed to load resources for template suggestion: {e}")
        return {"error": f"Failed to load catalog/classifier: {str(e)}"}
        
    ner_engine = getattr(classifier, 'ner_engine', None)
    if not ner_engine:
        return {"error": "NER Engine not available on classifier"}
        
    # 2. Extract brand/flavor entities from clean input
    raw_entities = []
    
    if domain == config.DOMAIN_FOOD:
        extracted = ner_engine.extract_entities(clean_input)
        flavors = extracted.get("flavor", set())
        for flavor in flavors:
            raw_entities.append((flavor, "flavor", ner_engine))
    else: # market domain
        extracted = ner_engine.extract_entities(clean_input)
        brands = extracted.get("brand", set())
        for brand in brands:
            raw_entities.append((brand, "brand", ner_engine))
            
        # For market domain, also scan for flavors from the food domain list
        try:
            food_classifier = get_classifier(config.DOMAIN_FOOD)
            food_ner = getattr(food_classifier, 'ner_engine', None)
            if food_ner:
                extracted_food = food_ner.extract_entities(clean_input)
                flavors = extracted_food.get("flavor", set())
                for flavor in flavors:
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
            "reason": f"No valid brand or flavor modifier entities extracted from SKU: '{clean_input}'"
        }
        
    # Helper to evaluate candidate records against entities
    def find_best_template_in_records(records: List[Dict[str, Any]], current_bt_constraint: str = ""):
        best_row = None
        best_ratio = 0.0
        best_base_entity = None
        best_new_entity = None
        best_entity_type = None
        
        curr_bt_lower = current_bt_constraint.strip().lower() if current_bt_constraint else ""

        for new_entity, entity_type, target_ner in entities_to_try:
            target_anchor = re.sub(r"\b" + re.escape(new_entity.lower()) + r"\b", "", clean_input.lower()).strip()
            target_anchor = " ".join(target_anchor.split())
            if not target_anchor or len(target_anchor) < 2:
                continue

            for row in records:
                base_sku = str(row.get("Name", "")).strip()
                if not base_sku or base_sku.lower() == clean_input.lower() or base_sku.lower() == sku_name_clean.lower():
                    continue
                    
                base_bt = str(row.get("basictype", "")).strip()
                if curr_bt_lower and base_bt:
                    if base_bt.lower() != curr_bt_lower and fuzz.ratio(base_bt.lower(), curr_bt_lower) < 70.0:
                        continue
                    
                base_entities = set()
                if entity_type == "flavor":
                    base_entity_val = row.get("Flavor")
                    if domain == config.DOMAIN_FOOD and pd.notna(base_entity_val) and str(base_entity_val).strip():
                        base_entities.add(str(base_entity_val).strip().lower())
                    else:
                        dict_strong, dict_weak = target_ner._get_dict_entities(base_sku)
                        base_entities.update(dict_strong)
                        base_entities.update(dict_weak)
                else:
                    base_entity_val = row.get("Brand")
                    if domain == config.DOMAIN_MARKET and pd.notna(base_entity_val) and str(base_entity_val).strip():
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
                    if len(base_ent_words) >= len(base_sku_words) or base_ent_clean == base_sku.lower():
                        continue
                        
                    base_anchor = re.sub(r"\b" + re.escape(base_ent_clean) + r"\b", "", base_sku.lower()).strip()
                    base_anchor = " ".join(base_anchor.split())
                    if not base_anchor or len(base_anchor) < 2:
                        continue
                        
                    anchor_sim = fuzz.ratio(base_anchor, target_anchor)
                    if anchor_sim < 70.0:
                        continue
                        
                    base_entity_pattern = r"\b" + re.escape(base_ent_clean) + r"\b"
                    substituted_name = re.sub(base_entity_pattern, new_entity.lower(), base_sku.lower())
                    
                    ratio = fuzz.ratio(substituted_name, clean_input.lower())
                    if ratio >= 85.0 and ratio > best_ratio:
                        best_ratio = ratio
                        best_row = row
                        best_base_entity = base_ent_clean
                        best_new_entity = new_entity
                        best_entity_type = entity_type
                        
        return best_row, best_ratio, best_base_entity, best_new_entity, best_entity_type

    best_match = None
    best_match_ratio = 0.0
    matched_base_entity = None
    matched_entity_val = None
    matched_entity_type = None

    # Scan cached in-memory catalog records (<1ms)
    catalog_records = get_catalog_records(domain, cat_df=cat_df)
    best_match, best_match_ratio, matched_base_entity, matched_entity_val, matched_entity_type = find_best_template_in_records(
        catalog_records, current_bt_constraint=current_bt or ""
    )
                    
    if best_match is None:
        tried_strs = [f"{v} ({t})" for v, t, _ in entities_to_try]
        return {
            "matched": False,
            "reason": f"No template match found in catalog for '{clean_input}' with entities: {', '.join(tried_strs)}"
        }
        
    base_sku_name = best_match.get("Name")
    base_bt = str(best_match.get("basictype", ""))
    base_gk_str = str(best_match.get("Generic keywords", ""))
    base_gk = [x.strip() for x in base_gk_str.split(",") if x.strip()]
    
    base_ent_lower = matched_base_entity.lower()
    new_ent_lower = matched_entity_val.lower()
    
    def swap_entity(text: str) -> str:
        pattern = re.compile(r"\b" + re.escape(base_ent_lower) + r"\b", re.IGNORECASE)
        return pattern.sub(new_ent_lower, text)
        
    swapped_bt_raw = swap_entity(base_bt) if base_bt else ""
    
    swapped_gk_raw = []
    for gk in base_gk:
        if base_ent_lower in gk.lower():
            swapped_gk_raw.append(swap_entity(gk))
        else:
            swapped_gk_raw.append(gk)

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
    if not bt_dict and hasattr(classifier, 'bt_labels'):
        bt_dict = getattr(classifier, 'bt_labels', [])

    gk_dict_map = {g.lower().strip(): g for g in gk_dict}
    bt_dict_map = {b.lower().strip(): b for b in bt_dict}
    
    def process_and_snap_tag(tag: str, is_bt: bool) -> Dict[str, Any]:
        tag_clean = tag.strip()
        tag_lower = tag_clean.lower()
        dict_map = bt_dict_map if is_bt else gk_dict_map
        new_entity_lower = matched_entity_val.lower().strip() if matched_entity_val else ""
        
        if tag_lower in dict_map:
            return {
                "original": tag_clean,
                "suggested": dict_map[tag_lower],
                "status": "exact_dictionary_match"
            }
            
        keys_list = list(dict_map.keys())
        if keys_list:
            filtered_keys = keys_list
            if new_entity_lower and new_entity_lower in tag_lower:
                filtered_keys = [k for k in keys_list if new_entity_lower in k]
                
            if filtered_keys:
                best_match_item = process.extractOne(tag_lower, filtered_keys, scorer=fuzz.ratio, score_cutoff=90.0)
                if best_match_item:
                    best_match_key = best_match_item[0]
                    best_score = best_match_item[1]
                    return {
                        "original": tag_clean,
                        "suggested": dict_map[best_match_key],
                        "status": "fuzzy_snapped",
                        "snap_score": best_score,
                        "snapped_from": tag_clean
                    }
                
        formatted_tag = title_case_with_exceptions(tag_clean)
        return {
            "original": tag_clean,
            "suggested": formatted_tag,
            "status": "new_unregistered"
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
        return {
            "matched": False,
            "reason": "No new template-based tag suggestions were generated."
        }
    
    return {
        "matched": True,
        "base_sku": base_sku_name,
        "base_entity": matched_base_entity,
        "new_entity": matched_entity_val,
        "entity_type": matched_entity_type,
        "suggested_bt": final_bt,
        "suggested_gk": final_gk,
        "bt_info": suggested_bt_info,
        "gk_info": suggested_gks_info
    }
