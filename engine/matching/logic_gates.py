import math
import re
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from engine import config
from engine.nlp.text_cleaner import TextPipeline
from engine.utils.flavor_utils import build_food_flavors_info

class LogicGates:
    """Engine for domain-specific matching logic and validation gates."""

    def __init__(self, embed_engine, brands_df=None):
        self.embedder = embed_engine
        self.brands_df = brands_df
        self.food_flavors_dict, _, _, _, _ = build_food_flavors_info(brands_df)
        if self.food_flavors_dict:
            sorted_terms = sorted(self.food_flavors_dict.keys(), key=len, reverse=True)
            self._flavor_pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in sorted_terms) + r")(?![a-z0-9])")
        else:
            self._flavor_pattern = None

    def _try_template_swap(
        self,
        input_clean: str,
        cat_clean: str,
        input_entities: Dict[str, Set[str]],
        cat_entities: Dict[str, Set[str]],
        domain: str
    ) -> Tuple[bool, List[str], List[Tuple[str, str]], float]:
        """
        Attempts to swap brand and/or flavor entities from the input SKU into the catalog SKU
        to see if they become a template match (fuzzy ratio >= 90%).
        Returns: (success, swaps_performed, replacements, ratio)
        """
        swap_types = ["flavor"] if domain == config.DOMAIN_FOOD else ["brand", "flavor"]
        substituted = cat_clean.lower()
        swaps_performed = []
        replacements = []

        for stype in swap_types:
            in_set = {x.lower() for x in input_entities.get(stype, set()) if x}
            cat_set = {x.lower() for x in cat_entities.get(stype, set()) if x}

            if in_set and cat_set and in_set.isdisjoint(cat_set):
                for c_ent in cat_set:
                    pattern = r"\b" + re.escape(c_ent) + r"\b"
                    if re.search(pattern, substituted):
                        for i_ent in in_set:
                            substituted = re.sub(pattern, i_ent, substituted)
                            swaps_performed.append(f"{c_ent}->{i_ent} ({stype})")
                            replacements.append((c_ent, i_ent))
                            break

        if swaps_performed:
            ratio = fuzz.partial_ratio(substituted, input_clean.lower())
            if ratio >= 90.0:
                return True, swaps_performed, replacements, ratio

        return False, [], [], 0.0

    def _apply_swaps_to_row(self, match_row: Dict, replacements: List[Tuple[str, str]]):
        """Swaps the resolved brand/flavor values in-place in the catalog match row payload."""
        for old_val, new_val in replacements:
            pattern = re.compile(r"\b" + re.escape(old_val) + r"\b", re.IGNORECASE)

            def swap_str(text: str) -> str:
                if not text or not isinstance(text, str):
                    return text

                def replace_match(match):
                    matched_text = match.group(0)
                    if matched_text.istitle():
                        return new_val.capitalize()
                    elif matched_text.isupper():
                        return new_val.upper()
                    else:
                        return new_val

                return pattern.sub(replace_match, text)

            if "Name" in match_row:
                match_row["Name"] = swap_str(match_row["Name"])
            if "clean_text" in match_row:
                match_row["clean_text"] = swap_str(match_row["clean_text"])

            for bt_key in ["basictype", "BasicType"]:
                if bt_key in match_row:
                    match_row[bt_key] = swap_str(match_row[bt_key])

            for gk_key in ["Generic keywords", "GenericKeywords"]:
                if gk_key in match_row:
                    match_row[gk_key] = swap_str(match_row[gk_key])



    # ─────────────────────────────────────────────────────────────
    # Market Logic Helpers
    # ─────────────────────────────────────────────────────────────


    def compare_entity_sets(self, set1: Set[str], set2: Set[str], raw_text1: str, raw_text2: str, entity_type: str = "") -> str:
        """
        3-Tiered entity comparison:
        1. Cross-Pollination (Checks if entity exists in opposing raw text).
        2. Subset/Exact match.
        3. Fuzzy typo check for longer words.
        Returns: "Match", "Neutral", or "Conflict".
        """
        if not set1 and not set2:
            return "Neutral"

        rt1, rt2 = raw_text1.lower(), raw_text2.lower()

        # Tier 1: Cross-Pollination (Only run if one of the sets is empty to prevent false positives when both sides have brand entities)
        if not set1 or not set2:
            if set1:
                for v1 in set1:
                    if v1 in rt2:
                        return "Match"
                    if len(v1) >= 5 and fuzz.partial_ratio(v1, rt2) >= 80:
                        return "Match"

            if set2:
                for v2 in set2:
                    if v2 in rt1:
                        return "Match"
                    if len(v2) >= 5 and fuzz.partial_ratio(v2, rt1) >= 80:
                        return "Match"

        # Tier 2 & 3: Subset & Fuzzy
        for v1 in set1:
            for v2 in set2:
                if v1 in v2 or v2 in v1:
                    return "Match"
                if len(v1) >= 5 and len(v2) >= 5:
                    if fuzz.ratio(v1, v2) >= 80:
                        return "Match"

        return "Conflict"

    # ─────────────────────────────────────────────────────────────
    # Food Logic Helpers
    # ─────────────────────────────────────────────────────────────

    def _resolve_flavors(self, flavor_set: Set[str]) -> Set[str]:
        """Returns the canonical flavor names for any flavors in the set.

        Uses the dynamically loaded food_flavors_dict. Multi-word flavor strings
        are checked for embedded flavor terms via substring matching.
        """
        result: Set[str] = set()
        for f in flavor_set:
            f_lower = f.lower()
            # Direct dict lookup (most common case: exact canonical name)
            if f_lower in self.food_flavors_dict:
                result.add(self.food_flavors_dict[f_lower])
            elif self._flavor_pattern:
                match = self._flavor_pattern.search(f_lower)
                if match:
                    canonical = self.food_flavors_dict.get(match.group(1))
                    if canonical:
                        result.add(canonical)
        return result

    def apply_food_logic_gates(
        self, input_clean: str, input_entities: Dict[str, Set[str]], match_row: Dict, raw_ai_score: float, input_price: float,
        input_description: str = ""
    ) -> Tuple[float, str, str]:
        """Logic gates specific to the food domain."""
        score = float(raw_ai_score)
        reasons = []
        cat_clean = match_row["clean_text"]

        # 1. Fuzzy Bypass
        input_no_weights = TextPipeline.strip_weights(input_clean)
        cat_no_weights = TextPipeline.strip_weights(cat_clean)
        token_ratio = fuzz.token_sort_ratio(input_no_weights, cat_no_weights)
        is_fuzzy_bypass = token_ratio >= 90

        if is_fuzzy_bypass:
            score = max(score, 100.0)
            reasons.append(f"Whole-SKU Fuzzy Match ({round(token_ratio)}%)")

        # 2. Flavor Conflict Gate
        cat_entities = match_row.get("entities", {})
        input_flavors = input_entities.get("flavor", set())
        catalog_flavors = cat_entities.get("flavor", set())

        # Identify flavors from each side using FOOD_FLAVORS_DICT.
        input_flavors_resolved  = self._resolve_flavors(input_flavors)
        catalog_flavors_resolved = self._resolve_flavors(catalog_flavors)

        # Fallback to scanning raw clean text if entities did not yield resolved flavors
        if not input_flavors_resolved and input_clean:
            input_flavors_resolved = self._resolve_flavors({input_clean})
        if not catalog_flavors_resolved and match_row.get("clean_text"):
            catalog_flavors_resolved = self._resolve_flavors({match_row["clean_text"]})

        if input_flavors_resolved and catalog_flavors_resolved:
            if input_flavors_resolved.isdisjoint(catalog_flavors_resolved):
                return -10.0, "Rejected", (
                    f"Flavor Conflict: {sorted(input_flavors_resolved)} vs {sorted(catalog_flavors_resolved)}"
                )
        elif catalog_flavors_resolved and not input_flavors_resolved:
            is_fuzzy_bypass = False
            score = min(score, 3.5)
            reasons.append(f"Flavor Mismatch: catalog has {sorted(catalog_flavors_resolved)}, input unspecified")
        elif input_flavors_resolved and not catalog_flavors_resolved:
            score -= 2.0
            reasons.append(f"Flavor Mismatch: input has {sorted(input_flavors_resolved)}, catalog unspecified")


        # Determine status
        status = "High Confidence" if score >= config.CONFIDENCE_THRESHOLD_HIGH else ("Medium Confidence" if score > 0 else "Low / Rejected")
        
        # Map score to probability fraction (0.0 to 1.0)
        if score == -10.0:
            prob_score = 0.0
        elif is_fuzzy_bypass or score >= 90.0:
            prob_score = min(1.0, score / 100.0)
        else:
            p = 1 / (1 + math.exp(-0.55 * score))
            prob_score = float(round(p, 4))
            
        return prob_score, status, "; ".join(reasons)

    # ─────────────────────────────────────────────────────────────
    # Unified Entry Point
    # ─────────────────────────────────────────────────────────────

    def apply_logic_gates(
        self,
        input_clean: str,
        input_entities: Dict[str, Set[str]],
        match_row: Dict,
        raw_ai_score: float,
        input_price: float,
        input_w_data: Tuple,
        input_no_weights: str,
        domain: str = config.DOMAIN_MARKET,
        input_description: str = "",
        input_category: str = "",
        predicted_bt: str = "",
    ) -> Tuple[float, str, str]:
        """Main entry point to apply logic gates based on domain."""
        if domain == config.DOMAIN_FOOD:
            return self.apply_food_logic_gates(input_clean, input_entities, match_row, raw_ai_score, input_price, input_description)

        # --- Market Logic ---
        score = float(raw_ai_score)
        reasons = []
        cat_entities = match_row["entities"]
        catalog_w_data = match_row["weight_val"]
        cat_clean = match_row["clean_text"]
        cat_no_weights = match_row.get("clean_no_weights") or TextPipeline.strip_weights(cat_clean)

        # 1. Whole-SKU Fuzzy Bypass
        token_ratio = fuzz.token_sort_ratio(input_no_weights, cat_no_weights)
        is_fuzzy_bypass = token_ratio >= 90
        
        if not is_fuzzy_bypass:
            in_words = input_no_weights.split()
            cat_words = cat_no_weights.split()
            aligned_in = []
            for iw in in_words:
                matched_cw = iw
                for cw in cat_words:
                    if len(iw) >= 5 and len(cw) >= 5 and fuzz.ratio(iw, cw) >= 80:
                        matched_cw = cw
                        break
                aligned_in.append(matched_cw)
            aligned_input_str = " ".join(aligned_in)
            typo_ratio = fuzz.token_sort_ratio(aligned_input_str, cat_no_weights)
            if typo_ratio >= 90:
                is_fuzzy_bypass = True
                token_ratio = typo_ratio

        if is_fuzzy_bypass:
            score = max(score, 100.0)
            reasons.append(f"Whole-SKU Fuzzy Match ({round(token_ratio)}%)")

        # 2. Physical Form Gate
        if input_w_data and input_w_data[0] is not None and catalog_w_data and catalog_w_data[0] is not None:
            _, _, in_type = input_w_data
            _, _, cat_type = catalog_w_data
            if in_type != cat_type:
                return -10.0, "Rejected", f"Physical Form Conflict: {in_type} vs {cat_type}"

        # 3. Ice Cream Packaging Gate
        is_ic_input = config.IC_TRIGGER_KEYWORD in input_clean
        cat_g_keywords = str(match_row.get("Generic keywords", match_row.get("GenericKeywords", ""))).lower()
        cat_categories = str(match_row.get("Categories", "")).lower()
        is_ic_catalog = any(config.IC_TRIGGER_KEYWORD in s for s in [cat_clean, cat_g_keywords, cat_categories])

        if (is_ic_input or is_ic_catalog) and input_w_data[0] is not None and catalog_w_data[0] is not None:
            ic_type_input = TextPipeline.get_ice_cream_type(str(input_w_data[0]))
            ic_type_catalog = TextPipeline.get_ice_cream_type(str(catalog_w_data[0]))
            if ic_type_input and ic_type_catalog and ic_type_input != ic_type_catalog:
                return -10.0, "Rejected", f"Ice Cream Packaging Mismatch ({ic_type_input} vs {ic_type_catalog})"

        # 4. Entity Validation (Skipped if fuzzy bypass triggered)
        if not is_fuzzy_bypass:
            # Brand Check
            brand_check = self.compare_entity_sets(input_entities["brand"], cat_entities["brand"], input_clean, cat_clean, "brand")
            # Flavor Check
            flavor_check = self.compare_entity_sets(input_entities.get("flavor", set()), cat_entities.get("flavor", set()), input_clean, cat_clean, "flavor")

            has_brand_conflict = (brand_check == "Conflict")
            has_flavor_conflict = (flavor_check == "Conflict")

            if has_brand_conflict:
                return -10.0, "Rejected", f"Brand Conflict: {input_entities['brand']} vs {cat_entities['brand']}"

            if has_flavor_conflict:
                return -10.0, "Rejected", f"Flavor Conflict: {input_entities.get('flavor')} vs {cat_entities.get('flavor')}"

            # Penalties for other attributes (variant, scent, active ingredient) still run
            for attr in ["variant", "scent", "active ingredient"]:
                attr_check = self.compare_entity_sets(input_entities.get(attr, set()), cat_entities.get(attr, set()), input_clean, cat_clean, attr)
                if attr_check == "Conflict":
                    score -= 3.0
                    reasons.append(f"{attr.capitalize()} Penalty: {input_entities.get(attr)} vs {cat_entities.get(attr)}")


        # 5. Weight Normalization Boost/Penalty
        weight_matched = False
        if input_w_data[0] is not None and catalog_w_data[0] is not None:
            in_val, cat_val = input_w_data[0], catalog_w_data[0]
            max_val = max(in_val, cat_val)
            if max_val == 0:
                score += 2.0
                reasons.append("Weight Match (0)")
                weight_matched = True
            else:
                diff_pct = abs(in_val - cat_val) / max_val * 100
                if diff_pct < 1.0:
                    score += 2.0
                    reasons.append(f"Weight Match ({int(in_val)})")
                    weight_matched = True
                elif diff_pct > 10.0:
                    reasons.append(f"Weight Mismatch ({int(in_val)} vs {int(cat_val)})")

        # 6. Price Validation
        if score < 6.0 and weight_matched:
            try:
                p_input = float(input_price)
                p_catalog = float(match_row.get("Price", 0))
                if p_input > 0 and p_catalog > 0:
                    if abs(p_input - p_catalog) <= 75:
                        score += 3.0
                        reasons.append(f"Price Match: {p_input} vs {p_catalog}")
                    else:
                        score -= 2.0
                        reasons.append(f"Price Mismatch: {p_input} vs {p_catalog}")
            except Exception:
                pass

        # 7. Category Match Bonus (+1)
        # If the input SKU has a category and it matches the catalog's Categories field, boost the score.
        in_cat = str(input_category).strip().lower() if input_category else ""
        if in_cat and in_cat not in ("", "nan"):
            catalog_cat = str(match_row.get("Categories", "")).strip().lower()
            if catalog_cat and in_cat in catalog_cat:
                score += 1.0
                reasons.append("Category Match Bonus")

        # 8. Predicted BT Alignment Boost
        if predicted_bt and not is_fuzzy_bypass:
            cand_bt = str(match_row.get("basictype", match_row.get("BasicType", ""))).strip().lower()
            if cand_bt and predicted_bt.strip().lower() == cand_bt:
                in_b = input_entities.get("brand", set()) if isinstance(input_entities, dict) else set()
                cat_b = cat_entities.get("brand", set()) if isinstance(cat_entities, dict) else set()
                in_f = input_entities.get("flavor", set()) if isinstance(input_entities, dict) else set()
                cat_f = cat_entities.get("flavor", set()) if isinstance(cat_entities, dict) else set()
                b_check = self.compare_entity_sets(in_b, cat_b, input_clean, cat_clean, "brand")
                f_check = self.compare_entity_sets(in_f, cat_f, input_clean, cat_clean, "flavor")
                if b_check != "Conflict" and f_check != "Conflict":
                    score += 3.5
                    reasons.append(f"Predicted BT Alignment Boost (+3.5: {predicted_bt})")

        # Determine status
        status = "High Confidence" if score >= config.CONFIDENCE_THRESHOLD_HIGH else ("Medium Confidence" if score > 0 else "Low / Rejected")
        
        # Map score to probability fraction (0.0 to 1.0)
        if score == -10.0:
            prob_score = 0.0
        elif is_fuzzy_bypass or score >= 90.0:
            prob_score = min(1.0, score / 100.0)
        else:
            p = 1 / (1 + math.exp(-0.55 * score))
            prob_score = float(round(p, 4))
            
        return prob_score, status, "; ".join(reasons)