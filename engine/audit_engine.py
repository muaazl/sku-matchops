import os
import sys
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from engine import config
from engine.nlp.text_cleaner import TextPipeline
from engine.rules_engine import run_rules_engine
from engine.classification.tagger import tag_all_skus

def run_sku_audit(
    sku_name: str,
    domain: str = "market",
    task: str = "pipeline",
    price: float = 0.0,
    description: str = "",
    category: str = "",
    pipeline=None,
    classifier=None,
    suggest_fn=None,
) -> dict:
    """
    Executes a complete, transparent, multi-stage diagnostic audit for a SKU item.
    Guarantees 100% telemetry alignment with the production backend pipeline.
    """
    domain = domain.lower()
    task = task.lower()
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        domain = config.DOMAIN_MARKET
    if task not in ("pipeline", "matcher", "classifier"):
        task = "pipeline"

    audit_data = {
        "input": {
            "sku_name": sku_name,
            "domain": domain,
            "task": task,
            "price": price,
            "description": description,
            "category": category
        },
        "stage1_nlp": {},
        "stage2_candidate_retrieval": {},
        "stage3_cross_encoder": {},
        "stage4_logic_gates": [],
        "stage5_matcher_result": {},
        "stage6_escalation": {},
        "stage7_template_enrichment": {},
        "stage8_rules_engine": {},
        "final_output": {}
    }

    # -------------------------------------------------------------
    # Stage 1: NLP Pre-Processing & Food Domain Flavor Enrichment
    # -------------------------------------------------------------
    effective_sku_name = sku_name
    food_suffix_added = ""
    
    clean_input = TextPipeline.normalize_final(TextPipeline.standardize_units(sku_name))
    input_no_weights = TextPipeline.strip_weights(clean_input)
    input_w_data = TextPipeline.extract_weight_feature(clean_input)
    
    if pipeline is None and task != "classifier":
        try:
            from engine.resource_loader import get_pipeline
            pipeline = get_pipeline(domain)
        except Exception:
            pipeline = None
    
    full_ner_input = sku_name
    if description and str(description).lower() not in ("nan", "none", "<na>"):
        full_ner_input += f" {description}"
    if category and str(category).lower() not in ("nan", "none", "<na>"):
        full_ner_input += f" {category}"
        
    ner_text = TextPipeline.prep_for_ner(full_ner_input)
    extracted_entities = pipeline.ner.extract_entities(ner_text) if pipeline and hasattr(pipeline, "ner") else {}
    
    # Food domain flavor suffix modification (System pipeline step in processor.py)
    if domain == config.DOMAIN_FOOD:
        ner_engine = getattr(pipeline, "ner", None)
        if ner_engine is None:
            try:
                from engine.resource_loader import _get_shared_models
                _, ner_engine = _get_shared_models()
            except Exception:
                ner_engine = None
        if ner_engine:
            flavor_set = extracted_entities.get("flavor", set())
            suffixes = []
            if any(f in getattr(ner_engine, 'vegetable_flavors', set()) for f in flavor_set):
                suffixes.append("veg")
            if any(f in getattr(ner_engine, 'seafood_flavors', set()) for f in flavor_set):
                suffixes.append("seafood")
            meat_count = sum(1 for f in flavor_set if f in getattr(ner_engine, 'meat_flavors', set()) and f != "egg")
            if meat_count >= 2:
                suffixes.append("mixed")
            if suffixes:
                food_suffix_added = " ".join(suffixes)
                effective_sku_name = f"{sku_name} {food_suffix_added}"

    audit_data["stage1_nlp"] = {
        "raw_sku": sku_name,
        "clean_input": clean_input,
        "weight_stripped_input": input_no_weights,
        "extracted_weights": {
            "value": input_w_data[0],
            "unit": input_w_data[1],
            "physical_form": input_w_data[2]
        },
        "ner_entities": {k: list(v) if isinstance(v, (set, list)) else v for k, v in extracted_entities.items()},
        "food_domain_effective_sku": effective_sku_name,
        "food_suffix_added": food_suffix_added
    }

    # If task is classifier-only, jump to Classifier processing path
    if task == "classifier":
        _audit_classifier_pipeline(audit_data, domain, sku_name, description, category, price, classifier=classifier)
        return audit_data

    # -------------------------------------------------------------
    # Stage 1.5: Exact Catalog Lookup & AI Bypass Check
    # -------------------------------------------------------------
    exact_bypass_row = None
    exact_bypass_reason = ""
    
    if hasattr(pipeline, "exact_match_map") and pipeline.exact_match_map:
        cat_idx = pipeline.exact_match_map.get(clean_input)
        if cat_idx is not None:
            exact_bypass_row = pipeline.raw_catalog.iloc[cat_idx]
            exact_bypass_reason = "Exact Text Match"
            
    if exact_bypass_row is None and hasattr(pipeline, "token_sorted_map") and pipeline.token_sorted_map:
        sorted_input_tokens = " ".join(sorted(str(input_no_weights).split()))
        cand_indices = pipeline.token_sorted_map.get(sorted_input_tokens)
        if cand_indices:
            if isinstance(cand_indices, int):
                cand_indices = [cand_indices]

            selected_idx = cand_indices[0]
            weight_reason = ""

            if input_w_data[0] is not None:
                in_val, _, in_type = input_w_data
                best_match_idx = None
                min_diff_pct = float("inf")

                for c_idx in cand_indices:
                    cat_row = pipeline.raw_catalog.iloc[c_idx]
                    catalog_w_data = cat_row.get("weight_val")
                    if catalog_w_data is not None and isinstance(catalog_w_data, (tuple, list)) and catalog_w_data[0] is not None:
                        cat_val, _, cat_type = catalog_w_data
                        if in_type == cat_type:
                            max_val = max(in_val, cat_val)
                            diff_pct = abs(in_val - cat_val) / max_val * 100 if max_val > 0 else 0
                            if diff_pct < min_diff_pct:
                                min_diff_pct = diff_pct
                                best_match_idx = c_idx

                if best_match_idx is not None:
                    selected_idx = best_match_idx
                    cat_row = pipeline.raw_catalog.iloc[selected_idx]
                    catalog_w_data = cat_row.get("weight_val")
                    cat_val = catalog_w_data[0]
                    if min_diff_pct < 1.0:
                        weight_reason = f" | Weight Match ({int(in_val)})"
                    else:
                        weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(cat_val)})"
                else:
                    cat_row = pipeline.raw_catalog.iloc[selected_idx]
                    catalog_w_data = cat_row.get("weight_val")
                    if catalog_w_data is not None and isinstance(catalog_w_data, (tuple, list)) and catalog_w_data[0] is not None:
                        weight_reason = f" | Weight Mismatch ({int(in_val)} vs {int(catalog_w_data[0])})"

            exact_bypass_row = pipeline.raw_catalog.iloc[selected_idx]
            exact_bypass_reason = f"Whole-SKU Fuzzy Match (100%){weight_reason}"

    # -------------------------------------------------------------
    # Stage 2: Basic Type Prediction & Candidate Search
    # -------------------------------------------------------------
    encoded = pipeline.embedder.embed_weighted_sku(
        [clean_input], [description], [category],
        weights=config.MATCHER_WEIGHTS
    )
    input_vec_dense = encoded["dense"][0]
    input_vec_sparse = encoded["sparse"][0]

    bt_filter = None
    bt_prediction = {}
    if pipeline.classifier:
        try:
            bt_tag, confidence, source = pipeline.classifier.predict_bt(input_vec_dense, price=price)
            threshold = 0.40 if source == "zero-shot" else 0.50
            applied = (confidence >= threshold)
            if applied:
                bt_filter = bt_tag
            bt_prediction = {
                "predicted_bt": bt_tag,
                "confidence": float(confidence),
                "source": source,
                "threshold": threshold,
                "filter_applied": applied
            }
        except Exception as e:
            bt_prediction = {"error": str(e)}

    filtered_cands = pd.DataFrame()
    unfiltered_cands = pd.DataFrame()
    search_strategy = "BT-Filtered Hybrid Query" if bt_filter else "Full Hybrid Query"

    if exact_bypass_row is not None:
        search_strategy = f"Exact Catalog Lookup ({exact_bypass_reason})"

    if bt_filter:
        filtered_cands = pipeline.search_candidates(input_vec_dense, input_vec_sparse, bt_filter=bt_filter)
    
    unfiltered_cands = pipeline.search_candidates(input_vec_dense, input_vec_sparse)

    # Brand-stripped search fallback
    brand_stripped_query = ""
    stripped_candidates_added = 0
    stripped_cands = pd.DataFrame()

    if getattr(config, "ENABLE_BRAND_STRIPPED_SEARCH", True) and extracted_entities and any(extracted_entities.get("brand", [])):
        brand_stripped = clean_input
        for b in extracted_entities.get("brand", []):
            pattern = re.compile(r"\b" + re.escape(b) + r"\b", re.IGNORECASE)
            brand_stripped = pattern.sub("", brand_stripped)
        brand_stripped = " ".join(brand_stripped.split()).strip()

        if brand_stripped:
            brand_stripped_query = brand_stripped
            try:
                stripped_encoded = pipeline.embedder.embed_weighted_sku(
                    [brand_stripped], [description], [category],
                    weights=config.MATCHER_WEIGHTS
                )
                stripped_cands = pipeline.search_candidates(stripped_encoded["dense"][0], stripped_encoded["sparse"][0])
                if not stripped_cands.empty:
                    stripped_candidates_added = len(stripped_cands)
            except Exception:
                pass

    # Combine pools for cross-encoder tracing
    c_list = [df for df in [filtered_cands, unfiltered_cands, stripped_cands] if not df.empty]
    if c_list:
        candidates = pd.concat(c_list, ignore_index=True).drop_duplicates(subset=["clean_text"])
    else:
        candidates = pd.DataFrame()

    audit_data["stage2_candidate_retrieval"] = {
        "classifier_bt_prediction": bt_prediction,
        "bt_filter_used": bt_filter,
        "search_strategy": search_strategy,
        "brand_stripped_query": brand_stripped_query,
        "stripped_candidates_added": stripped_candidates_added,
        "total_candidates_found": len(candidates)
    }

    if candidates.empty:
        audit_data["final_output"] = {
            "matched_catalog_name": "",
            "score": 0.0,
            "status": "Low / Rejected",
            "logic_notes": "No candidates found in vector store",
            "suggested_bt": "",
            "suggested_gk": "",
            "suggested_region": ""
        }
        return audit_data

    # -------------------------------------------------------------
    # Stage 3: Cross-Encoder Scoring & Rankings
    # -------------------------------------------------------------
    pairs = [
        [input_no_weights, str(c_row.get("clean_no_weights") or TextPipeline.strip_weights(c_row.get("clean_text", ""))).strip()]
        for _, c_row in candidates.iterrows()
    ]
    cross_scores = pipeline.embedder.score_cross_encoder(pairs)

    input_tokens = len(clean_input.split())
    cat_tokens = candidates["token_count"].values
    token_penalties = np.where(cat_tokens > input_tokens + 2, 2.0, 0.0)

    brand_boosts = np.zeros(len(candidates))
    if getattr(config, "ENABLE_TEMPLATE_AWARE_MATCHING", False) and extracted_entities and any(extracted_entities.get("brand", [])):
        has_brand_mask = np.array([
            isinstance(ent, dict) and bool(ent.get("brand"))
            for ent in candidates["entities"]
        ])
        brand_boosts = np.where(has_brand_mask, 2.0, 0.0)

    final_cross_scores = cross_scores - token_penalties + brand_boosts
    sorted_indices = np.argsort(final_cross_scores)[::-1]

    candidates["raw_cross_score"] = cross_scores
    candidates["token_penalty"] = token_penalties
    candidates["brand_boost"] = brand_boosts
    candidates["final_cross_score"] = final_cross_scores

    top_candidates_trace = []
    if exact_bypass_row is not None:
        top_candidates_trace.append({
            "cand_idx": 0,
            "catalog_name": str(exact_bypass_row["Name"]),
            "brand": str(exact_bypass_row.get("Brand", exact_bypass_row.get("brand", ""))),
            "basic_type": str(exact_bypass_row.get("basictype", exact_bypass_row.get("BasicType", ""))),
            "raw_cross_score": 100.0,
            "token_penalty": 0.0,
            "brand_boost": 0.0,
            "final_cross_score": 100.0
        })

    for idx in sorted_indices[:10]:
        row = candidates.iloc[idx]
        if exact_bypass_row is not None and str(row["Name"]).strip().lower() == str(exact_bypass_row["Name"]).strip().lower():
            continue
        top_candidates_trace.append({
            "cand_idx": int(idx),
            "catalog_name": str(row["Name"]),
            "brand": str(row.get("Brand", row.get("brand", ""))),
            "basic_type": str(row.get("basictype", row.get("BasicType", ""))),
            "raw_cross_score": float(row["raw_cross_score"]),
            "token_penalty": float(row["token_penalty"]),
            "brand_boost": float(row["brand_boost"]),
            "final_cross_score": float(row["final_cross_score"])
        })
    audit_data["stage3_cross_encoder"] = {
        "top_candidates": top_candidates_trace
    }

    # -------------------------------------------------------------
    # Stage 4: Detailed Logic Gate Evaluations & Winner Selection
    # -------------------------------------------------------------
    evaluated_results = []
    best_res = None
    win_reasons = ""

    def eval_cand_df(df_pool: pd.DataFrame):
        res_list = []
        if df_pool.empty:
            return res_list
        
        # Sort pool by raw_cross_score descending to mirror SKUMatcher candidate evaluation ordering
        pool_rows = []
        for _, cand_row in df_pool.iterrows():
            cand_txt = cand_row["clean_text"]
            matching_rows = candidates[candidates["clean_text"] == cand_txt]
            raw_score = float(matching_rows.iloc[0]["raw_cross_score"]) if not matching_rows.empty else -10.0
            pool_rows.append((raw_score, cand_row))
        
        pool_rows.sort(key=lambda x: x[0], reverse=True)

        for raw_score, cand_row in pool_rows:
            cand_row_copy = cand_row.copy()
            gate_score, gate_status, gate_reasons = pipeline.rules.apply_logic_gates(
                clean_input, extracted_entities, cand_row_copy, raw_score,
                price, input_w_data, input_no_weights,
                domain=domain, input_description=description,
                input_category=category, predicted_bt=bt_filter or ""
            )
            res_list.append({
                "candidate_name": str(cand_row_copy["Name"]),
                "raw_cross_score": raw_score,
                "computed_score": float(gate_score),
                "status": gate_status,
                "fuzzy_bypass": gate_reasons.startswith("Whole-SKU Fuzzy Match"),
                "reasons": gate_reasons,
                "cand_row": cand_row_copy
            })
        res_list.sort(key=lambda x: x["computed_score"], reverse=True)
        return res_list

    if exact_bypass_row is not None:
        best_res = {
            "candidate_name": str(exact_bypass_row["Name"]),
            "raw_cross_score": 100.0,
            "computed_score": 100.0,
            "status": "High Confidence",
            "fuzzy_bypass": True,
            "reasons": exact_bypass_reason,
            "cand_row": exact_bypass_row
        }
        evaluated_results.append({**best_res, "rank": 1})
        win_reasons = exact_bypass_reason

    elif bt_filter and not filtered_cands.empty:
        filtered_eval = eval_cand_df(filtered_cands)
        for rank_i, item in enumerate(filtered_eval[:10]):
            evaluated_results.append({**item, "rank": rank_i + 1})

        if filtered_eval and filtered_eval[0]["status"] == "High Confidence":
            best_res = filtered_eval[0]
            win_reasons = f"BT-Filtered ({bt_filter}) | {best_res['reasons']}"
        else:
            fallback_res = filtered_eval[0] if filtered_eval else None
            unfiltered_pool = pd.concat([unfiltered_cands, stripped_cands], ignore_index=True).drop_duplicates(subset=["clean_text"]) if not stripped_cands.empty else unfiltered_cands
            unfiltered_eval = eval_cand_df(unfiltered_pool)

            for rank_i, item in enumerate(unfiltered_eval[:10]):
                if not any(r["candidate_name"] == item["candidate_name"] for r in evaluated_results):
                    evaluated_results.append({**item, "rank": len(evaluated_results) + 1})

            best_uf = unfiltered_eval[0] if unfiltered_eval else None

            if fallback_res and (not best_uf or fallback_res["computed_score"] > best_uf["computed_score"]):
                best_res = fallback_res
                win_reasons = f"BT-Filtered ({bt_filter}) [Low Confidence Fallback] | {fallback_res['reasons']}"
            elif best_uf:
                best_res = best_uf
                if bt_filter:
                    win_reasons = f"BT-Fallback (BT: {bt_filter}, score: {best_uf['computed_score']:.4f}) | {best_uf['reasons']}"
                else:
                    win_reasons = best_uf["reasons"]
    else:
        unfiltered_pool = pd.concat([unfiltered_cands, stripped_cands], ignore_index=True).drop_duplicates(subset=["clean_text"]) if not stripped_cands.empty else unfiltered_cands
        unfiltered_eval = eval_cand_df(unfiltered_pool)
        for rank_i, item in enumerate(unfiltered_eval[:10]):
            evaluated_results.append({**item, "rank": rank_i + 1})

        if unfiltered_eval:
            best_res = unfiltered_eval[0]
            win_reasons = best_res["reasons"]

    # Add why non-winners didn't win
    winner_row = None
    best_res = None
    for res in evaluated_results:
        if res["status"] == "High Confidence":
            best_res = res
            break
    if best_res is None:
        for res in evaluated_results:
            if best_res is None or res["computed_score"] > best_res["computed_score"]:
                best_res = res

    winner_name = best_res["candidate_name"] if best_res else ""

    for res in evaluated_results:
        if res["candidate_name"] == winner_name:
            res["win_status_note"] = "WINNER (Selected as best matching catalog item)"
        else:
            if res["status"] == "Rejected":
                res["win_status_note"] = f"DID NOT WIN: Rejected by logic gate logic ({res['reasons']})"
            else:
                diff = best_res["computed_score"] - res["computed_score"]
                if abs(diff) < 1e-4:
                    raw_diff = best_res["raw_cross_score"] - res["raw_cross_score"]
                    res["win_status_note"] = f"DID NOT WIN: Tied computed score ({res['computed_score']:.1f}), lost tie-breaker on raw cross-encoder score by {raw_diff:.4f} pts ({res['reasons']})"
                else:
                    res["win_status_note"] = f"DID NOT WIN: Lower computed score by {diff:.4f} pts ({res['reasons']})"

    audit_data["stage4_logic_gates"] = [
        {k: v for k, v in res.items() if k != "cand_row"} for res in evaluated_results
    ]

    # -------------------------------------------------------------
    # Stage 5: Matcher Outcome & Generic Keyword Filtering
    # -------------------------------------------------------------
    if best_res:
        winner = best_res["cand_row"]
        win_score = best_res["computed_score"]
        win_status = best_res["status"]
        win_reasons = best_res["reasons"]
    else:
        winner = candidates.iloc[0]
        win_score = 0.0
        win_status = "Low / Rejected"
        win_reasons = "No candidates passed gates"

    winner_gk = str(winner.get("Generic keywords", winner.get("GenericKeywords", "")))
    orig_gk = winner_gk

    if winner is not None and domain == config.DOMAIN_FOOD and getattr(pipeline, "flavor_categories", None):
        input_flavors = set()
        if extracted_entities and isinstance(extracted_entities, dict):
            input_flavors.update(x.lower() for x in extracted_entities.get("flavor", set()) if x)
        resolved_input_flavors = pipeline.rules._resolve_flavors(input_flavors) if hasattr(pipeline.rules, "_resolve_flavors") else input_flavors

        input_meats = {f for f in resolved_input_flavors if pipeline.flavor_categories.get(f, (False, False, False))[0] and not pipeline.flavor_categories.get(f, (False, False, False))[2]}
        input_seafoods = {f for f in resolved_input_flavors if pipeline.flavor_categories.get(f, (False, False, False))[2]}
        input_vegs = {f for f in resolved_input_flavors if pipeline.flavor_categories.get(f, (False, False, False))[1]}

        if winner_gk:
            kws = [k.strip() for k in winner_gk.split(",") if k.strip()]
            filtered_kws = []
            for kw in kws:
                kw_lower = kw.lower()
                if any(re.search(r"\b" + re.escape(t) + r"\b", kw_lower) for t in pipeline.seafood_terms):
                    has_input_seafood_term = any(re.search(r"\b" + re.escape(t) + r"\b", clean_input) for t in pipeline.seafood_terms)
                    if not (has_input_seafood_term or input_seafoods):
                        continue
                elif any(re.search(r"\b" + re.escape(t) + r"\b", kw_lower) for t in pipeline.mixed_terms):
                    has_input_mixed_term = any(re.search(r"\b" + re.escape(t) + r"\b", clean_input) for t in pipeline.mixed_terms)
                    total_unique_items = len(input_meats) + len(input_seafoods)
                    is_mixed_by_flavors = (total_unique_items >= 2 and len(input_meats) >= 1)
                    if not (has_input_mixed_term or is_mixed_by_flavors):
                        continue
                if any(re.search(r"\b" + re.escape(t) + r"\b", kw_lower) for t in pipeline.veg_terms):
                    has_input_veg_term = any(re.search(r"\b" + re.escape(t) + r"\b", clean_input) for t in pipeline.veg_terms)
                    if not (has_input_veg_term or input_vegs):
                        continue

                keep_kw = True
                for term, (is_meat, is_veg, is_seafood) in pipeline.flavor_categories.items():
                    if term in pipeline.mixed_terms or term in pipeline.seafood_terms or term in pipeline.veg_terms:
                        continue
                    pattern = r"\b" + re.escape(term) + r"\b"
                    if re.search(pattern, kw_lower):
                        canonical = pipeline.rules.food_flavors_dict.get(term, term) if hasattr(pipeline.rules, "food_flavors_dict") else term
                        if is_seafood and canonical not in input_seafoods:
                            keep_kw = False; break
                        elif is_meat and canonical not in input_meats:
                            keep_kw = False; break
                        elif is_veg and canonical not in input_vegs:
                            keep_kw = False; break

                if keep_kw:
                    filtered_kws.append(kw)
            winner_gk = ", ".join(filtered_kws)

    region_val = str(winner.get("Region", winner.get("Categories", "")))
    
    audit_data["stage5_matcher_result"] = {
        "matched_catalog_name": str(winner["Name"]),
        "score": float(win_score),
        "status": win_status,
        "logic_notes": win_reasons,
        "suggested_bt": str(winner.get("basictype", winner.get("BasicType", ""))),
        "original_gk": orig_gk,
        "filtered_gk": winner_gk,
        "suggested_region": region_val
    }

    # If task is matcher-only, finish at stage 5 without escalation
    if task == "matcher":
        audit_data["final_output"] = {
            "matched_catalog_name": str(winner["Name"]),
            "score": float(win_score),
            "status": win_status,
            "logic_notes": win_reasons,
            "suggested_bt": str(winner.get("basictype", winner.get("BasicType", ""))),
            "suggested_gk": winner_gk,
            "suggested_region": region_val,
            "pipeline_source": "Matcher",
            "escalated": False,
            "rules_applied": []
        }
        return audit_data

    # Current working tags from matcher
    curr_bt = str(winner.get("basictype", winner.get("BasicType", "")))
    curr_gk = winner_gk
    curr_region = region_val
    pipeline_source = "Matcher"
    escalated = False

    # -------------------------------------------------------------
    # Stage 6: Escalation Check
    # -------------------------------------------------------------
    if win_status in config.PIPELINE_ESCALATE_STATUSES:
        escalated = True
        classifier = pipeline.classifier
        if classifier:
            try:
                from engine.classification.tagger import tag_all_skus
                query_embeddings = [{"dense": input_vec_dense, "sparse": input_vec_sparse}]
                class RerankerWrapper:
                    def predict(self, pairs): return pipeline.embedder.score_cross_encoder(pairs)
                reranker = RerankerWrapper() if (hasattr(pipeline.embedder, 'cross_session') and pipeline.embedder.cross_session) else None

                clf_results = tag_all_skus(
                    sku_names=[sku_name],
                    sku_categories=[category],
                    query_embeddings=query_embeddings,
                    vector_store=pipeline.vector_store,
                    reranker=reranker,
                    classifier=classifier,
                    sku_descriptions=[description],
                    sku_prices=[price or 0.0],
                    embed_engine=pipeline.embedder,
                    ner_engine=getattr(classifier, 'ner_engine', pipeline.ner)
                )

                clf = clf_results[0]
                bt_tag = str(clf.get("suggested_bt", ""))
                bt_conf = float(clf.get("bt_confidence", 0.0))
                gk_str = str(clf.get("suggested_gk", ""))
                gk_conf = float(clf.get("gk_confidence", 0.0))
                tt_val = str(clf.get("suggested_region", clf.get("suggested_category", "")))
                tt_conf = float(clf.get("region_confidence", clf.get("category_confidence", 0.0)))

                matcher_norm = min(win_score / 100.0, 1.0)
                classifier_won = (bt_conf > matcher_norm)
                pipeline_source = "Classifier" if classifier_won else "Matcher"

                if classifier_won:
                    curr_bt = bt_tag
                    curr_gk = gk_str
                    curr_region = tt_val

                audit_data["stage6_escalation"] = {
                    "escalated": True,
                    "escalation_reason": f"Matcher status '{win_status}' is in PIPELINE_ESCALATE_STATUSES",
                    "classifier_bt": bt_tag,
                    "classifier_bt_confidence": float(bt_conf),
                    "classifier_gk": gk_str,
                    "classifier_gk_confidence": float(gk_conf),
                    "classifier_region": tt_val,
                    "classifier_region_confidence": float(tt_conf),
                    "matcher_normalized_score": float(matcher_norm),
                    "escalation_winner": pipeline_source
                }
            except Exception as e:
                audit_data["stage6_escalation"] = {"escalated": True, "error": str(e)}
        else:
            audit_data["stage6_escalation"] = {"escalated": True, "note": "No classifier loaded"}
    else:
        audit_data["stage6_escalation"] = {
            "escalated": False,
            "note": f"Matcher status '{win_status}' does not trigger escalation"
        }

    # -------------------------------------------------------------
    # Stage 7: Template Tag Enrichment
    # -------------------------------------------------------------
    template_applied = False
    template_details = {}
    if getattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", True):
        try:
            if win_status not in ["Exact Text Match", "High Confidence"]:
                if suggest_fn is None:
                    try:
                        from engine.template_suggest import suggest_tags_from_template
                        suggest_fn = suggest_tags_from_template
                    except Exception:
                        suggest_fn = None
                        
                if suggest_fn is not None:
                    sug_res = suggest_fn(sku_name, domain=domain, current_bt=curr_bt)
                else:
                    sug_res = {"matched": False, "reason": "Template suggest service unavailable"}

                if sug_res.get("matched"):
                    template_applied = True
                    s_bt = sug_res.get("suggested_bt", "")
                    s_gk_list = sug_res.get("suggested_gk", [])
                    s_gk = ", ".join(s_gk_list) if isinstance(s_gk_list, list) else str(s_gk_list)
                    
                    if s_bt:
                        curr_bt = s_bt
                    if s_gk:
                        curr_gk = s_gk
                        
                    template_details = {
                        "matched_template": True,
                        "base_sku": sug_res.get("base_sku", ""),
                        "template_bt": s_bt,
                        "template_gk": s_gk,
                        "notes": f"Template tag enrichment applied from base SKU '{sug_res.get('base_sku', '')}'"
                    }
                else:
                    template_details = {
                        "matched_template": False,
                        "note": sug_res.get("reason", "No template match found in catalog")
                    }
            else:
                template_details = {
                    "matched_template": False,
                    "note": "Skipped template enrichment because matcher status was High Confidence / Exact Text Match"
                }
        except Exception as e:
            template_details = {"matched_template": False, "error": str(e)}
    else:
        template_details = {
            "matched_template": False,
            "note": "Template tag enrichment is disabled in configuration"
        }

    audit_data["stage7_template_enrichment"] = template_details

    # -------------------------------------------------------------
    # Stage 8: Business Rules Engine Execution
    # -------------------------------------------------------------
    gk_list = [x.strip() for x in str(curr_gk).split(",") if x.strip()]
    clf_conf = float(audit_data.get("stage6_escalation", {}).get("classifier_bt_confidence", 0.0)) if escalated else 0.0
    rules_confidence = max(win_score, clf_conf, 0.95 if template_applied else 0.0)

    record = {
        "sku_name": sku_name,
        "domain": domain,
        "bt": curr_bt,
        "gk": gk_list,
        "region": curr_region if domain == config.DOMAIN_FOOD else None,
        "category": curr_region if domain == config.DOMAIN_MARKET else None,
        "price": price,
        "confidence": rules_confidence,
        "match_source": "classifier" if pipeline_source == "Classifier" else "catalogue",
        "matched_sku": str(winner["Name"]),
        "reasoning": win_reasons
    }

    aug_record = run_rules_engine(record)

    final_bt = str(aug_record.get("bt") or "")
    final_gk = ", ".join(aug_record.get("gk", [])) if isinstance(aug_record.get("gk"), list) else str(aug_record.get("gk") or "")
    final_region = str(aug_record.get("region") or aug_record.get("category") or "")
    rules_applied = aug_record.get("rules_applied", [])

    audit_data["stage8_rules_engine"] = {
        "rules_applied_count": len(rules_applied),
        "rules_applied": rules_applied,
        "pre_rules_tags": {"bt": curr_bt, "gk": curr_gk, "region": curr_region},
        "post_rules_tags": {"bt": final_bt, "gk": final_gk, "region": final_region}
    }

    # -------------------------------------------------------------
    # Stage 9: Final Output Construction
    # -------------------------------------------------------------
    audit_data["final_output"] = {
        "matched_catalog_name": str(winner["Name"]),
        "score": float(win_score),
        "status": win_status,
        "logic_notes": win_reasons,
        "suggested_bt": final_bt,
        "suggested_gk": final_gk,
        "suggested_region": final_region,
        "pipeline_source": pipeline_source,
        "escalated": escalated,
        "rules_applied": rules_applied
    }

    return audit_data

def _audit_classifier_pipeline(audit_data: dict, domain: str, sku_name: str, description: str, category: str, price: float, classifier=None, embed_engine=None, ner_engine=None, vector_store=None):
    """Executes a deep multi-stage diagnostic audit for classifier-only task."""
    if classifier is None:
        try:
            from engine.resource_loader import get_classifier
            classifier = get_classifier(domain)
        except Exception:
            classifier = None
            
    if embed_engine is None or ner_engine is None:
        try:
            from engine.resource_loader import _get_shared_models
            e_mod, n_mod = _get_shared_models()
            embed_engine = embed_engine or e_mod
            ner_engine = ner_engine or n_mod
        except Exception:
            pass
            
    if vector_store is None:
        try:
            from engine.resource_loader import _get_vector_store
            vector_store = _get_vector_store()
        except Exception:
            pass
    
    # 1. Embed query
    query_embs = embed_engine.embed_weighted_sku([sku_name], [description], [category], weights=config.CLASSIFIER_WEIGHTS)
    vec_dense = query_embs["dense"][0]
    vec_sparse = query_embs["sparse"][0]
    vec_2d = vec_dense.reshape(1, -1)
    
    # Extract entities from stage1_nlp
    stage1 = audit_data.get("stage1_nlp", {})
    extracted_entities = stage1.get("ner_entities", {})
    
    market_brand = ""
    extracted_flavors = []
    if domain == "market":
        b_list = extracted_entities.get("brand", [])
        if b_list:
            market_brand = sorted(b_list, key=len, reverse=True)[0].title()
    elif domain == "food":
        extracted_flavors = list(extracted_entities.get("flavor", []))
        
    # -------------------------------------------------------------
    # Stage 2: Basic Type (BT) Prediction & Candidate Pool
    # -------------------------------------------------------------
    bt_candidates = []
    predicted_bt = ""
    bt_conf = 0.0
    bt_source = "zero-shot"
    p_val = float(price) if price is not None else 0.0
    
    if classifier._trained and getattr(classifier, "_bt_clf", None) is not None:
        scaled_p = classifier._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])
        proba = classifier._bt_clf.predict_proba(vec_with_price)[0]
        top_indices = np.argsort(proba)[::-1][:10]
        for idx in top_indices:
            bt_candidates.append({
                "bt": str(classifier._bt_enc.classes_[idx]),
                "score": float(proba[idx]),
                "source": "trained"
            })
        best_idx = int(np.argmax(proba))
        best_prob = float(proba[best_idx])
        if best_prob >= 0.40:
            predicted_bt = str(classifier._bt_enc.classes_[best_idx])
            bt_conf = best_prob
            bt_source = "trained"

    if not predicted_bt and classifier.bt_labels:
        scores_pure = (vec_2d @ classifier.bt_embs_pure.T)[0]
        scores_desc = (vec_2d @ classifier.bt_embs_desc.T)[0]
        scores = np.maximum(scores_pure, scores_desc)
        top_indices = np.argsort(scores)[::-1][:10]
        if not bt_candidates:
            for idx in top_indices:
                bt_candidates.append({
                    "bt": str(classifier.bt_labels[idx]),
                    "score": float(scores[idx]),
                    "source": "zero-shot"
                })
        best_idx = int(np.argmax(scores))
        predicted_bt = str(classifier.bt_labels[best_idx])
        bt_conf = float(scores[best_idx])
        bt_source = "zero-shot"

    # Flavor Conflict check in Food domain
    flavor_conflict = False
    conflict_notes = ""
    if extracted_flavors and predicted_bt and hasattr(classifier, "food_flavors_dict"):
        from engine.classification.tagger import _resolve_flavors_from_text
        bt_flavors = _resolve_flavors_from_text(predicted_bt, classifier.food_flavors_dict)
        input_flavors = set()
        for f in extracted_flavors:
            input_flavors.update(_resolve_flavors_from_text(f, classifier.food_flavors_dict))
        if bt_flavors and input_flavors and bt_flavors.isdisjoint(input_flavors):
            flavor_conflict = True
            conflict_notes = f"Flavor conflict: BT '{predicted_bt}' flavors {list(bt_flavors)} disjoint from input flavors {list(input_flavors)}"
            predicted_bt = ""
            bt_conf = 0.0
            bt_source = "conflict"

    from engine.classification.tagger import get_status
    bt_status = get_status(bt_conf, bool(predicted_bt), bt_source)

    audit_data["stage2_bt_classification"] = {
        "predicted_bt": predicted_bt,
        "confidence": bt_conf,
        "source": bt_source,
        "status": bt_status,
        "threshold": 0.40 if bt_source == "zero-shot" else 0.50,
        "flavor_conflict": flavor_conflict,
        "conflict_notes": conflict_notes,
        "top_candidates": bt_candidates
    }

    # -------------------------------------------------------------
    # Stage 3: Generic Keywords (GK) Retrieval, Scoring & Filtering Audit
    # -------------------------------------------------------------
    from engine.classification.tagger import (
        get_strategy, _rrf_fusion, _weighted_fusion, _get_gk_regex,
        FUSION_METHOD, TOP_K_FUSED, USE_RERANKER, RERANKER_THRESHOLD
    )
    strategy = get_strategy(classifier.domain)

    # 1. Guaranteed umbrella tags
    guaranteed = classifier.get_guaranteed_gk(predicted_bt)
    if not isinstance(guaranteed, list):
        guaranteed = list(guaranteed)
    else:
        guaranteed = list(guaranteed)

    bt_gk_map = getattr(classifier, 'bt_gk_map', {})
    allowed_gks_for_bt = bt_gk_map.get(predicted_bt, [])
    allowed_gks_lower = set(kw.lower().strip() for kw in allowed_gks_for_bt)

    # BT-as-GK fallback
    if predicted_bt and predicted_bt not in guaranteed:
        bt_lower = predicted_bt.lower().strip()
        all_known_gks_lower = set()
        for gk_list in bt_gk_map.values():
            for gk in gk_list:
                all_known_gks_lower.add(gk.lower().strip())
        if bt_lower in all_known_gks_lower:
            guaranteed.append(predicted_bt)

    # Literal regex match from SKU Name
    literal_matched_gks = []
    sku_name_lower = sku_name.lower()
    if allowed_gks_for_bt:
        gk_pool = [gk.lower().strip() for gk in allowed_gks_for_bt if gk.strip()]
        gk_map = {gk.lower().strip(): gk for gk in allowed_gks_for_bt if gk.strip()}
    else:
        gk_map = {}
        for gk_list in bt_gk_map.values():
            for gk in gk_list:
                clean = gk.lower().strip()
                if clean and clean not in gk_map:
                    gk_map[clean] = gk
        gk_pool = list(gk_map.keys())

    if gk_pool:
        pattern = _get_gk_regex(tuple(gk_pool))
        for match in pattern.finditer(sku_name_lower):
            matched_gk_lower = match.group(1)
            original_gk = gk_map[matched_gk_lower]
            literal_matched_gks.append(original_gk)
            if original_gk not in guaranteed:
                guaranteed.append(original_gk)

    # 2. Trained GK tags
    trained_gk = []
    trained_conf = 0.0
    trained_gk_trace = []
    if classifier._trained and getattr(classifier, "_gk_clf", None) is not None:
        scaled_p = classifier._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])
        gk_proba = classifier._gk_clf.predict_proba(vec_with_price)[0]
        threshold = 0.50
        predicted_indices = np.where(gk_proba >= threshold)[0]
        if len(predicted_indices) > 0:
            trained_gk = classifier._gk_enc.classes_[predicted_indices].tolist()
            trained_conf = float(np.mean(gk_proba[predicted_indices]))
        top_gk_idx = np.argsort(gk_proba)[::-1][:15]
        for idx in top_gk_idx:
            prob = float(gk_proba[idx])
            if prob >= 0.05:
                trained_gk_trace.append({
                    "tag": str(classifier._gk_enc.classes_[idx]),
                    "score": prob,
                    "threshold_passed": bool(prob >= threshold)
                })

    # Strategy: filter search tags
    search_allowed_tags, early_return = strategy.filter_search_tags(allowed_gks_for_bt, trained_conf, "trained" if trained_gk else "zero-shot", guaranteed)
    
    dense_hits, sparse_hits = [], []
    fused_candidates = []
    if early_return is None and vector_store is not None:
        dense_hits, sparse_hits = vector_store.search_hybrid_tags(
            dense_query=vec_dense,
            sparse_query=vec_sparse,
            limit=50,
            filter_dict_type="gk",
            domain=classifier.domain,
            allowed_tags=search_allowed_tags
        )
        if FUSION_METHOD == "rrf":
            fused_candidates = _rrf_fusion(dense_hits, sparse_hits)
        else:
            fused_candidates = _weighted_fusion(dense_hits, sparse_hits)

    top_candidates = fused_candidates[:TOP_K_FUSED]
    raw_vector_cands_count = len(top_candidates)

    # Strategy: post search filters
    top_candidates = strategy.apply_post_search_filters(top_candidates, allowed_gks_lower)

    # Strategy: Inject synthetic tags
    synth_before = len(top_candidates)
    top_candidates = strategy.inject_synthetic_tags(top_candidates, market_brand, predicted_bt, category, extracted_flavors)
    synth_added = len(top_candidates) - synth_before

    # Strategy: GK Flavor Leak & Conflict Filters
    ner_engine_clf = getattr(classifier, "ner_engine", ner_engine)
    guaranteed_clean, trained_gk_clean, top_candidates_clean = strategy.apply_flavor_leak_filters(
        list(guaranteed), list(trained_gk), list(top_candidates), extracted_flavors, ner_engine_clf
    )

    pruned_by_flavor = []
    top_cands_clean_tags = {c["tag"].lower().strip() for c in top_candidates_clean}
    for c in top_candidates:
        if c["tag"].lower().strip() not in top_cands_clean_tags:
            pruned_by_flavor.append(c["tag"])

    # Reranking Layer
    class RerankerWrapper:
        def predict(self, pairs): return embed_engine.score_cross_encoder(pairs)
    reranker = RerankerWrapper() if (hasattr(embed_engine, 'cross_session') and embed_engine.cross_session) else None

    reranked_tags = []
    final_conf = 0.0
    scored_candidates = []

    if top_candidates_clean and USE_RERANKER and reranker is not None:
        query_text = (sku_name + " " + description).strip()
        pairs = [[query_text, c["tag"]] for c in top_candidates_clean]
        raw_scores = reranker.predict(pairs)
        if isinstance(raw_scores, float) or (hasattr(raw_scores, 'item') and raw_scores.ndim == 0):
            scores = [float(raw_scores)]
        else:
            scores = [float(s) for s in raw_scores]
        for c, score in zip(top_candidates_clean, scores):
            tag_lower = c["tag"].lower().strip()
            passed = strategy.get_reranker_threshold(score, tag_lower, allowed_gks_lower)
            scored_candidates.append({
                "tag": c["tag"],
                "score": score,
                "threshold_passed": bool(passed),
                "source": c.get("source", "hybrid_search")
            })
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        reranked_tags = [item["tag"] for item in scored_candidates if item["threshold_passed"]]
        if scored_candidates:
            final_conf = scored_candidates[0]["score"]
    else:
        reranked_tags = [c["tag"] for c in top_candidates_clean]
        if top_candidates_clean:
            final_conf = min(1.0, top_candidates_clean[0]["score"])

    # Merge final tags
    trained_gk_final = strategy.filter_final_trained_gk(trained_gk_clean, allowed_gks_lower)
    merged_gk = []
    seen = set()
    for tag in guaranteed_clean + trained_gk_final + reranked_tags:
        key = tag.lower().strip()
        if key not in seen:
            seen.add(key)
            merged_gk.append(tag)

    gk_conf = trained_conf if (trained_gk_final and trained_conf >= 0.8) else final_conf
    gk_status = get_status(gk_conf, bool(merged_gk))

    # Consolidated keywords considered table
    all_considered_map = {}
    
    # 1. Guaranteed / Literal
    for g_tag in guaranteed:
        key = g_tag.lower().strip()
        all_considered_map[key] = {
            "tag": g_tag,
            "source": "Guaranteed / Schema" if g_tag not in literal_matched_gks else "Literal SKU Match",
            "base_score": 1.0,
            "reranker_score": None,
            "threshold_passed": True,
            "flavor_filter_passed": g_tag in guaranteed_clean,
            "prune_reason": None if g_tag in guaranteed_clean else "Flavor conflict with SKU entities",
            "is_selected": g_tag in merged_gk
        }

    # 2. Trained ML GKs
    for t_item in trained_gk_trace:
        key = t_item["tag"].lower().strip()
        if key not in all_considered_map:
            is_kept = t_item["tag"] in trained_gk_final
            all_considered_map[key] = {
                "tag": t_item["tag"],
                "source": "Trained Multi-Label ML",
                "base_score": t_item["score"],
                "reranker_score": None,
                "threshold_passed": t_item["threshold_passed"],
                "flavor_filter_passed": t_item["tag"] in trained_gk_clean,
                "prune_reason": None if is_kept else ("Probability < 0.50" if not t_item["threshold_passed"] else "Filtered by schema / flavor"),
                "is_selected": t_item["tag"] in merged_gk
            }

    # 3. Vector Hybrid Candidates & Reranked
    for s_item in scored_candidates:
        key = s_item["tag"].lower().strip()
        if key not in all_considered_map:
            is_kept = s_item["tag"] in merged_gk
            all_considered_map[key] = {
                "tag": s_item["tag"],
                "source": "Synthetic Injected" if s_item["source"] == "synthetic" else "Vector Hybrid Search",
                "base_score": None,
                "reranker_score": s_item["score"],
                "threshold_passed": s_item["threshold_passed"],
                "flavor_filter_passed": True,
                "prune_reason": None if s_item["threshold_passed"] else f"Reranker score ({s_item['score']:.3f}) < threshold ({RERANKER_THRESHOLD})",
                "is_selected": is_kept
            }
        else:
            all_considered_map[key]["reranker_score"] = s_item["score"]

    # Also include any vector candidates that were pruned before reranking
    for f_item in fused_candidates[:TOP_K_FUSED]:
        key = f_item["tag"].lower().strip()
        if key not in all_considered_map:
            pruned_reason = "Flavor conflict" if f_item["tag"] in pruned_by_flavor else "Filtered by schema mapping"
            all_considered_map[key] = {
                "tag": f_item["tag"],
                "source": "Vector Hybrid Search",
                "base_score": f_item["score"],
                "reranker_score": None,
                "threshold_passed": False,
                "flavor_filter_passed": f_item["tag"] not in pruned_by_flavor,
                "prune_reason": pruned_reason,
                "is_selected": False
            }

    considered_keywords_list = list(all_considered_map.values())
    considered_keywords_list.sort(key=lambda x: (not x["is_selected"], -(x["reranker_score"] if x["reranker_score"] is not None else (x["base_score"] or 0))))

    audit_data["stage3_gk_classification"] = {
        "suggested_gk": ", ".join(merged_gk),
        "confidence": gk_conf,
        "status": gk_status,
        "guaranteed_tags": guaranteed_clean,
        "trained_gk_tags": trained_gk_final,
        "literal_matched_tags": literal_matched_gks,
        "vector_search_hits_count": len(dense_hits) + len(sparse_hits),
        "synthetic_tags_injected": synth_added,
        "pruned_by_flavor_leak": pruned_by_flavor,
        "considered_keywords": considered_keywords_list
    }

    # -------------------------------------------------------------
    # Stage 4: Third Tag (Region / Category) Prediction Audit
    # -------------------------------------------------------------
    third_tag_candidates = []
    chosen_third_tag = ""
    third_tag_conf = 0.0
    third_tag_source = "zero-shot"
    override_applied = False

    if predicted_bt and predicted_bt in classifier.third_tag_overrides:
        chosen_third_tag = classifier.third_tag_overrides[predicted_bt]
        third_tag_conf = 1.0
        third_tag_source = "override"
        override_applied = True
        third_tag_candidates.append({
            "tag": chosen_third_tag,
            "score": 1.0,
            "source": "override",
            "note": f"Exact catalog override mapped from BT '{predicted_bt}'"
        })
    elif classifier._trained and getattr(classifier, "_third_tag_clf", None) is not None:
        scaled_p = classifier._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])
        tt_proba = classifier._third_tag_clf.predict_proba(vec_with_price)[0]
        top_tt_indices = np.argsort(tt_proba)[::-1][:10]
        for idx in top_tt_indices:
            third_tag_candidates.append({
                "tag": str(classifier._third_tag_enc.classes_[idx]),
                "score": float(tt_proba[idx]),
                "source": "trained"
            })
        best_tt_idx = int(np.argmax(tt_proba))
        best_tt_conf = float(tt_proba[best_tt_idx])
        if best_tt_conf >= 0.40:
            chosen_third_tag = str(classifier._third_tag_enc.classes_[best_tt_idx])
            third_tag_conf = best_tt_conf
            third_tag_source = "trained"

    if not chosen_third_tag and classifier.third_tag_labels:
        scores_pure = (vec_2d @ classifier.third_tag_embs_pure.T)[0]
        scores_desc = (vec_2d @ classifier.third_tag_embs_desc.T)[0]
        scores = np.maximum(scores_pure, scores_desc)
        top_tt_indices = np.argsort(scores)[::-1][:10]
        if not third_tag_candidates:
            for idx in top_tt_indices:
                third_tag_candidates.append({
                    "tag": str(classifier.third_tag_labels[idx]),
                    "score": float(scores[idx]),
                    "source": "zero-shot"
                })
        best_tt_idx = int(np.argmax(scores))
        chosen_third_tag = str(classifier.third_tag_labels[best_tt_idx])
        third_tag_conf = float(scores[best_tt_idx])
        third_tag_source = "zero-shot"

    third_tag_status = get_status(third_tag_conf, bool(chosen_third_tag), third_tag_source)
    tag_label = "Region" if domain == "food" else "Category"

    audit_data["stage4_third_tag_classification"] = {
        "tag_label": tag_label,
        "suggested_tag": chosen_third_tag,
        "confidence": third_tag_conf,
        "status": third_tag_status,
        "source": third_tag_source,
        "override_applied": override_applied,
        "top_candidates": third_tag_candidates
    }

    # -------------------------------------------------------------
    # Stage 5: Business Rules Engine Execution
    # -------------------------------------------------------------
    rules_confidence = max(bt_conf, gk_conf, third_tag_conf)
    record = {
        "sku_name": sku_name,
        "domain": domain,
        "bt": predicted_bt,
        "gk": merged_gk,
        "region": chosen_third_tag if domain == config.DOMAIN_FOOD else None,
        "category": chosen_third_tag if domain == config.DOMAIN_MARKET else None,
        "price": price,
        "confidence": rules_confidence,
        "match_source": "classifier",
        "matched_sku": "",
        "reasoning": f"Classifier predicted BT='{predicted_bt}', GK='{', '.join(merged_gk)}', {tag_label}='{chosen_third_tag}'"
    }

    aug_record = run_rules_engine(record)
    final_bt = str(aug_record.get("bt") or "")
    final_gk = ", ".join(aug_record.get("gk", [])) if isinstance(aug_record.get("gk"), list) else str(aug_record.get("gk") or "")
    final_region = str(aug_record.get("region") or aug_record.get("category") or "")
    rules_applied = aug_record.get("rules_applied", [])

    audit_data["stage5_rules_engine"] = {
        "rules_applied_count": len(rules_applied),
        "rules_applied": rules_applied,
        "pre_rules_tags": {"bt": predicted_bt, "gk": ", ".join(merged_gk), "region": chosen_third_tag},
        "post_rules_tags": {"bt": final_bt, "gk": final_gk, "region": final_region}
    }

    # -------------------------------------------------------------
    # Stage 6: Final Classifier Output
    # -------------------------------------------------------------
    audit_data["final_output"] = {
        "suggested_bt": final_bt,
        "bt_confidence": bt_conf,
        "bt_status": bt_status,
        "bt_source": bt_source,
        "suggested_gk": final_gk,
        "gk_confidence": gk_conf,
        "gk_status": gk_status,
        "suggested_region": final_region,
        "region_confidence": third_tag_conf,
        "region_status": third_tag_status,
        "region_source": third_tag_source,
        "rules_applied": rules_applied,
        "pipeline_source": "Classifier"
    }

