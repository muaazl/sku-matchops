import re

import numpy as np
import pandas as pd

from engine.core import config
from engine.pipelines.classification_pipeline import (
    _get_gk_regex,
    _rrf_fusion,
    _weighted_fusion,
    get_status,
    get_strategy,
    resolve_gk_confidence_and_status,
    tag_all_skus,
)
from engine.pipelines.matching_pipeline import (
    filter_matched_generic_keywords,
    is_better_gated_candidate,
)
from engine.pipelines.rules_evaluation_pipeline import run_rules_engine
from engine.pipelines.runtime_pipeline import (
    _get_reranker_model,
    _get_shared_models,
    _get_vector_store,
    get_classifier,
    get_ner_engine,
    get_pipeline,
)
from engine.pipelines.template_pipeline import resolve_template_enrichment_for_sku
from engine.preprocessing.flavor_preprocessor import (
    extract_canonical_flavors,
    resolve_food_mixed_suffix,
)
from engine.preprocessing.text_normalizer import (
    build_ner_input_text,
    extract_weight_feature,
    format_catalog_row_for_reranker,
    format_matcher_rerank_text,
    normalize_sku_text,
    prepare_ner_text,
    standardize_sku_units,
    strip_weight_tokens,
)
from engine.preprocessing.weight_preprocessor import (
    resolve_bt_consensus_indices,
    resolve_weight_bypass_candidate,
)


def _run_audit_template_enrichment(
    sku_name: str,
    domain: str,
    curr_bt: str,
    curr_gk: str,
    status_to_check: str,
    suggest_fn=None,
) -> tuple[str, str, bool, str, str, dict]:
    """Shared template enrichment helper across matcher, pipeline, and classifier audit paths."""
    return resolve_template_enrichment_for_sku(
        sku_name=sku_name,
        domain=domain,
        curr_bt=curr_bt,
        curr_gk=curr_gk,
        status_to_check=status_to_check,
        suggest_fn=suggest_fn,
    )


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
            "category": category,
        },
        "stage1_nlp": {},
        "stage2_candidate_retrieval": {},
        "stage3_cross_encoder": {},
        "stage4_logic_gates": [],
        "stage5_matcher_result": {},
        "stage6_escalation": {},
        "stage7_template_enrichment": {},
        "stage8_rules_engine": {},
        "final_output": {},
    }

    effective_sku_name = sku_name
    food_suffix_added = ""

    clean_input = normalize_sku_text(standardize_sku_units(sku_name))
    input_no_weights = strip_weight_tokens(clean_input)
    input_w_data = extract_weight_feature(clean_input)

    if pipeline is None and task != "classifier":
        try:
            pipeline = get_pipeline(domain)
        except Exception:
            pipeline = None

    full_ner_input = build_ner_input_text(sku_name, description, category)
    ner_text = prepare_ner_text(full_ner_input)
    ner_engine = getattr(pipeline, "ner", None) if pipeline else None
    if ner_engine is None and classifier and getattr(classifier, "ner_engine", None):
        ner_engine = classifier.ner_engine
    if ner_engine is None:
        try:
            ner_engine = get_ner_engine(domain)
        except Exception:
            try:
                _, ner_engine = _get_shared_models()
            except Exception:
                ner_engine = None

    extracted_entities = (
        ner_engine.extract_entities(ner_text)
        if ner_engine and hasattr(ner_engine, "extract_entities")
        else {}
    )

    if domain == config.DOMAIN_FOOD and ner_engine:
        flavor_set = extracted_entities.get("flavor", set())
        suffix = resolve_food_mixed_suffix(
            sku_name,
            flavor_set,
            meat_flavors=getattr(ner_engine, "meat_flavors", set()),
            vegetable_flavors=getattr(ner_engine, "vegetable_flavors", set()),
            seafood_flavors=getattr(ner_engine, "seafood_flavors", set()),
        )
        if suffix:
            food_suffix_added = suffix
            effective_sku_name = f"{sku_name} {food_suffix_added}"
            clean_input = normalize_sku_text(standardize_sku_units(effective_sku_name))
            input_no_weights = strip_weight_tokens(clean_input)
            input_w_data = extract_weight_feature(clean_input)
            flavor_set.add(suffix)
            extracted_entities["flavor"] = flavor_set

    audit_data["stage1_nlp"] = {
        "raw_sku": sku_name,
        "clean_input": clean_input,
        "weight_stripped_input": input_no_weights,
        "extracted_weights": {
            "value": input_w_data[0],
            "unit": input_w_data[1],
            "physical_form": input_w_data[2],
        },
        "ner_entities": {
            k: list(v) if isinstance(v, (set, list)) else v for k, v in extracted_entities.items()
        },
        "food_domain_effective_sku": effective_sku_name,
        "food_suffix_added": food_suffix_added,
    }

    if task == "classifier":
        _audit_classifier_pipeline(
            audit_data,
            domain,
            effective_sku_name,
            description,
            category,
            price,
            classifier=classifier,
            ner_engine=ner_engine,
            suggest_fn=suggest_fn,
        )
        return audit_data

    encoded = pipeline.embedder.embed_weighted_sku(
        [clean_input], [description], [category], weights=config.MATCHER_WEIGHTS
    )
    input_vec_dense = encoded["dense"][0]
    input_vec_bm25 = (encoded.get("bm25") or [{}])[0]
    input_vec_sparse = (encoded.get("sparse") or [None])[0]

    bt_filter = None
    bt_prediction = {}
    if pipeline.classifier:
        try:
            bt_tag, confidence, source, _ = pipeline.classifier.predict_bt(
                input_vec_dense,
                price=price,
                sku_name=effective_sku_name,
                sku_description=description,
            )
            threshold = config.get_bt_confidence_threshold(source)
            applied = confidence >= threshold
            if applied:
                bt_filter = bt_tag
            bt_prediction = {
                "predicted_bt": bt_tag,
                "confidence": float(confidence),
                "source": source,
                "model": getattr(pipeline.classifier, "active_bt_model", "logreg"),
                "threshold": threshold,
                "filter_applied": applied,
            }
        except Exception as e:
            bt_prediction = {"error": str(e)}

    exact_bypass_row = None
    exact_bypass_reason = ""

    if hasattr(pipeline, "exact_match_map") and pipeline.exact_match_map:
        cat_indices = pipeline.exact_match_map.get(clean_input)
        if cat_indices is None and clean_input.endswith(" mixed"):
            cat_indices = pipeline.exact_match_map.get(clean_input[:-6])

        if cat_indices is not None:
            if isinstance(cat_indices, int):
                cat_indices = [cat_indices]

            consensus_indices = resolve_bt_consensus_indices(
                pipeline.raw_catalog,
                cat_indices,
                conf_bt=bt_filter,
                cat_bts_lower=getattr(pipeline, "_catalog_bts_lower", None),
            )
            selected_idx = consensus_indices[0] if consensus_indices else None

            if selected_idx is not None:
                exact_bypass_row = pipeline.raw_catalog.iloc[selected_idx]
                exact_bypass_reason = "Exact Text Match"

    if (
        exact_bypass_row is None
        and hasattr(pipeline, "token_sorted_map")
        and pipeline.token_sorted_map
    ):
        sorted_input_tokens = " ".join(sorted(str(input_no_weights).split()))
        cand_indices = pipeline.token_sorted_map.get(sorted_input_tokens)
        if cand_indices:
            if isinstance(cand_indices, int):
                cand_indices = [cand_indices]

            valid_cands = resolve_bt_consensus_indices(
                pipeline.raw_catalog,
                cand_indices,
                conf_bt=bt_filter,
                cat_bts_lower=getattr(pipeline, "_catalog_bts_lower", None),
            )

            if valid_cands:
                selected_idx, weight_reason = resolve_weight_bypass_candidate(
                    pipeline.raw_catalog, valid_cands, input_w_data
                )
                exact_bypass_row = pipeline.raw_catalog.iloc[selected_idx]
                exact_bypass_reason = f"Fuzzy Match (100%){weight_reason}"

    filtered_cands = pd.DataFrame()
    search_strategy = "BT-Filtered Hybrid Query" if bt_filter else "Full Hybrid Query"

    if exact_bypass_row is not None:
        search_strategy = f"Exact Catalog Lookup ({exact_bypass_reason})"

    if bt_filter:
        filtered_cands = pipeline.search_candidates(
            input_vec_dense,
            input_vec_sparse,
            bt_filter=bt_filter,
            input_vec_bm25=input_vec_bm25,
        )

    unfiltered_cands = pipeline.search_candidates(
        input_vec_dense, input_vec_sparse, input_vec_bm25=input_vec_bm25
    )

    brand_stripped_query = ""
    stripped_candidates_added = 0
    stripped_cands = pd.DataFrame()

    if (
        getattr(config, "ENABLE_BRAND_STRIPPED_SEARCH", True)
        and extracted_entities
        and any(extracted_entities.get("brand", []))
    ):
        brand_stripped = clean_input
        for b in extracted_entities.get("brand", []):
            pattern = re.compile(r"\b" + re.escape(b) + r"\b", re.IGNORECASE)
            brand_stripped = pattern.sub("", brand_stripped)
        brand_stripped = " ".join(brand_stripped.split()).strip()

        if brand_stripped:
            brand_stripped_query = brand_stripped
            try:
                stripped_encoded = pipeline.embedder.embed_weighted_sku(
                    [brand_stripped], [description], [category], weights=config.MATCHER_WEIGHTS
                )
                stripped_cands = pipeline.search_candidates(
                    stripped_encoded["dense"][0],
                    (stripped_encoded.get("sparse") or [None])[0],
                    input_vec_bm25=(stripped_encoded.get("bm25") or [{}])[0],
                )
                if not stripped_cands.empty:
                    stripped_candidates_added = len(stripped_cands)
            except Exception:
                pass

    all_cands_records = []
    for df_pool in (filtered_cands, unfiltered_cands, stripped_cands):
        if not df_pool.empty:
            all_cands_records.extend(df_pool.to_dict(orient="records"))

    all_cands_records.sort(
        key=lambda c: (
            -round(float(c.get("_qdrant_score_", 0.0) or 0.0), 6),
            str(c.get("clean_text", "")),
            format_catalog_row_for_reranker(c, domain=domain),
            str(c.get("basictype", c.get("BasicType", ""))),
            str(c.get("db_uid", "")),
        )
    )

    seen_cand_keys = set()
    seen_formatted = set()
    rerank_records = []
    rerank_limit = getattr(config, "MATCHER_RERANK_CANDIDATES", 18)
    for cand_row in all_cands_records:
        cand_key = cand_row.get("db_uid") or cand_row.get("clean_text", "")
        if cand_key in seen_cand_keys:
            continue
        formatted_cand = format_catalog_row_for_reranker(cand_row, domain=domain)
        if formatted_cand not in seen_formatted and len(seen_formatted) >= rerank_limit:
            continue
        seen_formatted.add(formatted_cand)
        seen_cand_keys.add(cand_key)
        rerank_records.append(cand_row)

    candidates = pd.DataFrame(rerank_records) if rerank_records else pd.DataFrame()

    audit_data["stage2_candidate_retrieval"] = {
        "classifier_bt_prediction": bt_prediction,
        "bt_filter_used": bt_filter,
        "search_strategy": search_strategy,
        "brand_stripped_query": brand_stripped_query,
        "stripped_candidates_added": stripped_candidates_added,
        "total_candidates_found": len(candidates),
    }

    if candidates.empty and exact_bypass_row is None:
        audit_data["final_output"] = {
            "matched_catalog_name": "",
            "score": 0.0,
            "status": "Low / Rejected",
            "logic_notes": "No candidates found in vector store",
            "suggested_bt": "",
            "suggested_gk": "",
            "suggested_region": "",
        }
        return audit_data

    in_bf_key = "flavor" if domain == config.DOMAIN_FOOD else "brand"
    in_brand_or_flavor = ", ".join(
        sorted(str(x) for x in (extracted_entities or {}).get(in_bf_key, []) if x)
    )
    formatted_query = format_matcher_rerank_text(
        name=input_no_weights,
        seller_category=category,
        brand_or_flavor=in_brand_or_flavor,
        description=description,
        domain=domain,
    )

    score_by_key = {}
    sorted_indices = np.array([], dtype=int)
    if rerank_records:
        pairs = [
            [
                formatted_query,
                format_catalog_row_for_reranker(cand_row, domain=domain),
            ]
            for cand_row in rerank_records
        ]

        rerank_model = getattr(pipeline, "reranker", None) or pipeline.embedder
        cross_scores = rerank_model.score_cross_encoder(pairs)
        for cand_row, sc in zip(rerank_records, cross_scores):
            cand_key = cand_row.get("db_uid") or cand_row.get("clean_text", "")
            score_by_key[cand_key] = float(sc)

        sorted_indices = np.argsort(cross_scores)[::-1]
        candidates["raw_cross_score"] = cross_scores
        candidates["final_cross_score"] = cross_scores

    top_candidates_trace = []
    if exact_bypass_row is not None:
        top_candidates_trace.append(
            {
                "cand_idx": 0,
                "catalog_name": str(exact_bypass_row["Name"]),
                "description": str(
                    exact_bypass_row.get("Description", exact_bypass_row.get("description", ""))
                ),
                "seller_category": str(
                    exact_bypass_row.get("SellerCategory")
                    or exact_bypass_row.get("Seller_Category")
                    or exact_bypass_row.get("category")
                    or exact_bypass_row.get("region")
                    or ""
                ),
                "brand": str(exact_bypass_row.get("Brand", exact_bypass_row.get("brand", ""))),
                "basic_type": str(
                    exact_bypass_row.get("basictype", exact_bypass_row.get("BasicType", ""))
                ),
                "generic_keywords": str(
                    exact_bypass_row.get(
                        "Generic keywords", exact_bypass_row.get("GenericKeywords", "")
                    )
                ),
                "region_category": str(
                    exact_bypass_row.get("region") or exact_bypass_row.get("category") or ""
                ),
                "raw_cross_score": 100.0,
                "token_penalty": 0.0,
                "brand_boost": 0.0,
                "final_cross_score": 100.0,
            }
        )

    for idx in sorted_indices[:10]:
        row = candidates.iloc[idx]
        if (
            exact_bypass_row is not None
            and str(row["Name"]).strip().lower() == str(exact_bypass_row["Name"]).strip().lower()
        ):
            continue
        top_candidates_trace.append(
            {
                "cand_idx": int(idx),
                "catalog_name": str(row["Name"]),
                "description": str(row.get("Description", row.get("description", ""))),
                "seller_category": str(
                    row.get("SellerCategory")
                    or row.get("Seller_Category")
                    or row.get("category")
                    or row.get("region")
                    or ""
                ),
                "brand": str(row.get("Brand", row.get("brand", ""))),
                "basic_type": str(row.get("basictype", row.get("BasicType", ""))),
                "generic_keywords": str(
                    row.get("Generic keywords", row.get("GenericKeywords", ""))
                ),
                "region_category": str(row.get("region") or row.get("category") or ""),
                "raw_cross_score": float(row["raw_cross_score"]),
                "token_penalty": 0.0,
                "brand_boost": 0.0,
                "final_cross_score": float(row["final_cross_score"]),
            }
        )
    audit_data["stage3_cross_encoder"] = {"top_candidates": top_candidates_trace}

    evaluated_results = []
    best_res = None
    win_reasons = ""
    gate_limit = getattr(config, "MATCHER_LOGIC_GATE_CANDIDATES", 5)

    def eval_cand_df(df_pool: pd.DataFrame) -> list[dict]:
        if df_pool.empty:
            return []

        pool_records = df_pool.to_dict(orient="records")
        scored_pool = []
        for cand_row in pool_records:
            cand_key = cand_row.get("db_uid") or cand_row.get("clean_text", "")
            raw_score = float(score_by_key.get(cand_key, -10.0))
            scored_pool.append((raw_score, cand_row))

        scored_pool.sort(
            key=lambda item: (
                -float(np.round(item[0], 3)),
                str(item[1].get("clean_text", "")),
                str(item[1].get("db_uid", "")),
            )
        )

        res_list = []
        for raw_score, cand_row in scored_pool[:gate_limit]:
            cand_row_copy = dict(cand_row)
            gate_score, gate_status, gate_reasons = pipeline.rules.apply_logic_gates(
                clean_input,
                extracted_entities,
                cand_row_copy,
                raw_score,
                price,
                input_w_data,
                input_no_weights,
                domain=domain,
                input_description=description,
                input_category=category,
                predicted_bt=bt_filter or "",
            )
            res_list.append(
                {
                    "candidate_name": str(cand_row_copy["Name"]),
                    "clean_text": str(
                        cand_row_copy.get("clean_text", cand_row_copy.get("Name", ""))
                    ),
                    "raw_cross_score": raw_score,
                    "computed_score": float(gate_score),
                    "status": gate_status,
                    "fuzzy_bypass": gate_reasons.startswith("Whole-SKU Fuzzy Match"),
                    "reasons": gate_reasons,
                    "cand_row": cand_row_copy,
                }
            )
        return res_list

    def _pick_best(eval_list: list[dict]) -> dict | None:
        best = None
        for item in eval_list:
            if is_better_gated_candidate(
                item["computed_score"],
                item["raw_cross_score"],
                item["clean_text"],
                best["computed_score"] if best is not None else None,
                best["raw_cross_score"] if best is not None else None,
                best["clean_text"] if best is not None else None,
            ):
                best = item
        return best

    if exact_bypass_row is not None:
        best_res = {
            "candidate_name": str(exact_bypass_row["Name"]),
            "clean_text": str(exact_bypass_row.get("clean_text", exact_bypass_row["Name"])),
            "raw_cross_score": 100.0,
            "computed_score": 1.0,
            "status": "High Confidence",
            "fuzzy_bypass": True,
            "reasons": exact_bypass_reason,
            "cand_row": exact_bypass_row,
        }
        evaluated_results.append({**best_res, "rank": 1})
        win_reasons = exact_bypass_reason

    elif bt_filter and not filtered_cands.empty:
        filtered_eval = eval_cand_df(filtered_cands)
        fallback_res = _pick_best(filtered_eval)
        for rank_i, item in enumerate(
            sorted(filtered_eval, key=lambda x: x["computed_score"], reverse=True)
        ):
            evaluated_results.append({**item, "rank": rank_i + 1})

        if fallback_res and fallback_res["status"] == "High Confidence":
            best_res = fallback_res
            win_reasons = f"BT-Filtered ({bt_filter}) | {best_res['reasons']}"
        else:
            unfiltered_pool = (
                pd.concat([unfiltered_cands, stripped_cands], ignore_index=True).drop_duplicates(
                    subset=["clean_text"]
                )
                if (not unfiltered_cands.empty or not stripped_cands.empty)
                else pd.DataFrame()
            )
            unfiltered_eval = eval_cand_df(unfiltered_pool)

            for item in sorted(unfiltered_eval, key=lambda x: x["computed_score"], reverse=True):
                if not any(
                    r["candidate_name"] == item["candidate_name"] for r in evaluated_results
                ):
                    evaluated_results.append({**item, "rank": len(evaluated_results) + 1})

            best_uf = _pick_best(unfiltered_eval)

            if fallback_res is not None and (
                best_uf is None or fallback_res["computed_score"] > best_uf["computed_score"]
            ):
                best_res = fallback_res
                win_reasons = f"BT-Filtered ({bt_filter}) [Low Confidence Fallback] | {fallback_res['reasons']}"
            elif best_uf is not None:
                best_res = best_uf
                win_reasons = f"BT-Fallback (BT: {bt_filter}, score: {best_uf['computed_score']:.4f}) | {best_uf['reasons']}"
    else:
        unfiltered_pool = (
            pd.concat([unfiltered_cands, stripped_cands], ignore_index=True).drop_duplicates(
                subset=["clean_text"]
            )
            if (not unfiltered_cands.empty or not stripped_cands.empty)
            else pd.DataFrame()
        )
        unfiltered_eval = eval_cand_df(unfiltered_pool)
        for rank_i, item in enumerate(
            sorted(unfiltered_eval, key=lambda x: x["computed_score"], reverse=True)
        ):
            evaluated_results.append({**item, "rank": rank_i + 1})

        best_res = _pick_best(unfiltered_eval)
        if best_res:
            if bt_filter:
                win_reasons = f"BT-Fallback (BT: {bt_filter}, score: {best_res['computed_score']:.4f}) | {best_res['reasons']}"
            else:
                win_reasons = best_res["reasons"]

    winner_name = best_res["candidate_name"] if best_res else ""

    for res in evaluated_results:
        if res["candidate_name"] == winner_name:
            res["win_status_note"] = "WINNER (Selected as best matching catalog item)"
        elif res["status"] == "Rejected":
            res["win_status_note"] = f"DID NOT WIN: Rejected by logic gate logic ({res['reasons']})"
        else:
            diff = best_res["computed_score"] - res["computed_score"]
            if abs(diff) < 1e-4:
                raw_diff = best_res["raw_cross_score"] - res["raw_cross_score"]
                res["win_status_note"] = (
                    f"DID NOT WIN: Tied computed score ({res['computed_score']:.1f}), lost tie-breaker on raw cross-encoder score by {raw_diff:.4f} pts ({res['reasons']})"
                )
            else:
                res["win_status_note"] = (
                    f"DID NOT WIN: Lower computed score by {diff:.4f} pts ({res['reasons']})"
                )

    audit_data["stage4_logic_gates"] = [
        {k: v for k, v in res.items() if k != "cand_row"} for res in evaluated_results
    ]

    if best_res:
        winner = best_res["cand_row"]
        win_score = best_res["computed_score"]
        win_status = best_res["status"]
        if not win_reasons:
            win_reasons = best_res["reasons"]
    else:
        winner = candidates.iloc[0]
        win_score = 0.0
        win_status = "Low / Rejected"
        win_reasons = "No candidates passed gates"

    orig_gk = str(winner.get("Generic keywords", winner.get("GenericKeywords", "")))
    winner_gk = filter_matched_generic_keywords(
        match_gk=orig_gk,
        domain=domain,
        clean_input=clean_input,
        input_entities=extracted_entities,
        winner_row=winner,
        rules=getattr(pipeline, "rules", None),
        flavor_categories=getattr(pipeline, "flavor_categories", None),
        seafood_pattern=getattr(pipeline, "_seafood_pattern", None),
        mixed_pattern=getattr(pipeline, "_mixed_pattern", None),
        veg_pattern=getattr(pipeline, "_veg_pattern", None),
        specific_flavor_pattern=getattr(pipeline, "_specific_flavor_pattern", None),
    )

    def _clean_val(row, key):
        if key not in row:
            return None
        v = row[key]
        if pd.isna(v):
            return None
        s = str(v).strip()
        return s if s else None

    region_val = (
        _clean_val(winner, "region")
        or _clean_val(winner, "Region")
        or _clean_val(winner, "category")
        or _clean_val(winner, "Category")
        or _clean_val(winner, "Categories")
        or ""
    )

    audit_data["stage5_matcher_result"] = {
        "matched_catalog_name": str(winner["Name"]),
        "score": float(win_score),
        "status": win_status,
        "logic_notes": win_reasons,
        "suggested_bt": str(winner.get("basictype", winner.get("BasicType", ""))),
        "original_gk": orig_gk,
        "filtered_gk": winner_gk,
        "suggested_region": region_val,
    }

    if task == "matcher":
        matcher_bt = str(winner.get("basictype", winner.get("BasicType", "")))
        matcher_bt, matcher_gk, _, _, _, tmpl_details = _run_audit_template_enrichment(
            effective_sku_name,
            domain,
            matcher_bt,
            winner_gk,
            win_status,
            suggest_fn=suggest_fn,
        )
        audit_data["stage7_template_enrichment"] = tmpl_details
        audit_data["final_output"] = {
            "matched_catalog_name": str(winner["Name"]),
            "score": float(win_score),
            "status": win_status,
            "logic_notes": win_reasons,
            "suggested_bt": matcher_bt,
            "suggested_gk": matcher_gk,
            "suggested_region": region_val,
            "pipeline_source": "Matcher",
            "escalated": False,
            "rules_applied": [],
        }
        return audit_data

    curr_bt = str(winner.get("basictype", winner.get("BasicType", "")))
    curr_gk = winner_gk
    curr_region = region_val
    pipeline_source = "Matcher"
    escalated = False

    if win_status in config.PIPELINE_ESCALATE_STATUSES:
        escalated = True
        classifier = pipeline.classifier
        if classifier:
            try:
                esc_embs = pipeline.embedder.embed_weighted_sku(
                    [effective_sku_name],
                    [description],
                    [category],
                    weights=config.CLASSIFIER_WEIGHTS,
                )
                query_embeddings = [
                    {
                        "dense": esc_embs["dense"][0],
                        "bm25": (esc_embs.get("bm25") or [{}])[0],
                        "sparse": (esc_embs.get("sparse") or [None])[0],
                    }
                ]

                rerank_model = getattr(pipeline, "reranker", None) or _get_reranker_model(
                    pipeline.embedder
                )
                reranker = (
                    rerank_model
                    if (rerank_model is not None and getattr(rerank_model, "cross_session", None))
                    else None
                )

                clf_results = tag_all_skus(
                    sku_names=[effective_sku_name],
                    sku_categories=[category],
                    query_embeddings=query_embeddings,
                    vector_store=pipeline.vector_store,
                    reranker=reranker,
                    classifier=classifier,
                    sku_descriptions=[description],
                    sku_prices=[price or 0.0],
                    embed_engine=pipeline.embedder,
                    ner_engine=getattr(classifier, "ner_engine", pipeline.ner),
                )

                clf = clf_results[0]
                bt_tag = str(clf.get("suggested_bt", "") or "").strip()
                bt_conf = float(clf.get("bt_confidence", 0.0) or 0.0) if bt_tag else 0.0
                gk_str = str(clf.get("suggested_gk", ""))
                gk_conf = float(clf.get("gk_confidence", 0.0))
                tt_val = str(clf.get("suggested_region", clf.get("suggested_category", "")))
                tt_conf = float(clf.get("region_confidence", clf.get("category_confidence", 0.0)))

                scale_factor = (
                    0.75 / max(float(config.CONFIDENCE_THRESHOLD_HIGH), 0.01)
                    if float(config.CONFIDENCE_THRESHOLD_HIGH) < 0.75
                    else 1.0
                )
                matcher_norm = min(float(win_score) * scale_factor, 1.0)
                matcher_bt_confirmed = "BasicType Match (" in str(
                    win_reasons or ""
                ) and win_status not in ("Low / Rejected", "Rejected")
                classifier_won = (
                    bool(bt_tag) and not matcher_bt_confirmed and bt_conf > matcher_norm
                )
                pipeline_source = "Classifier" if classifier_won else "Matcher"

                if classifier_won:
                    same_bt_with_catalog_gk = (
                        bool(curr_bt and curr_gk)
                        and curr_bt.strip().lower() == bt_tag.strip().lower()
                    )
                    curr_bt = bt_tag
                    if not same_bt_with_catalog_gk:
                        curr_gk = gk_str
                    curr_region = tt_val

                clf_bt_status = str(clf.get("bt_status", ""))
                audit_data["stage6_escalation"] = {
                    "escalated": True,
                    "escalation_reason": f"Matcher status '{win_status}' is in PIPELINE_ESCALATE_STATUSES",
                    "classifier_bt": bt_tag,
                    "classifier_bt_confidence": float(bt_conf),
                    "classifier_bt_status": clf_bt_status,
                    "classifier_gk": gk_str,
                    "classifier_gk_confidence": float(gk_conf),
                    "classifier_region": tt_val,
                    "classifier_region_confidence": float(tt_conf),
                    "matcher_normalized_score": float(matcher_norm),
                    "escalation_winner": pipeline_source,
                }
            except Exception as e:
                audit_data["stage6_escalation"] = {"escalated": True, "error": str(e)}
        else:
            audit_data["stage6_escalation"] = {"escalated": True, "note": "No classifier loaded"}
    else:
        audit_data["stage6_escalation"] = {
            "escalated": False,
            "note": f"Matcher status '{win_status}' does not trigger escalation",
        }

    esc_clf_status = (
        str(audit_data.get("stage6_escalation", {}).get("classifier_bt_status", ""))
        if (escalated and pipeline_source == "Classifier")
        else ""
    )
    status_to_check = esc_clf_status or win_status
    curr_bt, curr_gk, template_applied, _, _, template_details = _run_audit_template_enrichment(
        effective_sku_name,
        domain,
        curr_bt,
        curr_gk,
        status_to_check,
        suggest_fn=suggest_fn,
    )

    audit_data["stage7_template_enrichment"] = template_details

    gk_list = [x.strip() for x in str(curr_gk).split(",") if x.strip()]
    clf_conf = (
        float(audit_data.get("stage6_escalation", {}).get("classifier_bt_confidence", 0.0))
        if escalated
        else 0.0
    )
    rules_confidence = max(win_score, clf_conf, 0.95 if template_applied else 0.0)

    record = {
        "sku_name": effective_sku_name,
        "domain": domain,
        "bt": curr_bt,
        "gk": gk_list,
        "region": curr_region if domain == config.DOMAIN_FOOD else None,
        "category": curr_region if domain == config.DOMAIN_MARKET else None,
        "price": price,
        "confidence": rules_confidence,
        "match_source": "Classifier" if pipeline_source == "Classifier" else "Matcher",
        "matched_sku": str(winner["Name"]),
        "reasoning": win_reasons,
    }

    aug_record = run_rules_engine(record)
    final_bt = str(aug_record.get("bt") or "")
    final_gk = (
        ", ".join(aug_record.get("gk", []))
        if isinstance(aug_record.get("gk"), list)
        else str(aug_record.get("gk") or "")
    )
    final_region = str(aug_record.get("region") or aug_record.get("category") or "")
    rules_applied = aug_record.get("rules_applied", [])

    audit_data["stage8_rules_engine"] = {
        "rules_applied_count": len(rules_applied),
        "rules_applied": rules_applied,
        "pre_rules_tags": {"bt": curr_bt, "gk": curr_gk, "region": curr_region},
        "post_rules_tags": {"bt": final_bt, "gk": final_gk, "region": final_region},
    }

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
        "rules_applied": rules_applied,
    }

    return audit_data


def _audit_classifier_pipeline(
    audit_data: dict,
    domain: str,
    sku_name: str,
    description: str,
    category: str,
    price: float,
    classifier=None,
    embed_engine=None,
    ner_engine=None,
    vector_store=None,
    suggest_fn=None,
) -> dict:
    """Executes a deep multi-stage diagnostic audit for classifier-only task."""
    if classifier is None:
        try:
            classifier = get_classifier(domain)
        except Exception:
            classifier = None

    if embed_engine is None:
        try:
            e_mod, _ = _get_shared_models()
            embed_engine = e_mod
        except Exception:
            pass

    if ner_engine is None:
        if classifier and getattr(classifier, "ner_engine", None):
            ner_engine = classifier.ner_engine
        else:
            try:
                ner_engine = get_ner_engine(domain)
            except Exception:
                pass

    if vector_store is None:
        try:
            vector_store = _get_vector_store()
        except Exception:
            pass

    query_embs = embed_engine.embed_weighted_sku(
        [sku_name], [description], [category], weights=config.CLASSIFIER_WEIGHTS
    )
    vec_dense = query_embs["dense"][0]
    vec_bm25 = (query_embs.get("bm25") or [{}])[0]
    vec_sparse = (query_embs.get("sparse") or [None])[0]
    vec_2d = vec_dense.reshape(1, -1)

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

    bt_candidates = []
    predicted_bt = ""
    bt_conf = 0.0
    bt_source = "zero-shot"
    p_val = float(price) if price is not None else 0.0

    if (
        getattr(classifier, "bt_model", "") == "arcface"
        and getattr(classifier, "_arcface_session", None) is not None
    ):
        proba = classifier._get_arcface_model().predict_probas(vec_2d, [p_val])[0]
        top_indices = np.argsort(proba)[::-1][:10]
        for idx in top_indices:
            bt_candidates.append(
                {
                    "bt": str(classifier._arcface_classes[idx]),
                    "score": float(proba[idx]),
                    "source": "trained",
                }
            )
        best_idx = int(np.argmax(proba))
        best_prob = float(proba[best_idx])
        if best_prob >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
            predicted_bt = str(classifier._arcface_classes[best_idx])
            bt_conf = best_prob
            bt_source = "trained"
    elif (
        getattr(classifier, "bt_model", "") == "logreg"
        and classifier._trained
        and getattr(classifier, "_bt_clf", None) is not None
    ):
        scaled_p = classifier._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])
        proba = classifier._bt_clf.predict_proba(vec_with_price)[0]
        top_indices = np.argsort(proba)[::-1][:10]
        for idx in top_indices:
            bt_candidates.append(
                {
                    "bt": str(classifier._bt_enc.classes_[idx]),
                    "score": float(proba[idx]),
                    "source": "trained",
                }
            )
        best_idx = int(np.argmax(proba))
        best_prob = float(proba[best_idx])
        if best_prob >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
            predicted_bt = str(classifier._bt_enc.classes_[best_idx])
            bt_conf = best_prob
            bt_source = "trained"

    cold_router = getattr(classifier, "cold_start_router", None)
    if not predicted_bt and cold_router is not None:
        r_tag, r_conf, r_src, _ = cold_router.route_single(
            vec_2d[0], sku_name=sku_name, sku_description=description, price=price
        )
        if r_tag and r_conf >= config.BT_ZERO_SHOT_CONFIDENCE_THRESHOLD:
            predicted_bt = r_tag
            bt_conf = r_conf
            bt_source = r_src

    if not predicted_bt and getattr(classifier, "bt_labels", None):
        scores_pure = (vec_2d @ classifier.bt_embs_pure.T)[0]
        scores_desc = (vec_2d @ classifier.bt_embs_desc.T)[0]
        scores = np.maximum(scores_pure, scores_desc)
        top_indices = np.argsort(scores)[::-1][:10]
        if not bt_candidates:
            for idx in top_indices:
                bt_candidates.append(
                    {
                        "bt": str(classifier.bt_labels[idx]),
                        "score": float(scores[idx]),
                        "source": "zero-shot",
                    }
                )
        best_idx = int(np.argmax(scores))
        predicted_bt = str(classifier.bt_labels[best_idx])
        bt_conf = float(scores[best_idx])
        bt_source = "zero-shot"

    flavor_conflict = False
    conflict_notes = ""
    if extracted_flavors and predicted_bt and hasattr(classifier, "food_flavors_dict"):
        bt_flavors = extract_canonical_flavors(predicted_bt, classifier.food_flavors_dict)
        input_flavors = set()
        for f in extracted_flavors:
            input_flavors.update(extract_canonical_flavors(f, classifier.food_flavors_dict))
        if bt_flavors and input_flavors and bt_flavors.isdisjoint(input_flavors):
            flavor_conflict = True
            conflict_notes = f"Flavor conflict: BT '{predicted_bt}' flavors {list(bt_flavors)} disjoint from input flavors {list(input_flavors)}"
            predicted_bt = ""
            bt_conf = 0.0
            bt_source = "conflict"

    bt_status = get_status(bt_conf, bool(predicted_bt), bt_source)

    audit_data["stage2_bt_classification"] = {
        "predicted_bt": predicted_bt,
        "confidence": bt_conf,
        "source": bt_source,
        "model": getattr(classifier, "active_bt_model", "logreg"),
        "status": bt_status,
        "threshold": config.get_bt_confidence_threshold(bt_source),
        "flavor_conflict": flavor_conflict,
        "conflict_notes": conflict_notes,
        "top_candidates": bt_candidates,
    }

    strategy = get_strategy(classifier.domain)

    guaranteed = list(classifier.get_guaranteed_gk(predicted_bt))
    bt_gk_map = getattr(classifier, "bt_gk_map", {})
    allowed_gks_for_bt = bt_gk_map.get(predicted_bt, [])
    allowed_gks_lower = {kw.lower().strip() for kw in allowed_gks_for_bt}

    if predicted_bt and predicted_bt not in guaranteed:
        bt_lower = predicted_bt.lower().strip()
        all_known_gks_lower = {
            gk.lower().strip() for gk_list in bt_gk_map.values() for gk in gk_list
        }
        if bt_lower in all_known_gks_lower:
            guaranteed.append(predicted_bt)

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

    trained_gk = []
    trained_conf = 0.0
    trained_gk_trace = []
    if classifier._trained and getattr(classifier, "_gk_clf", None) is not None:
        scaled_p = classifier._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])
        gk_proba = classifier._gk_clf.predict_proba(vec_with_price)[0]
        threshold = config.GK_TRAINED_CONFIDENCE_THRESHOLD
        predicted_indices = np.where(gk_proba >= threshold)[0]
        if len(predicted_indices) > 0:
            trained_gk = classifier._gk_enc.classes_[predicted_indices].tolist()
            trained_conf = float(np.mean(gk_proba[predicted_indices]))
        for idx in np.argsort(gk_proba)[::-1][:15]:
            prob = float(gk_proba[idx])
            if prob >= 0.05:
                trained_gk_trace.append(
                    {
                        "tag": str(classifier._gk_enc.classes_[idx]),
                        "score": prob,
                        "threshold_passed": bool(prob >= threshold),
                    }
                )

    trained_gk_source = "trained" if trained_gk else "zero-shot"
    search_allowed_tags, early_return = strategy.filter_search_tags(
        allowed_gks_for_bt, trained_conf, trained_gk_source, guaranteed
    )

    dense_hits, bm25_hits, sparse_hits = [], [], []
    fused_candidates = []
    scored_candidates = []
    trained_gk_clean = []
    pruned_by_flavor = []
    synth_added = 0
    guaranteed_clean = list(guaranteed)
    trained_gk_final = []

    if early_return is not None:
        merged_gk = list(early_return[0])
        gk_conf, gk_status, _ = resolve_gk_confidence_and_status(
            trained_conf, trained_gk_source, float(early_return[1]), False, bool(merged_gk)
        )
    else:
        if vector_store is not None:
            dense_hits, bm25_hits, sparse_hits = vector_store.search_tags_hybrid(
                dense_query=vec_dense,
                sparse_query=vec_sparse,
                limit=config.TAG_SEARCH_LIMIT,
                filter_dict_type="gk",
                domain=classifier.domain,
                allowed_tags=search_allowed_tags,
                bm25_query=vec_bm25,
            )
            if config.FUSION_METHOD == "rrf":
                fused_candidates = _rrf_fusion(dense_hits, bm25_hits, sparse_hits)
            else:
                fused_candidates = _weighted_fusion(dense_hits, bm25_hits, sparse_hits)

        top_candidates = fused_candidates[: config.TOP_K_RETRIEVAL]
        top_candidates = strategy.apply_post_search_filters(top_candidates, allowed_gks_lower)

        synth_before = len(top_candidates)
        top_candidates = strategy.inject_synthetic_tags(
            top_candidates, market_brand, predicted_bt, category, extracted_flavors
        )
        synth_added = len(top_candidates) - synth_before

        ner_engine_clf = ner_engine or getattr(classifier, "ner_engine", None)
        guaranteed_clean, trained_gk_clean, top_candidates_clean = (
            strategy.apply_flavor_leak_filters(
                list(guaranteed),
                list(trained_gk),
                list(top_candidates),
                extracted_flavors,
                ner_engine_clf,
            )
        )

        top_cands_clean_tags = {c["tag"].lower().strip() for c in top_candidates_clean}
        for c in top_candidates:
            if c["tag"].lower().strip() not in top_cands_clean_tags:
                pruned_by_flavor.append(c["tag"])

        rerank_model = _get_reranker_model(embed_engine)
        has_rerank = rerank_model is not None and getattr(rerank_model, "cross_session", None)
        reranked_tags = []
        final_conf = 0.0
        used_reranker = bool(top_candidates_clean and config.USE_RERANKER and has_rerank)

        if used_reranker:
            query_text = (sku_name + " " + description).strip()
            pairs = [[query_text, c["tag"]] for c in top_candidates_clean]
            raw_scores = rerank_model.score_cross_encoder(pairs)
            if isinstance(raw_scores, float) or (
                hasattr(raw_scores, "item") and raw_scores.ndim == 0
            ):
                scores = [float(raw_scores)]
            else:
                scores = [float(s) for s in raw_scores]
            for c, score in zip(top_candidates_clean, scores):
                tag_lower = c["tag"].lower().strip()
                passed = strategy.get_reranker_threshold(score, tag_lower, allowed_gks_lower)
                scored_candidates.append(
                    {
                        "tag": c["tag"],
                        "score": score,
                        "threshold_passed": bool(passed),
                        "source": c.get("source", "hybrid_search"),
                    }
                )
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            if scored_candidates:
                top_score = scored_candidates[0]["score"]
                margin_cutoff = max(config.RERANKER_THRESHOLD, top_score - config.RERANKER_MARGIN)
                max_reranked = int(getattr(config, "MAX_RERANKED_GK_TAGS", 5) or 0)
                kept_reranked = 0
                for item in scored_candidates:
                    if item["threshold_passed"] and item["score"] < margin_cutoff:
                        item["threshold_passed"] = False
                        item["prune_reason_override"] = (
                            f"Reranker score ({item['score']:.3f}) < margin cutoff ({margin_cutoff:.3f})"
                        )
                    elif item["threshold_passed"]:
                        kept_reranked += 1
                        if max_reranked > 0 and kept_reranked > max_reranked:
                            item["threshold_passed"] = False
                            item["prune_reason_override"] = (
                                f"Exceeded top-{max_reranked} reranked GK cap"
                            )

            reranked_tags = [item["tag"] for item in scored_candidates if item["threshold_passed"]]
            if scored_candidates:
                final_conf = scored_candidates[0]["score"]
        else:
            reranked_tags = [c["tag"] for c in top_candidates_clean]
            if top_candidates_clean:
                final_conf = min(1.0, top_candidates_clean[0]["score"])

        trained_gk_final = strategy.filter_final_trained_gk(trained_gk_clean, allowed_gks_lower)
        merged_gk = []
        seen = set()
        for tag in guaranteed_clean + trained_gk_final + reranked_tags:
            key = tag.lower().strip()
            if key not in seen:
                seen.add(key)
                merged_gk.append(tag)

        gk_conf, gk_status, _ = resolve_gk_confidence_and_status(
            trained_conf, trained_gk_source, final_conf, used_reranker, bool(merged_gk)
        )

    all_considered_map = {}

    for g_tag in guaranteed:
        key = g_tag.lower().strip()
        all_considered_map[key] = {
            "tag": g_tag,
            "source": "Guaranteed / Schema"
            if g_tag not in literal_matched_gks
            else "Literal SKU Match",
            "base_score": 1.0,
            "reranker_score": None,
            "threshold_passed": True,
            "flavor_filter_passed": g_tag in guaranteed_clean,
            "prune_reason": None
            if g_tag in guaranteed_clean
            else "Flavor conflict with SKU entities",
            "is_selected": g_tag in merged_gk,
        }

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
                "prune_reason": None
                if is_kept
                else (
                    "Probability < 0.50"
                    if not t_item["threshold_passed"]
                    else "Filtered by schema / flavor"
                ),
                "is_selected": t_item["tag"] in merged_gk,
            }

    for s_item in scored_candidates:
        key = s_item["tag"].lower().strip()
        if key not in all_considered_map:
            is_kept = s_item["tag"] in merged_gk
            all_considered_map[key] = {
                "tag": s_item["tag"],
                "source": "Synthetic Injected"
                if s_item["source"] == "synthetic"
                else "Vector Hybrid Search",
                "base_score": None,
                "reranker_score": s_item["score"],
                "threshold_passed": s_item["threshold_passed"],
                "flavor_filter_passed": True,
                "prune_reason": None
                if s_item["threshold_passed"]
                else s_item.get(
                    "prune_reason_override",
                    f"Reranker score ({s_item['score']:.3f}) < threshold ({config.RERANKER_THRESHOLD})",
                ),
                "is_selected": is_kept,
            }
        else:
            all_considered_map[key]["reranker_score"] = s_item["score"]

    for f_item in fused_candidates[: config.TOP_K_RETRIEVAL]:
        key = f_item["tag"].lower().strip()
        if key not in all_considered_map:
            pruned_reason = (
                "Flavor conflict"
                if f_item["tag"] in pruned_by_flavor
                else "Filtered by schema mapping"
            )
            all_considered_map[key] = {
                "tag": f_item["tag"],
                "source": "Vector Hybrid Search",
                "base_score": f_item["score"],
                "reranker_score": None,
                "threshold_passed": False,
                "flavor_filter_passed": f_item["tag"] not in pruned_by_flavor,
                "prune_reason": pruned_reason,
                "is_selected": False,
            }

    considered_keywords_list = list(all_considered_map.values())
    considered_keywords_list.sort(
        key=lambda x: (
            not x["is_selected"],
            -(x["reranker_score"] if x["reranker_score"] is not None else (x["base_score"] or 0)),
        )
    )

    audit_data["stage3_gk_classification"] = {
        "suggested_gk": ", ".join(merged_gk),
        "confidence": gk_conf,
        "status": gk_status,
        "guaranteed_tags": guaranteed_clean,
        "trained_gk_tags": trained_gk_final,
        "literal_matched_tags": literal_matched_gks,
        "vector_search_hits_count": len(dense_hits) + len(bm25_hits) + len(sparse_hits),
        "synthetic_tags_injected": synth_added,
        "pruned_by_flavor_leak": pruned_by_flavor,
        "considered_keywords": considered_keywords_list,
    }

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
        third_tag_candidates.append(
            {
                "tag": chosen_third_tag,
                "score": 1.0,
                "source": "override",
                "note": f"Exact catalog override mapped from BT '{predicted_bt}'",
            }
        )
    elif classifier._trained and getattr(classifier, "_third_tag_clf", None) is not None:
        scaled_p = classifier._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])
        tt_proba = classifier._third_tag_clf.predict_proba(vec_with_price)[0]
        for idx in np.argsort(tt_proba)[::-1][:10]:
            third_tag_candidates.append(
                {
                    "tag": str(classifier._third_tag_enc.classes_[idx]),
                    "score": float(tt_proba[idx]),
                    "source": "trained",
                }
            )
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
        if not third_tag_candidates:
            for idx in np.argsort(scores)[::-1][:10]:
                third_tag_candidates.append(
                    {
                        "tag": str(classifier.third_tag_labels[idx]),
                        "score": float(scores[idx]),
                        "source": "zero-shot",
                    }
                )
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
        "top_candidates": third_tag_candidates,
    }

    curr_bt, new_gk_str, tmpl_ok, s_bt, s_gk, template_details = _run_audit_template_enrichment(
        sku_name,
        domain,
        predicted_bt,
        ", ".join(merged_gk),
        bt_status,
        suggest_fn=suggest_fn,
    )
    curr_gk_list = [x.strip() for x in new_gk_str.split(",") if x.strip()] if new_gk_str else []
    if tmpl_ok:
        if s_bt:
            bt_conf = max(bt_conf, 0.95)
            bt_status = "AUTO"
            bt_source = "template"
        if s_gk:
            gk_conf = max(gk_conf, 0.95)
            gk_status = "AUTO"

    audit_data["stage7_template_enrichment"] = template_details

    rules_confidence = max(bt_conf, gk_conf, third_tag_conf)
    record = {
        "sku_name": sku_name,
        "domain": domain,
        "bt": curr_bt,
        "gk": curr_gk_list,
        "region": chosen_third_tag if domain == config.DOMAIN_FOOD else None,
        "category": chosen_third_tag if domain == config.DOMAIN_MARKET else None,
        "price": price,
        "confidence": rules_confidence,
        "match_source": "Classifier",
        "matched_sku": "",
        "reasoning": f"Classifier predicted BT='{curr_bt}', GK='{', '.join(curr_gk_list)}', {tag_label}='{chosen_third_tag}'",
    }

    aug_record = run_rules_engine(record)
    final_bt = str(aug_record.get("bt") or "")
    final_gk = (
        ", ".join(aug_record.get("gk", []))
        if isinstance(aug_record.get("gk"), list)
        else str(aug_record.get("gk") or "")
    )
    final_region = str(aug_record.get("region") or aug_record.get("category") or "")
    rules_applied = aug_record.get("rules_applied", [])

    audit_data["stage5_rules_engine"] = {
        "rules_applied_count": len(rules_applied),
        "rules_applied": rules_applied,
        "pre_rules_tags": {
            "bt": curr_bt,
            "gk": ", ".join(curr_gk_list),
            "region": chosen_third_tag,
        },
        "post_rules_tags": {"bt": final_bt, "gk": final_gk, "region": final_region},
    }

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
        "pipeline_source": "Classifier",
    }
    return audit_data
