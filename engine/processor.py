"""
SKU MatchOps Engine - Request Processor
Executes Matching, Classification, and Pipeline escalation tasks.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from engine import config
from engine.classification.tagger import tag_all_skus
from engine.rules_engine import run_rules_engine
from engine.resource_loader import (
    _get_shared_models,
    _get_vector_store,
    get_pipeline,
    get_classifier,
    check_models_loaded,
)
from engine.template_suggest import suggest_tags_from_template

logger = logging.getLogger("matchops.engine.processor")


class RerankerWrapper:
    """Lightweight wrapper to score candidate pairs with the Cross-Encoder."""
    def __init__(self, embed_engine):
        self.embed_engine = embed_engine

    def predict(self, pairs):
        return self.embed_engine.score_cross_encoder(pairs)


def _build_sku_dataframe(skus: List[Dict[str, Any]]) -> pd.DataFrame:
    """Constructs a standard SKU DataFrame from SKU dictionaries."""
    data = {
        "SKU": [
            str(sku.get("name") or sku.get("sku_raw") or sku.get("Name") or sku.get("SKU") or "").strip()
            for sku in skus
        ],
        "Price": [sku.get("price", 0.0) or 0.0 for sku in skus],
        "Description": [sku.get("description", "") or "" for sku in skus],
        "Category": [sku.get("category", "") or "" for sku in skus],
    }
    return pd.DataFrame(data)


def _clean_field(row, key: str) -> Optional[str]:
    """Helper to safely extract a non-empty string or None from a pandas row."""
    if key not in row:
        return None
    val = row[key]
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _extract_query_embeddings(embed_engine, names: List[str], descriptions: List[str], categories: List[str]):
    """Encodes query texts with the Bi-Encoder using fallback for description/category."""
    safe_descs = [d if d else "" for d in descriptions]
    safe_cats = [c if c else "" for c in categories]
    return embed_engine.embed_weighted_sku(names, safe_descs, safe_cats)


def _apply_template_tag_enrichment(skus: List[Dict[str, Any]], results: List[dict], domain: str, mode: str = "classifier"):
    """Enriches predicted tags using template-based keyword substitution if enabled."""
    if not getattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", False):
        return

    try:
        for i, sku in enumerate(skus):
            if i >= len(results):
                break
            if results[i].get("status") in ["Exact Text Match", "High Confidence"]:
                continue
            sku_name = sku.get("name", "")
            sug_res = suggest_tags_from_template(sku_name, domain=domain, current_bt=results[i].get("suggested_bt"))
            if sug_res.get("matched"):
                s_bt = sug_res.get("suggested_bt", "")
                s_gk_list = sug_res.get("suggested_gk", [])
                s_gk = ", ".join(s_gk_list) if isinstance(s_gk_list, list) else str(s_gk_list)
                if s_bt:
                    results[i]["suggested_bt"] = s_bt
                    if mode == "classifier":
                        results[i]["bt_status"] = "AUTO"
                        results[i]["bt_confidence"] = max(results[i].get("bt_confidence", 0.0), 0.95)
                    elif mode == "pipeline":
                        results[i]["bt_status"] = "High Confidence"
                        results[i]["bt_confidence"] = max(results[i].get("bt_confidence") or 0.0, 0.95)
                if s_gk:
                    results[i]["suggested_gk"] = s_gk
                    if mode == "classifier":
                        results[i]["gk_status"] = "AUTO"
                        results[i]["gk_confidence"] = max(results[i].get("gk_confidence", 0.0), 0.95)
                    elif mode == "pipeline":
                        results[i]["gk_status"] = "High Confidence"
                        results[i]["gk_confidence"] = max(results[i].get("gk_confidence") or 0.0, 0.95)
    except Exception as e:
        logger.warning(f"Template tag enrichment failed for {mode}: {e}")


def process_request(
    skus: List[Dict[str, Any]],
    task: str,
    domain: str = "market",
    job_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float, Optional[int]], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """
    Executes Matching, Classification, or Pipeline for a batch of SKUs.
    Calls progress_callback(stage_name, progress_pct, eta_seconds) dynamically.
    """
    task = task.lower()
    domain = domain or "market"
    check_models_loaded(domain, task)

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise InterruptedError("Job cancelled by user.")

    def emit_progress(stage: str, pct: float, eta: Optional[int] = None):
        if progress_callback:
            progress_callback(stage, min(100.0, max(0.0, round(float(pct), 2))), eta)

    start_time = time.time()
    total_skus = len(skus)

    # Food domain flavor entity enrichment
    if domain == config.DOMAIN_FOOD:
        embed_engine, ner_engine = _get_shared_models()
        if ner_engine:
            sku_texts = [
                f"{sku.get('name', '')} {sku.get('description', '')} {sku.get('category', '')}".strip()
                for sku in skus
            ]
            ner_results = ner_engine.batch_extract_entities(sku_texts)
            for i, sku in enumerate(skus):
                flavor_set = ner_results[i].get("flavor", set())
                name = sku.get("name", "")
                if any(f in getattr(ner_engine, 'vegetable_flavors', set()) for f in flavor_set):
                    name += " veg"
                if any(f in getattr(ner_engine, 'seafood_flavors', set()) for f in flavor_set):
                    name += " seafood"
                meat_count = sum(1 for f in flavor_set if f in getattr(ner_engine, 'meat_flavors', set()) and f != "egg")
                if meat_count >= 2:
                    name += " mixed"
                sku["name"] = name

    if task == "matcher":
        emit_progress("embedding", 5.0)
        matcher = get_pipeline(domain)
        input_df = _build_sku_dataframe(skus)

        def matcher_cb(pct, msg=None):
            check_cancel()
            elapsed = max(0.1, time.time() - start_time)
            p = float(pct)
            if p < 30.0:
                stage = "embedding"
            elif p < 60.0:
                stage = "vector_search"
            elif p < 85.0:
                stage = "reranking"
            else:
                stage = "writing_results"
            eta = max(1, int((elapsed / (p / 100.0)) - elapsed)) if p > 1.0 else None
            emit_progress(stage, p, eta)

        results_df = matcher.process_inputs(input_df, progress_callback=matcher_cb)

        out = []
        for i, sku in enumerate(skus):
            if i < len(results_df):
                row = results_df.iloc[i]
                region_val = _clean_field(row, "Region") or _clean_field(row, "Categories")
                out.append({
                    "matched_catalog_name": _clean_field(row, "Matched Catalog Name") or "",
                    "score": float(row.get("Final Score", 0.0)),
                    "status": _clean_field(row, "Status") or "",
                    "logic_notes": _clean_field(row, "Logic Notes") or "",
                    "suggested_bt": _clean_field(row, "BasicType") or "",
                    "suggested_gk": _clean_field(row, "GenericKeywords") or "",
                    "suggested_region": region_val or "",
                    "rules_applied": ""
                })
            else:
                out.append({
                    "matched_catalog_name": "",
                    "score": 0.0,
                    "status": "Low / Rejected",
                    "logic_notes": "No candidate found",
                    "suggested_bt": "",
                    "suggested_gk": "",
                    "suggested_region": "",
                    "rules_applied": ""
                })

        emit_progress("writing_results", 95.0)
        _apply_template_tag_enrichment(skus, out, domain, mode="matcher")
        emit_progress("done", 100.0, 0)
        return {"domain": domain, "total": len(out), "results": out}

    elif task == "classifier":
        emit_progress("embedding", 5.0)
        classifier = get_classifier(domain)
        embed_engine, ner_engine = _get_shared_models()
        vector_store = _get_vector_store()
        reranker = RerankerWrapper(embed_engine) if (hasattr(embed_engine, 'cross_session') and embed_engine.cross_session) else None

        chunk_size = getattr(config, "CLASSIFY_CHUNK_SIZE", 250)
        results = []

        for chunk_idx, chunk_start in enumerate(range(0, total_skus, chunk_size)):
            check_cancel()
            chunk_end = min(chunk_start + chunk_size, total_skus)
            chunk_skus = skus[chunk_start:chunk_end]
            sku_names = [sku.get("name", "") for sku in chunk_skus]
            sku_descs = [sku.get("description", "") for sku in chunk_skus]
            sku_cats = [sku.get("category", "") for sku in chunk_skus]
            sku_prices = [sku.get("price", 0.0) or 0.0 for sku in chunk_skus]

            query_embeddings = _extract_query_embeddings(embed_engine, sku_names, sku_descs, sku_cats)

            def clf_progress_cb(pct, msg=None):
                check_cancel()
                overall_pct = ((chunk_start + (float(pct) / 100.0) * len(chunk_skus)) / total_skus) * 100.0
                elapsed = max(0.1, time.time() - start_time)
                if overall_pct < 15.0:
                    stage = "embedding"
                elif overall_pct < 40.0:
                    stage = "vector_search"
                elif overall_pct < 70.0:
                    stage = "reranking"
                else:
                    stage = "classifying"
                eta = max(1, int((elapsed / (overall_pct / 100.0)) - elapsed)) if overall_pct > 1.0 else None
                emit_progress(stage, overall_pct, eta)

            chunk_results = tag_all_skus(
                sku_names=sku_names,
                sku_categories=sku_cats,
                query_embeddings=query_embeddings,
                vector_store=vector_store,
                reranker=reranker,
                classifier=classifier,
                sku_descriptions=sku_descs,
                sku_prices=sku_prices,
                embed_engine=embed_engine,
                ner_engine=getattr(classifier, 'ner_engine', ner_engine),
                is_cancelled=is_cancelled or (lambda: False),
                progress_callback=clf_progress_cb
            )
            results.extend(chunk_results)
            import gc
            gc.collect()

        out = [
            {
                "suggested_bt": str(res.get("suggested_bt") or ""),
                "bt_confidence": float(res.get("bt_confidence") or 0.0),
                "bt_status": str(res.get("bt_status") or ""),
                "bt_source": str(res.get("bt_source") or ""),
                "suggested_gk": str(res.get("suggested_gk") or ""),
                "gk_confidence": float(res.get("gk_confidence") or 0.0),
                "gk_status": str(res.get("gk_status") or ""),
                "suggested_region": str(res.get("suggested_region") or res.get("suggested_category") or ""),
                "region_confidence": float(res.get("region_confidence") or res.get("category_confidence") or 0.0),
                "region_status": str(res.get("region_status") or res.get("category_status") or ""),
                "region_source": str(res.get("region_source") or res.get("category_source") or ""),
                "rules_applied": str(res.get("rules_applied") or ""),
                "logic_notes": str(res.get("reasoning") or "")
            }
            for res in results
        ]

        emit_progress("writing_results", 95.0)
        _apply_template_tag_enrichment(skus, out, domain, mode="classifier")
        emit_progress("done", 100.0, 0)
        return {"domain": domain, "total": len(out), "results": out}

    elif task == "pipeline":
        emit_progress("embedding", 5.0)
        matcher = get_pipeline(domain)
        input_df = _build_sku_dataframe(skus)

        def matcher_pipeline_cb(pct, msg=None):
            check_cancel()
            overall_pct = float(pct) * 0.70
            if overall_pct < 20.0:
                stage = "embedding"
            elif overall_pct < 45.0:
                stage = "vector_search"
            elif overall_pct < 65.0:
                stage = "reranking"
            else:
                stage = "classifying"
            elapsed = max(0.1, time.time() - start_time)
            eta = max(1, int((elapsed / (overall_pct / 100.0)) - elapsed)) if overall_pct > 1.0 else None
            emit_progress(stage, overall_pct, eta)

        match_df = matcher.process_inputs(input_df, progress_callback=matcher_pipeline_cb)

        pipeline_out = []
        escalate_indices = []

        for i, sku in enumerate(skus):
            if i < len(match_df):
                row = match_df.iloc[i]
                m_status = _clean_field(row, "Status") or "Low / Rejected"
                m_score  = float(row.get("Final Score", 0.0))
                region_val = _clean_field(row, "Region") or _clean_field(row, "Categories")

                result = {
                    "matched_catalog_name": _clean_field(row, "Matched Catalog Name") or "",
                    "score": m_score,
                    "status": m_status,
                    "logic_notes": _clean_field(row, "Logic Notes") or "",
                    "suggested_bt": _clean_field(row, "BasicType") or "",
                    "bt_confidence": m_score, "bt_status": m_status,
                    "suggested_gk": _clean_field(row, "GenericKeywords") or "",
                    "gk_confidence": m_score, "gk_status": m_status,
                    "suggested_region": region_val,
                    "region_confidence": m_score, "region_status": m_status,
                    "pipeline_source": "Matcher",
                    "escalated": False,
                }
            else:
                m_status = "Low / Rejected"
                m_score  = 0.0
                result = {
                    "matched_catalog_name": "", "score": 0.0,
                    "status": "Low / Rejected", "logic_notes": "No candidate found",
                    "suggested_bt": "", "bt_confidence": 0.0, "bt_status": "Low / Rejected",
                    "suggested_gk": "", "gk_confidence": 0.0, "gk_status": "Low / Rejected",
                    "suggested_region": "", "region_confidence": 0.0, "region_status": "Low / Rejected",
                    "pipeline_source": "", "escalated": False,
                }

            if m_status in config.PIPELINE_ESCALATE_STATUSES:
                escalate_indices.append(i)

            pipeline_out.append(result)

        if escalate_indices:
            emit_progress("classifying", 70.0)
            classifier = get_classifier(domain)
            embed_engine, ner_engine = _get_shared_models()
            vector_store = _get_vector_store()
            reranker = RerankerWrapper(embed_engine) if (hasattr(embed_engine, 'cross_session') and embed_engine.cross_session) else None

            total_esc = len(escalate_indices)
            esc_chunk_size = getattr(config, "CLASSIFY_CHUNK_SIZE", 250)

            for esc_chunk_idx, esc_start in enumerate(range(0, total_esc, esc_chunk_size)):
                check_cancel()
                esc_end = min(esc_start + esc_chunk_size, total_esc)
                chunk_esc_indices = escalate_indices[esc_start:esc_end]
                esc_skus = [skus[i] for i in chunk_esc_indices]
                sku_names = [s.get("name", "") for s in esc_skus]
                sku_descs = [s.get("description", "") or "" for s in esc_skus]
                sku_cats  = [s.get("category", "") or "" for s in esc_skus]
                sku_prices = [s.get("price", 0.0) or 0.0 for s in esc_skus]

                query_embeddings = _extract_query_embeddings(embed_engine, sku_names, sku_descs, sku_cats)

                def pipeline_clf_cb(pct, msg=None):
                    check_cancel()
                    chunk_overall_esc_pct = ((esc_start + (float(pct) / 100.0) * len(chunk_esc_indices)) / total_esc)
                    overall_pct = 70.0 + (chunk_overall_esc_pct * 20.0)
                    elapsed = max(0.1, time.time() - start_time)
                    eta = max(1, int((elapsed / (overall_pct / 100.0)) - elapsed)) if overall_pct > 1.0 else None
                    emit_progress("classifying", overall_pct, eta)

                clf_results = tag_all_skus(
                    sku_names=sku_names,
                    sku_categories=sku_cats,
                    query_embeddings=query_embeddings,
                    vector_store=vector_store,
                    reranker=reranker,
                    classifier=classifier,
                    sku_descriptions=sku_descs,
                    sku_prices=sku_prices,
                    embed_engine=embed_engine,
                    ner_engine=getattr(classifier, 'ner_engine', ner_engine),
                    is_cancelled=is_cancelled or (lambda: False),
                    progress_callback=pipeline_clf_cb
                )

                for j, orig_idx in enumerate(chunk_esc_indices):
                    if j >= len(clf_results):
                        continue
                    clf = clf_results[j]
                    row = pipeline_out[orig_idx]

                    row["suggested_bt"]      = str(clf.get("suggested_bt", ""))
                    row["bt_confidence"]     = float(clf.get("bt_confidence", 0.0))
                    row["bt_status"]         = str(clf.get("bt_status", ""))
                    row["suggested_gk"]      = str(clf.get("suggested_gk", ""))
                    row["gk_confidence"]     = float(clf.get("gk_confidence", 0.0))
                    row["gk_status"]         = str(clf.get("gk_status", ""))
                    row["suggested_region"]  = str(clf.get("suggested_region", clf.get("suggested_category", "")))
                    row["region_confidence"] = float(clf.get("region_confidence", clf.get("category_confidence", 0.0)))
                    row["region_status"]     = str(clf.get("region_status", clf.get("category_status", "")))
                    row["escalated"] = True

                    matcher_norm   = min(row["score"] / 100.0, 1.0)
                    clf_confidence = row["bt_confidence"]
                    row["pipeline_source"] = "Classifier" if clf_confidence > matcher_norm else "Matcher"

                import gc
                gc.collect()
        else:
            emit_progress("classifying", 90.0)

        # Template Tag Enrichment for Pipeline Mode
        _apply_template_tag_enrichment(skus, pipeline_out, domain, mode="pipeline")

        emit_progress("applying_rules", 92.0)
        for j, row in enumerate(pipeline_out):
            check_cancel()
            overall_pct = 92.0 + (((j + 1) / len(pipeline_out)) * 7.0)
            emit_progress("applying_rules", overall_pct)
            sku = skus[j]
            record = {
                "sku_name": sku.get("name", ""),
                "domain": domain,
                "bt": row.get("suggested_bt", ""),
                "gk": [x.strip() for x in str(row.get("suggested_gk", "")).split(",") if x.strip()],
                "region": row.get("suggested_region") if domain == config.DOMAIN_FOOD else None,
                "category": row.get("suggested_region") if domain == config.DOMAIN_MARKET else None,
                "price": sku.get("price", 0.0),
                "confidence": max(row.get("score", 0), row.get("bt_confidence", 0) or 0),
                "match_source": "classifier" if row.get("pipeline_source") == "Classifier" else "catalogue",
                "matched_sku": row.get("matched_catalog_name", ""),
                "reasoning": row.get("logic_notes", "")
            }
            
            aug_record = run_rules_engine(record)
            
            row["suggested_bt"] = str(aug_record.get("bt") or "")
            row["suggested_gk"] = ", ".join(aug_record.get("gk", []))
            row["suggested_region"] = str(aug_record.get("region") or "") if domain == config.DOMAIN_FOOD else str(aug_record.get("category") or "")
            
            applied = aug_record.get("rules_applied", [])
            row["rules_applied"] = json.dumps(applied) if applied else ""

        emit_progress("writing_results", 99.0)
        emit_progress("done", 100.0, 0)
        return {
            "domain": domain,
            "total": len(pipeline_out),
            "escalated_count": len(escalate_indices),
            "results": pipeline_out,
        }

    else:
        raise ValueError(f"Unknown task: '{task}'. Must be matcher | classifier | pipeline.")
