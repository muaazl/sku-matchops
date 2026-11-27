"""
SKU MatchOps Engine - Request Processor
Executes Matching, Classification, and Pipeline escalation tasks.
"""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from engine.core import config
from engine.pipelines.classification_pipeline import compute_weighted_confidence, tag_all_skus
from engine.pipelines.rules_evaluation_pipeline import run_rules_engine
from engine.pipelines.runtime_pipeline import (
    _get_reranker_model,
    _get_shared_models,
    _get_vector_store,
    check_models_loaded,
    get_classifier,
    get_ner_engine,
    get_pipeline,
)
from engine.pipelines.template_pipeline import (
    apply_template_tag_enrichment,
    suggest_tags_from_template,
)
from engine.preprocessing.flavor_preprocessor import resolve_food_mixed_suffix
from engine.preprocessing.text_normalizer import build_ner_input_text, prepare_ner_text

logger = logging.getLogger("matchops.engine.processor")


def _build_sku_dataframe(skus: list[dict[str, Any]]) -> pd.DataFrame:
    """Constructs a standard SKU DataFrame from SKU dictionaries."""
    data = {
        "SKU": [
            str(
                sku.get("name") or sku.get("sku_raw") or sku.get("Name") or sku.get("SKU") or ""
            ).strip()
            for sku in skus
        ],
        "Price": [sku.get("price", 0.0) or 0.0 for sku in skus],
        "Description": [sku.get("description", "") or "" for sku in skus],
        "Category": [sku.get("category", "") or "" for sku in skus],
    }
    return pd.DataFrame(data)


def _clean_field(row, key: str) -> str | None:
    """Helper to safely extract a non-empty string or None from a pandas row."""
    if key not in row:
        return None
    val = row[key]
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _extract_query_embeddings(
    embed_engine, names: list[str], descriptions: list[str], categories: list[str]
):
    """Encodes query texts with the active embedding model using fallback for description/category."""
    safe_descs = [d if d else "" for d in descriptions]
    safe_cats = [c if c else "" for c in categories]
    return embed_engine.embed_weighted_sku(names, safe_descs, safe_cats)


def _apply_template_tag_enrichment(
    skus: list[dict[str, Any]], results: list[dict], domain: str, mode: str = "classifier"
) -> None:
    """Enriches predicted tags using template-based keyword substitution if enabled."""
    apply_template_tag_enrichment(
        skus,
        results,
        domain,
        mode=mode,
        suggest_fn=suggest_tags_from_template,
    )


def _resolve_active_reranker(rerank_engine):
    if rerank_engine is None or not getattr(rerank_engine, "cross_session", None):
        return None
    return rerank_engine


def process_request(
    skus: list[dict[str, Any]],
    task: str,
    domain: str = "market",
    job_id: str | None = None,
    progress_callback: Callable[[str, float, int | None], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Executes Matching, Classification, or Pipeline for a batch of SKUs.
    Calls progress_callback(stage_name, progress_pct, eta_seconds) dynamically.
    """
    task = task.lower()
    domain = domain or "market"
    check_models_loaded(domain, task)
    skus = [s.copy() for s in skus]

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise InterruptedError("Job cancelled by user.")

    def emit_progress(stage: str, pct: float, eta: int | None = None):
        if progress_callback:
            progress_callback(stage, min(100.0, max(0.0, round(float(pct), 2))), eta)

    start_time = time.time()
    total_skus = len(skus)
    stage_timings: dict[str, float] = {}
    _eta_state: dict[str, Any] = {
        "stage": None,
        "stage_t0": start_time,
        "stage_p0": 0.0,
        "fast_rate": 0.5,
    }

    def compute_eta(stage: str, pct: float) -> int | None:
        now = time.time()
        p = float(pct)
        if p <= 1.0:
            return None
        if p >= 100.0:
            return 0
        if stage != _eta_state["stage"]:
            elapsed_prev = now - _eta_state["stage_t0"]
            dp_prev = p - _eta_state["stage_p0"]
            if (
                _eta_state["stage"] in ("embedding", "vector_search")
                and dp_prev > 1.0
                and elapsed_prev > 0.1
            ):
                _eta_state["fast_rate"] = max(0.05, min(2.0, elapsed_prev / dp_prev))
            _eta_state["stage"] = stage
            _eta_state["stage_t0"] = now
            _eta_state["stage_p0"] = p

        dt = max(0.0, now - _eta_state["stage_t0"])
        dp = p - _eta_state["stage_p0"]
        if dp >= 0.25 and dt >= 0.2:
            stage_rate = dt / dp
            if stage == "reranking":
                rerank_end = 75.0 if task == "matcher" else (90.0 if task == "classifier" else 52.5)
                rem_rerank = max(0.0, rerank_end - p)
                rem_after = max(0.0, 100.0 - max(p, rerank_end))
                return max(1, int(rem_rerank * stage_rate + rem_after * _eta_state["fast_rate"]))
            return max(1, int((100.0 - p) * stage_rate))

        elapsed = max(0.1, now - start_time)
        return max(1, int((elapsed / (p / 100.0)) - elapsed))

    if domain == config.DOMAIN_FOOD:
        t_ner_start = time.perf_counter()
        _embed_engine, _ = _get_shared_models()
        ner_engine = get_ner_engine(domain)
        if ner_engine:
            sku_texts = [
                prepare_ner_text(
                    build_ner_input_text(
                        sku.get("name", ""), sku.get("description", ""), sku.get("category", "")
                    )
                )
                for sku in skus
            ]
            ner_results = ner_engine.batch_extract_entities(sku_texts)
            for i, sku in enumerate(skus):
                flavor_set = ner_results[i].get("flavor", set())
                name = sku.get("name", "")
                suffix = resolve_food_mixed_suffix(
                    name,
                    flavor_set,
                    meat_flavors=getattr(ner_engine, "meat_flavors", set()),
                    vegetable_flavors=getattr(ner_engine, "vegetable_flavors", set()),
                    seafood_flavors=getattr(ner_engine, "seafood_flavors", set()),
                )
                if suffix:
                    name = f"{name} {suffix}"
                sku["name"] = name
        stage_timings["food_ner_prepass_ms"] = round((time.perf_counter() - t_ner_start) * 1000, 2)

    if task == "matcher":
        emit_progress("embedding", 5.0)
        matcher = get_pipeline(domain)
        input_df = _build_sku_dataframe(skus)

        def matcher_cb(pct, msg=None):
            check_cancel()
            p = float(pct)
            if p < 30.0:
                stage = "embedding"
            elif p < 50.0:
                stage = "vector_search"
            elif p < 75.0:
                stage = "reranking"
            else:
                stage = "writing_results"
            emit_progress(stage, p, compute_eta(stage, p))

        t_matcher_start = time.perf_counter()
        results_df = matcher.process_inputs(input_df, progress_callback=matcher_cb)
        stage_timings["matcher_core_ms"] = round((time.perf_counter() - t_matcher_start) * 1000, 2)

        out = []
        for i in range(len(skus)):
            if i < len(results_df):
                row = results_df.iloc[i]
                region_val = _clean_field(row, "Region") or _clean_field(row, "Categories")
                out.append(
                    {
                        "matched_catalog_name": _clean_field(row, "Matched Catalog Name") or "",
                        "score": float(row.get("Final Score", 0.0)),
                        "status": _clean_field(row, "Status") or "",
                        "logic_notes": _clean_field(row, "Logic Notes") or "",
                        "suggested_bt": _clean_field(row, "BasicType") or "",
                        "suggested_gk": _clean_field(row, "GenericKeywords") or "",
                        "suggested_region": region_val or "",
                        "input_entities": row.get("Input Entities") or {},
                        "catalog_entities": row.get("Catalog Entities") or {},
                        "rules_applied": "",
                    }
                )
            else:
                out.append(
                    {
                        "matched_catalog_name": "",
                        "score": 0.0,
                        "status": "Low / Rejected",
                        "logic_notes": "No candidate found",
                        "suggested_bt": "",
                        "suggested_gk": "",
                        "suggested_region": "",
                        "input_entities": {},
                        "catalog_entities": {},
                        "rules_applied": "",
                    }
                )

        emit_progress("writing_results", 95.0)
        t_tmpl_start = time.perf_counter()
        _apply_template_tag_enrichment(skus, out, domain, mode="matcher")
        stage_timings["template_enrichment_ms"] = round(
            (time.perf_counter() - t_tmpl_start) * 1000, 2
        )
        stage_timings["total_ms"] = round((time.time() - start_time) * 1000, 2)
        emit_progress("done", 100.0, 0)
        return {"domain": domain, "total": len(out), "results": out, "stage_timings": stage_timings}

    if task == "classifier":
        emit_progress("embedding", 5.0)
        classifier = get_classifier(domain)
        embed_engine, _ = _get_shared_models()
        ner_engine = getattr(classifier, "ner_engine", None) or get_ner_engine(domain)
        vector_store = _get_vector_store()
        reranker = _resolve_active_reranker(_get_reranker_model(embed_engine))

        chunk_size = getattr(config, "CLASSIFY_CHUNK_SIZE", 250)
        results = []
        t_clf_tag_start = time.perf_counter()

        for chunk_start in range(0, total_skus, chunk_size):
            check_cancel()
            chunk_end = min(chunk_start + chunk_size, total_skus)
            chunk_skus = skus[chunk_start:chunk_end]
            sku_names = [sku.get("name", "") for sku in chunk_skus]
            sku_descs = [sku.get("description", "") for sku in chunk_skus]
            sku_cats = [sku.get("category", "") for sku in chunk_skus]
            sku_prices = [sku.get("price", 0.0) or 0.0 for sku in chunk_skus]

            query_embeddings = _extract_query_embeddings(
                embed_engine, sku_names, sku_descs, sku_cats
            )

            def clf_progress_cb(
                pct,
                msg=None,
                _start=chunk_start,
                _len=len(chunk_skus),
            ):
                check_cancel()
                overall_pct = ((_start + (float(pct) / 100.0) * _len) / total_skus) * 100.0
                if overall_pct < 15.0:
                    stage = "embedding"
                elif overall_pct < 45.0:
                    stage = "vector_search"
                elif overall_pct < 90.0:
                    stage = "reranking"
                else:
                    stage = "classifying"
                emit_progress(stage, overall_pct, compute_eta(stage, overall_pct))

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
                ner_engine=getattr(classifier, "ner_engine", ner_engine),
                is_cancelled=is_cancelled or (lambda: False),
                progress_callback=clf_progress_cb,
            )
            results.extend(chunk_results)

        stage_timings["tag_all_skus_ms"] = round((time.perf_counter() - t_clf_tag_start) * 1000, 2)

        out = [
            {
                "suggested_bt": str(res.get("suggested_bt") or ""),
                "bt_confidence": float(res.get("bt_confidence") or 0.0),
                "bt_status": str(res.get("bt_status") or ""),
                "bt_source": str(res.get("bt_source") or ""),
                "model": str(res.get("model") or "logreg"),
                "bt_model": str(res.get("bt_model") or res.get("model") or "logreg"),
                "suggested_gk": str(res.get("suggested_gk") or ""),
                "gk_confidence": float(res.get("gk_confidence") or 0.0),
                "gk_status": str(res.get("gk_status") or ""),
                "suggested_region": str(
                    res.get("suggested_region") or res.get("suggested_category") or ""
                ),
                "region_confidence": float(
                    res.get("region_confidence") or res.get("category_confidence") or 0.0
                ),
                "region_status": str(res.get("region_status") or res.get("category_status") or ""),
                "region_source": str(res.get("region_source") or res.get("category_source") or ""),
                "rules_applied": str(res.get("rules_applied") or ""),
                "logic_notes": str(res.get("reasoning") or ""),
            }
            for res in results
        ]

        emit_progress("writing_results", 95.0)
        t_tmpl_start = time.perf_counter()
        _apply_template_tag_enrichment(skus, out, domain, mode="classifier")
        stage_timings["template_enrichment_ms"] = round(
            (time.perf_counter() - t_tmpl_start) * 1000, 2
        )

        emit_progress("applying_rules", 96.0)
        t_rules_start = time.perf_counter()
        for j, row in enumerate(out):
            check_cancel()
            sku = skus[j]
            record = {
                "sku_name": sku.get("name", ""),
                "domain": domain,
                "bt": row.get("suggested_bt", ""),
                "gk": [x.strip() for x in str(row.get("suggested_gk", "")).split(",") if x.strip()],
                "region": row.get("suggested_region") if domain == config.DOMAIN_FOOD else None,
                "category": row.get("suggested_region") if domain == config.DOMAIN_MARKET else None,
                "price": sku.get("price", 0.0),
                "confidence": compute_weighted_confidence(
                    row.get("bt_confidence"),
                    row.get("gk_confidence"),
                    row.get("region_confidence", row.get("category_confidence")),
                    has_bt=bool(row.get("suggested_bt")),
                    has_gk=bool(row.get("suggested_gk")),
                    has_region=bool(row.get("suggested_region") or row.get("suggested_category")),
                ),
                "match_source": "Classifier",
                "matched_sku": "",
                "reasoning": row.get("logic_notes", ""),
            }

            aug_record = run_rules_engine(record)
            row["suggested_bt"] = str(aug_record.get("bt") or "")
            row["suggested_gk"] = ", ".join(aug_record.get("gk", []))
            row["suggested_region"] = (
                str(aug_record.get("region") or "")
                if domain == config.DOMAIN_FOOD
                else str(aug_record.get("category") or "")
            )
            applied = aug_record.get("rules_applied", [])
            row["rules_applied"] = json.dumps(applied) if applied else ""

        stage_timings["rules_engine_ms"] = round((time.perf_counter() - t_rules_start) * 1000, 2)
        stage_timings["total_ms"] = round((time.time() - start_time) * 1000, 2)
        emit_progress("done", 100.0, 0)
        return {"domain": domain, "total": len(out), "results": out, "stage_timings": stage_timings}

    if task == "pipeline":
        emit_progress("embedding", 5.0)
        matcher = get_pipeline(domain)
        input_df = _build_sku_dataframe(skus)

        def matcher_pipeline_cb(pct, msg=None):
            check_cancel()
            overall_pct = float(pct) * 0.70
            if overall_pct < 20.0:
                stage = "embedding"
            elif overall_pct < 35.0:
                stage = "vector_search"
            elif overall_pct < 52.5:
                stage = "reranking"
            else:
                stage = "classifying"
            emit_progress(stage, overall_pct, compute_eta(stage, overall_pct))

        t_pipe_match_start = time.perf_counter()
        match_df = matcher.process_inputs(input_df, progress_callback=matcher_pipeline_cb)
        stage_timings["pipeline_matcher_ms"] = round(
            (time.perf_counter() - t_pipe_match_start) * 1000, 2
        )

        pipeline_out = []
        escalate_indices = []

        for i in range(len(skus)):
            if i < len(match_df):
                row = match_df.iloc[i]
                m_status = _clean_field(row, "Status") or "Low / Rejected"
                m_score = float(row.get("Final Score", 0.0))
                region_val = _clean_field(row, "Region") or _clean_field(row, "Categories")

                result = {
                    "matched_catalog_name": _clean_field(row, "Matched Catalog Name") or "",
                    "score": m_score,
                    "status": m_status,
                    "logic_notes": _clean_field(row, "Logic Notes") or "",
                    "suggested_bt": _clean_field(row, "BasicType") or "",
                    "bt_confidence": None,
                    "bt_status": m_status,
                    "suggested_gk": _clean_field(row, "GenericKeywords") or "",
                    "gk_confidence": None,
                    "gk_status": m_status,
                    "suggested_region": region_val,
                    "region_confidence": None,
                    "region_status": m_status,
                    "pipeline_source": "Matcher",
                    "escalated": False,
                }
            else:
                m_status = "Low / Rejected"
                result = {
                    "matched_catalog_name": "",
                    "score": 0.0,
                    "status": "Low / Rejected",
                    "logic_notes": "No candidate found",
                    "suggested_bt": "",
                    "bt_confidence": None,
                    "bt_status": "Low / Rejected",
                    "suggested_gk": "",
                    "gk_confidence": None,
                    "gk_status": "Low / Rejected",
                    "suggested_region": "",
                    "region_confidence": None,
                    "region_status": "Low / Rejected",
                    "pipeline_source": "Matcher",
                    "escalated": False,
                }

            if m_status in config.PIPELINE_ESCALATE_STATUSES:
                escalate_indices.append(i)

            pipeline_out.append(result)

        if escalate_indices:
            emit_progress("classifying", 70.0)
            classifier = get_classifier(domain)
            embed_engine, _ = _get_shared_models()
            ner_engine = getattr(classifier, "ner_engine", None) or get_ner_engine(domain)
            vector_store = _get_vector_store()
            reranker = _resolve_active_reranker(_get_reranker_model(embed_engine))

            total_esc = len(escalate_indices)
            esc_chunk_size = getattr(config, "CLASSIFY_CHUNK_SIZE", 250)
            t_esc_clf_start = time.perf_counter()

            for esc_start in range(0, total_esc, esc_chunk_size):
                check_cancel()
                esc_end = min(esc_start + esc_chunk_size, total_esc)
                chunk_esc_indices = escalate_indices[esc_start:esc_end]
                esc_skus = [skus[i] for i in chunk_esc_indices]
                sku_names = [s.get("name", "") for s in esc_skus]
                sku_descs = [s.get("description", "") or "" for s in esc_skus]
                sku_cats = [s.get("category", "") or "" for s in esc_skus]
                sku_prices = [s.get("price", 0.0) or 0.0 for s in esc_skus]

                query_embeddings = _extract_query_embeddings(
                    embed_engine, sku_names, sku_descs, sku_cats
                )

                def pipeline_clf_cb(
                    pct,
                    msg=None,
                    _start=esc_start,
                    _len=len(chunk_esc_indices),
                ):
                    check_cancel()
                    chunk_overall_esc_pct = (_start + (float(pct) / 100.0) * _len) / total_esc
                    overall_pct = 70.0 + (chunk_overall_esc_pct * 20.0)
                    emit_progress(
                        "classifying", overall_pct, compute_eta("classifying", overall_pct)
                    )

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
                    ner_engine=getattr(classifier, "ner_engine", ner_engine),
                    is_cancelled=is_cancelled or (lambda: False),
                    progress_callback=pipeline_clf_cb,
                )

                for j, orig_idx in enumerate(chunk_esc_indices):
                    if j >= len(clf_results):
                        continue
                    clf = clf_results[j]
                    row = pipeline_out[orig_idx]
                    row["escalated"] = True

                    raw_m_score = float(row.get("score", 0.0))
                    scale_factor = (
                        0.75 / max(float(config.CONFIDENCE_THRESHOLD_HIGH), 0.01)
                        if float(config.CONFIDENCE_THRESHOLD_HIGH) < 0.75
                        else 1.0
                    )
                    matcher_norm = min(raw_m_score * scale_factor, 1.0)
                    clf_bt_tag = str(clf.get("suggested_bt") or "").strip()
                    clf_confidence = float(clf.get("bt_confidence") or 0.0) if clf_bt_tag else 0.0
                    prior_notes = str(row.get("logic_notes") or "")
                    prior_status = str(row.get("status") or "")
                    matcher_bt_confirmed = (
                        "BasicType Match (" in prior_notes
                        and prior_status not in ("Low / Rejected", "Rejected")
                    )
                    classifier_won = (
                        bool(clf_bt_tag)
                        and not matcher_bt_confirmed
                        and clf_confidence > matcher_norm
                    )
                    row["pipeline_source"] = "Classifier" if classifier_won else "Matcher"

                    if classifier_won:
                        prior_bt = str(row.get("suggested_bt") or "").strip()
                        prior_gk = str(row.get("suggested_gk") or "").strip()
                        same_bt_with_catalog_gk = (
                            bool(prior_bt and prior_gk) and prior_bt.lower() == clf_bt_tag.lower()
                        )
                        row["suggested_bt"] = str(clf.get("suggested_bt", ""))
                        row["bt_confidence"] = float(clf.get("bt_confidence") or 0.0)
                        row["bt_status"] = str(clf.get("bt_status", ""))
                        if not same_bt_with_catalog_gk:
                            row["suggested_gk"] = str(clf.get("suggested_gk", ""))
                        row["gk_confidence"] = float(clf.get("gk_confidence") or 0.0)
                        row["gk_status"] = str(clf.get("gk_status", ""))
                        row["suggested_region"] = str(
                            clf.get("suggested_region", clf.get("suggested_category", ""))
                        )
                        row["region_confidence"] = float(
                            clf.get("region_confidence") or clf.get("category_confidence") or 0.0
                        )
                        row["region_status"] = str(
                            clf.get("region_status", clf.get("category_status", ""))
                        )
                        row["model"] = str(clf.get("model") or "logreg")
                        row["bt_model"] = str(clf.get("bt_model") or clf.get("model") or "logreg")

                        prior = str(row.get("logic_notes") or "").strip()
                        clf_note = str(clf.get("reasoning", ""))
                        row["logic_notes"] = (
                            f"Escalated from matcher [{prior}] -> {clf_note}" if prior else clf_note
                        )

            stage_timings["pipeline_escalated_clf_ms"] = round(
                (time.perf_counter() - t_esc_clf_start) * 1000, 2
            )
        else:
            emit_progress("classifying", 90.0)

        t_tmpl_start = time.perf_counter()
        _apply_template_tag_enrichment(skus, pipeline_out, domain, mode="pipeline")
        stage_timings["template_enrichment_ms"] = round(
            (time.perf_counter() - t_tmpl_start) * 1000, 2
        )

        emit_progress("applying_rules", 92.0)
        t_rules_start = time.perf_counter()
        for j, row in enumerate(pipeline_out):
            check_cancel()
            overall_pct = 92.0 + (((j + 1) / len(pipeline_out)) * 7.0)
            emit_progress("applying_rules", overall_pct)
            sku = skus[j]
            is_clf = row.get("pipeline_source") == "Classifier"
            if is_clf:
                conf_val = compute_weighted_confidence(
                    row.get("bt_confidence"),
                    row.get("gk_confidence"),
                    row.get("region_confidence", row.get("category_confidence")),
                    has_bt=bool(row.get("suggested_bt")),
                    has_gk=bool(row.get("suggested_gk")),
                    has_region=bool(row.get("suggested_region") or row.get("category")),
                )
            else:
                conf_val = float(row.get("score") or 0.0)

            record = {
                "sku_name": sku.get("name", ""),
                "domain": domain,
                "bt": row.get("suggested_bt", ""),
                "gk": [x.strip() for x in str(row.get("suggested_gk", "")).split(",") if x.strip()],
                "region": row.get("suggested_region") if domain == config.DOMAIN_FOOD else None,
                "category": row.get("suggested_region") if domain == config.DOMAIN_MARKET else None,
                "price": sku.get("price", 0.0),
                "confidence": conf_val,
                "match_source": "Classifier" if is_clf else "Matcher",
                "matched_sku": row.get("matched_catalog_name", ""),
                "reasoning": row.get("logic_notes", ""),
            }

            aug_record = run_rules_engine(record)
            row["suggested_bt"] = str(aug_record.get("bt") or "")
            row["suggested_gk"] = ", ".join(aug_record.get("gk", []))
            row["suggested_region"] = (
                str(aug_record.get("region") or "")
                if domain == config.DOMAIN_FOOD
                else str(aug_record.get("category") or "")
            )
            applied = aug_record.get("rules_applied", [])
            row["rules_applied"] = json.dumps(applied) if applied else ""

        stage_timings["rules_engine_ms"] = round((time.perf_counter() - t_rules_start) * 1000, 2)
        stage_timings["total_ms"] = round((time.time() - start_time) * 1000, 2)

        emit_progress("writing_results", 99.0)
        emit_progress("done", 100.0, 0)
        return {
            "domain": domain,
            "total": len(pipeline_out),
            "escalated_count": len(escalate_indices),
            "results": pipeline_out,
            "stage_timings": stage_timings,
        }

    raise ValueError(f"Unknown task: '{task}'. Must be matcher | classifier | pipeline.")
