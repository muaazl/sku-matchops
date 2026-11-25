"""Job progress, completion, and failure callback orchestration service."""

import json
import logging
import uuid
from typing import Any

from backend.core.helpers import compute_weighted_confidence, parse_optional_float
from backend.core.observability import MATCH_CONFIDENCE
from backend.repositories import job_progress_repository, job_repository
from backend.services import engine_gateway_service

logger = logging.getLogger("matchops.services.job_callback")


def handle_job_progress(
    *,
    job_id: str,
    progress_pct: float,
    eta: int | None,
    current_stage: str,
) -> None:
    """Persists stage transitions to PostgreSQL and broadcasts live progress via Redis."""
    j_id = str(job_id)
    last_stage, has_mem_stage = job_progress_repository.fetch_cached_job_stage(j_id)

    if last_stage != current_stage or not has_mem_stage:
        try:
            updated, cancel_row = job_repository.update_job_stage_if_active(j_id, current_stage)
            if not updated:
                job_progress_repository.clear_job_progress(j_id)
                if cancel_row and (cancel_row[0] == "cancelled" or bool(cancel_row[1])):
                    try:
                        engine_gateway_service.cancel_engine_job(j_id)
                    except Exception as cancel_err:
                        logger.debug(
                            f"Could not re-signal cancellation for job {j_id}: {cancel_err}"
                        )
                return
        except Exception as e:
            logger.error(f"Failed to update progress for job {j_id} in DB: {e}")

    job_progress_repository.set_job_progress(j_id, pct=progress_pct, eta=eta, stage=current_stage)
    job_progress_repository.publish_job_event(
        {
            "id": j_id,
            "status": "running",
            "current_stage": current_stage,
            "progress_pct": progress_pct,
            "eta_seconds": eta,
        }
    )


def build_processed_sku_rows(
    job_id: str,
    domain: str,
    task: str,
    input_skus: list[dict],
    results: list[dict[str, Any]],
    rule_version: str,
) -> list[dict[str, Any]]:
    """Maps engine results onto `processed_skus` rows keyed by (job, index, rule_version)."""
    task_lower = (task or "pipeline").lower()
    rows = []
    for i, res in enumerate(results):
        input_sku = input_skus[i] if i < len(input_skus) else {}

        sku_name = input_sku.get(
            "name", input_sku.get("Name", input_sku.get("sku_name", res.get("sku_name", "")))
        )
        input_price = input_sku.get("price", input_sku.get("Price", res.get("price")))
        input_description = (
            input_sku.get(
                "description", input_sku.get("Description", res.get("input_description", ""))
            )
            or ""
        )
        input_category = (
            input_sku.get("category", input_sku.get("Category", res.get("input_category", "")))
            or ""
        )

        gk = res.get("suggested_gk", "")
        if gk is None or gk == "None":
            gk = ""
        if task_lower == "matcher":
            conf = res.get("score", 0.0)
            source = "Matcher"
        elif task_lower == "classifier":
            source = "Classifier"
            conf = compute_weighted_confidence(
                res.get("bt_confidence"),
                res.get("gk_confidence"),
                res.get("region_confidence", res.get("category_confidence")),
                has_bt=bool(res.get("suggested_bt")),
                has_gk=bool(res.get("suggested_gk")),
                has_region=bool(res.get("suggested_region") or res.get("suggested_category")),
            )
        else:
            p_src = str(res.get("pipeline_source") or "").strip().lower()
            if p_src == "classifier":
                source = "Classifier"
            elif p_src in ("matcher", "catalogue"):
                source = "Matcher"
            elif res.get("matched_catalog_name") or (
                res.get("score") and float(res.get("score", 0)) > 0
            ):
                source = "Matcher"
            elif (res.get("bt_confidence") and float(res.get("bt_confidence", 0)) > 0) or res.get(
                "suggested_bt"
            ):
                source = "Classifier"
            else:
                source = "Matcher"

            if source == "Classifier":
                conf = compute_weighted_confidence(
                    res.get("bt_confidence"),
                    res.get("gk_confidence"),
                    res.get("region_confidence", res.get("category_confidence")),
                    has_bt=bool(res.get("suggested_bt")),
                    has_gk=bool(res.get("suggested_gk")),
                    has_region=bool(res.get("suggested_region") or res.get("suggested_category")),
                )
            else:
                conf = res.get("score", 0.0)

        input_entities = res.get("input_entities") or {}
        flavor_list = list(input_entities.get("flavor") or [])
        brand_list = list(input_entities.get("brand") or [])
        gk_val = [x.strip() for x in gk.split(",") if x.strip()] if isinstance(gk, str) else gk

        rows.append(
            {
                "id": str(uuid.uuid4()),
                "batch_id": job_id,
                "sku_index": i,
                "rule_version": rule_version,
                "sku_name": sku_name,
                "domain": domain,
                "bt": res.get("suggested_bt", ""),
                "gk_json": json.dumps(gk_val) if gk_val else "[]",
                "region": res.get("suggested_region", ""),
                "confidence": parse_optional_float(conf),
                "match_source": source,
                "rules_applied_json": res.get("rules_applied", ""),
                "logic_notes": res.get("logic_notes", ""),
                "matched_catalog_name": res.get("matched_catalog_name", ""),
                "match_score": parse_optional_float(res.get("score", 0.0)),
                "bt_confidence": (
                    parse_optional_float(res.get("bt_confidence"))
                    if source == "Classifier"
                    else None
                ),
                "gk_confidence": (
                    parse_optional_float(res.get("gk_confidence"))
                    if source == "Classifier"
                    else None
                ),
                "region_confidence": (
                    parse_optional_float(
                        res.get("region_confidence", res.get("category_confidence"))
                    )
                    if source == "Classifier"
                    else None
                ),
                "input_price": parse_optional_float(input_price),
                "input_description": input_description,
                "input_category": input_category,
                "flavor_extraction": json.dumps(flavor_list) if flavor_list else None,
                "brand_extraction": json.dumps(brand_list) if brand_list else None,
            }
        )
    return rows


def record_job_completion(
    job_id: str,
    domain: str,
    task: str,
    results: list[dict[str, Any]],
    rule_version: str,
    duration_minutes: float,
    high_conf: int,
    med_conf: int,
    low_conf: int,
    match_rate: float,
) -> int:
    """Bulk-upserts per-SKU results and marks the job + batch completed atomically."""
    written, rows = job_repository.complete_job_atomically(
        job_id=job_id,
        row_builder=lambda input_skus: build_processed_sku_rows(
            job_id, domain, task, input_skus, results, rule_version
        ),
        completed_items=len(results),
        duration_minutes=duration_minutes,
        high_conf=high_conf,
        med_conf=med_conf,
        low_conf=low_conf,
        match_rate=match_rate,
    )
    if written == -1:
        logger.warning(f"[JOB {job_id}] Ignoring completion for cancelled job.")
        job_progress_repository.clear_job_progress(job_id)
        return -1

    for row in rows:
        if row["confidence"] is not None:
            MATCH_CONFIDENCE.observe(row["confidence"])
    logger.info(f"[JOB {job_id}] Persisted {len(rows)} SKU results (rule_version={rule_version}).")

    job_progress_repository.clear_job_progress(job_id)
    job_progress_repository.publish_job_event(
        {
            "id": str(job_id),
            "status": "completed",
            "current_stage": "done",
            "progress_pct": 100.0,
            "eta_seconds": 0,
            "high_conf": high_conf,
            "med_conf": med_conf,
            "low_conf": low_conf,
            "match_rate": match_rate,
            "duration_minutes": duration_minutes,
        }
    )
    return written


def record_job_failure(
    job_id: str,
    error_message: str,
    duration_minutes: float = 0.0,
    is_cancelled: bool = False,
) -> None:
    """Marks a job (and its batch) failed or cancelled and broadcasts the terminal event."""
    job_repository.mark_job_failed_or_cancelled(
        job_id=job_id,
        error_message=error_message,
        duration_minutes=duration_minutes,
        is_cancelled=is_cancelled,
    )
    job_progress_repository.clear_job_progress(job_id)
    job_progress_repository.publish_job_event(
        {
            "id": str(job_id),
            "status": "cancelled" if is_cancelled else "failed",
            "current_stage": "cancelled" if is_cancelled else "failed",
            "error_message": error_message,
            "duration_minutes": duration_minutes,
            "eta_seconds": None,
        }
    )
