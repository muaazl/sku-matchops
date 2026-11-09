"""
SKU MatchOps Backend - PostgreSQL persistence of job outcomes.
Writes per-SKU results and terminal job status through the backend connection pool.
"""

import json
import logging
import uuid
from typing import Any

from sqlalchemy import text

from backend.app.db.session import get_engine, upsert_processed_skus
from engine.classification.tagger import compute_weighted_confidence
from engine.core.observability import MATCH_CONFIDENCE

logger = logging.getLogger("matchops.backend.persistence")


def _to_float(val) -> float | None:
    try:
        return float(val) if val not in (None, "") else None
    except (TypeError, ValueError):
        return None


def build_processed_sku_rows(
    job_id: str,
    domain: str,
    task: str,
    input_skus: list[dict],
    results: list[dict[str, Any]],
    rule_version: str,
    tenant_id: str = "default",
) -> list[dict[str, Any]]:
    """Maps engine results onto processed_skus rows, keyed by (job, input position, rule version)."""
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
        else:  # pipeline / upload
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
                "tenant_id": tenant_id,
                "sku_name": sku_name,
                "domain": domain,
                "bt": res.get("suggested_bt", ""),
                "gk_json": json.dumps(gk_val) if gk_val else "[]",
                "region": res.get("suggested_region", ""),
                "confidence": _to_float(conf),
                "match_source": source,
                "rules_applied_json": res.get("rules_applied", ""),
                "logic_notes": res.get("logic_notes", ""),
                "matched_catalog_name": res.get("matched_catalog_name", ""),
                "match_score": _to_float(res.get("score", 0.0)),
                "bt_confidence": _to_float(res.get("bt_confidence"))
                if source == "Classifier"
                else None,
                "gk_confidence": _to_float(res.get("gk_confidence"))
                if source == "Classifier"
                else None,
                "region_confidence": _to_float(
                    res.get("region_confidence", res.get("category_confidence"))
                )
                if source == "Classifier"
                else None,
                "input_price": _to_float(input_price),
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
    """Bulk-upserts the job's per-SKU results and marks the job completed, in one transaction."""
    with get_engine().begin() as conn:
        input_row = conn.execute(
            text("SELECT input_skus_json, tenant_id FROM jobs WHERE id = :id"), {"id": job_id}
        ).fetchone()
        input_skus = []
        if input_row and input_row[0]:
            try:
                input_skus = json.loads(input_row[0])
            except Exception:
                pass

        tenant_id = input_row[1] if input_row else "default"
        rows = build_processed_sku_rows(
            job_id, domain, task, input_skus, results, rule_version, tenant_id
        )
        upsert_processed_skus(conn, rows)

        conn.execute(
            text("""
                UPDATE jobs SET
                    status = 'completed',
                    current_stage = 'done',
                    completed_items = :completed_items,
                    completed_at = now(),
                    duration_minutes = :duration_minutes,
                    high_conf = :high_conf,
                    med_conf = :med_conf,
                    low_conf = :low_conf,
                    match_rate = :match_rate
                WHERE id = :id
            """),
            {
                "completed_items": len(results),
                "duration_minutes": duration_minutes,
                "high_conf": high_conf,
                "med_conf": med_conf,
                "low_conf": low_conf,
                "match_rate": match_rate,
                "id": job_id,
            },
        )
        # Mirror the terminal status onto the batches row, if this job originated from one
        conn.execute(
            text("UPDATE batches SET status = 'completed', completed_at = now() WHERE id = :id"),
            {"id": job_id},
        )
    for row in rows:
        if row["confidence"] is not None:
            MATCH_CONFIDENCE.observe(row["confidence"])
    logger.info(f"[JOB {job_id}] Persisted {len(rows)} SKU results (rule_version={rule_version}).")
    return len(rows)


def record_job_failure(
    job_id: str, error_message: str, duration_minutes: float = 0.0, is_cancelled: bool = False
) -> None:
    """Marks the job (and its batch, if any) failed or cancelled."""
    status = "cancelled" if is_cancelled else "failed"
    current_stage = "cancelled" if is_cancelled else "failed"
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE jobs SET
                    status = :status,
                    current_stage = :current_stage,
                    error_message = :error_message,
                    duration_minutes = :duration_minutes,
                    completed_at = now()
                WHERE id = :id
            """),
            {
                "status": status,
                "current_stage": current_stage,
                "error_message": error_message,
                "duration_minutes": duration_minutes,
                "id": job_id,
            },
        )
        conn.execute(
            text("UPDATE batches SET status = :status, completed_at = now() WHERE id = :id"),
            {"status": status, "id": job_id},
        )
