"""
SKU MatchOps Engine - Direct PostgreSQL persistence of job outcomes.
Writes per-SKU results and terminal job status through the shared connection pool.
"""

import json
import logging
import uuid
from typing import Any

from sqlalchemy import text

from engine.core.db import get_engine, upsert_processed_skus
from engine.core.observability import MATCH_CONFIDENCE

logger = logging.getLogger("matchops.engine.persistence")


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
            conf = res.get("bt_confidence", 0.0)
            source = "Classifier"
        else:  # pipeline / upload
            conf = res.get("bt_confidence", res.get("score", 0.0))
            p_src = str(res.get("pipeline_source") or "").strip().lower()
            if p_src == "classifier":
                source = "Classifier"
            elif p_src in ("matcher", "catalogue"):
                source = "Matcher"
            elif res.get("matched_catalog_name") or (res.get("score") and float(res.get("score", 0)) > 0):
                source = "Matcher"
            elif (res.get("bt_confidence") and float(res.get("bt_confidence", 0)) > 0) or res.get("suggested_bt"):
                source = "Classifier"
            else:
                source = "Matcher"

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
                "bt_confidence": _to_float(res.get("bt_confidence", 0.0)),
                "gk_confidence": _to_float(res.get("gk_confidence", 0.0)),
                "region_confidence": _to_float(
                    res.get("region_confidence", res.get("category_confidence", 0.0))
                ),
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
        # (CSV upload/merchant fetch) — a no-op UPDATE for jobs with no corresponding batch.
        conn.execute(
            text("UPDATE batches SET status = 'completed', completed_at = now() WHERE id = :id"),
            {"id": job_id},
        )
    for row in rows:
        if row["confidence"] is not None:
            MATCH_CONFIDENCE.observe(row["confidence"])
    logger.info(f"[JOB {job_id}] Persisted {len(rows)} SKU results (rule_version={rule_version}).")
    return len(rows)


def record_job_failure(job_id: str, error_message: str, duration_minutes: float) -> None:
    """Marks the job (and its batch, if any) failed."""
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE jobs SET
                    status = 'failed',
                    current_stage = 'failed',
                    error_message = :error_message,
                    duration_minutes = :duration_minutes,
                    completed_at = now()
                WHERE id = :id
            """),
            {"error_message": error_message, "duration_minutes": duration_minutes, "id": job_id},
        )
        conn.execute(
            text("UPDATE batches SET status = 'failed', completed_at = now() WHERE id = :id"),
            {"id": job_id},
        )
