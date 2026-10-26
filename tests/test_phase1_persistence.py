"""Phase 1 persistence: PostgreSQL schema, bulk batch inserts, and idempotent re-delivery."""

import json
import uuid

import pytest
from sqlalchemy import event, inspect, text

from engine.core.db import upsert_processed_skus
from engine.pipeline.persistence import record_job_completion

pytestmark = pytest.mark.usefixtures("pg")


@pytest.fixture
def job_id(pg):
    """A unique job id whose rows are removed after the test."""
    jid = f"test-{uuid.uuid4().hex[:12]}"
    yield jid
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM processed_skus WHERE batch_id = :id"), {"id": jid})
        conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": jid})


def _rows(job_id: str, n: int, rule_version: str = "rv1", confidence: float = 0.5) -> list[dict]:
    return [
        {
            "id": str(uuid.uuid4()),
            "batch_id": job_id,
            "sku_index": i,
            "rule_version": rule_version,
            "sku_name": f"SKU {i}",
            "domain": "market",
            "bt": "Butter",
            "gk_json": "[]",
            "confidence": confidence,
            "match_source": "matcher",
        }
        for i in range(n)
    ]


def _count(pg, job_id: str) -> int:
    with pg.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM processed_skus WHERE batch_id = :id"), {"id": job_id}
        ).scalar()


def _count_statements(pg):
    """Counts INSERT statements sent to the server while the returned list is attached."""
    inserts = []

    def _before(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("INSERT INTO PROCESSED_SKUS"):
            inserts.append(statement)

    event.listen(pg, "before_cursor_execute", _before)
    return inserts, lambda: event.remove(pg, "before_cursor_execute", _before)


def test_schema_created_with_lookup_indexes_and_idempotency_key(pg):
    insp = inspect(pg)
    tables = set(insp.get_table_names())
    assert {
        "jobs",
        "batches",
        "rules",
        "conditions",
        "actions",
        "processed_skus",
        "api_requests",
        "catalog_items",
        "brand_flavors",
        "classifier_dictionaries",
        "bt_gk_map",
        "alembic_version",
    } <= tables

    uniques = {u["name"]: u["column_names"] for u in insp.get_unique_constraints("processed_skus")}
    assert uniques["uq_processed_skus_job_sku_rule"] == ["batch_id", "sku_index", "rule_version"]

    def indexed(table):
        return {tuple(ix["column_names"]) for ix in insp.get_indexes(table)}

    assert ("batch_id",) in indexed("processed_skus")
    assert ("status", "started_at") in indexed("jobs")
    assert ("domain", "priority") in indexed("rules")
    assert ("rule_id",) in indexed("conditions")
    assert ("rule_id",) in indexed("actions")

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
    with pg.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == head


def test_bulk_insert_1000_plus_rows_in_single_batch(pg, job_id):
    inserts, detach = _count_statements(pg)
    try:
        with pg.begin() as conn:
            written = upsert_processed_skus(conn, _rows(job_id, 1200), chunk_size=1200)
    finally:
        detach()

    assert written == 1200
    assert len(inserts) == 1, "expected one multi-row INSERT for the whole batch"
    assert _count(pg, job_id) == 1200


def test_bulk_insert_flushes_in_configured_chunks(pg, job_id):
    inserts, detach = _count_statements(pg)
    try:
        with pg.begin() as conn:
            upsert_processed_skus(conn, _rows(job_id, 2500), chunk_size=1000)
    finally:
        detach()

    assert len(inserts) == 3  # 1000 + 1000 + 500
    assert _count(pg, job_id) == 2500


def test_reinserting_same_batch_updates_instead_of_duplicating(pg, job_id):
    with pg.begin() as conn:
        upsert_processed_skus(conn, _rows(job_id, 1000, confidence=0.5))
    with pg.connect() as conn:
        first_ids = dict(
            conn.execute(
                text("SELECT sku_index, id FROM processed_skus WHERE batch_id = :id"),
                {"id": job_id},
            ).fetchall()
        )

    # Retry of the same batch (fresh uuids, new values) must not crash or duplicate
    with pg.begin() as conn:
        upsert_processed_skus(conn, _rows(job_id, 1000, confidence=0.9))

    assert _count(pg, job_id) == 1000
    with pg.connect() as conn:
        rows = conn.execute(
            text("SELECT sku_index, id, confidence FROM processed_skus WHERE batch_id = :id"),
            {"id": job_id},
        ).fetchall()
    assert all(conf == 0.9 for _, _, conf in rows)
    assert {idx: rid for idx, rid, _ in rows} == first_ids  # rows updated in place


def test_new_rule_version_is_a_distinct_result_set(pg, job_id):
    with pg.begin() as conn:
        upsert_processed_skus(conn, _rows(job_id, 10, rule_version="rv1"))
        upsert_processed_skus(conn, _rows(job_id, 10, rule_version="rv2"))
    assert _count(pg, job_id) == 20


def test_engine_job_completion_is_idempotent_on_retry(pg, job_id):
    skus = [
        {"name": f"Item {i}", "price": "", "description": "d", "category": "c"} for i in range(1100)
    ]
    with pg.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO jobs (id, batch_id, type, status, current_stage, total_items, started_at, domain, input_skus_json) "
                "VALUES (:id, :id, 'pipeline', 'running', 'matching', :n, now(), 'market', :skus)"
            ),
            {"id": job_id, "n": len(skus), "skus": json.dumps(skus)},
        )
    results = [
        {
            "suggested_bt": "Butter",
            "suggested_gk": "Dairy, Butter",
            "score": 0.92,
            "bt_confidence": 0.9,
        }
        for _ in skus
    ]
    kwargs = {
        "job_id": job_id,
        "domain": "market",
        "task": "pipeline",
        "results": results,
        "rule_version": "rv1",
        "duration_minutes": 0.5,
        "high_conf": len(skus),
        "med_conf": 0,
        "low_conf": 0,
        "match_rate": 100.0,
    }

    assert record_job_completion(**kwargs) == 1100
    assert record_job_completion(**kwargs) == 1100  # engine retry of the same batch

    assert _count(pg, job_id) == 1100
    with pg.connect() as conn:
        job = (
            conn.execute(text("SELECT * FROM jobs WHERE id = :id"), {"id": job_id})
            .mappings()
            .fetchone()
        )
        sample = (
            conn.execute(
                text("SELECT * FROM processed_skus WHERE batch_id = :id AND sku_index = 7"),
                {"id": job_id},
            )
            .mappings()
            .fetchone()
        )
    assert job["status"] == "completed"
    assert job["completed_items"] == 1100
    assert sample["sku_name"] == "Item 7"
    assert sample["input_price"] is None  # blank price from the sheet stored as NULL
    assert json.loads(sample["gk_json"]) == ["Dairy", "Butter"]
