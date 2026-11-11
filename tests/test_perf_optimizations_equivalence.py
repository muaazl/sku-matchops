import csv
import io
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from backend.app.api.endpoints.history import _sanitize_csv_cell, export_history
from backend.app.api.endpoints.jobs import _enrich_job_rows
from backend.app.core.security import Principal
from engine import config
from engine.data_pipeline.vector_store import VectorStore
from engine.matching.logic_gates import LogicGates
from engine.matching.matcher import SKUMatcher
from engine.nlp.embedding_engine import EmbeddingEngine
from engine.nlp.ner_engine import NEREngine
from engine.nlp.text_cleaner import TextPipeline
from engine.pipeline import worker_runner


def _norm_row(row: dict) -> dict:
    clean = TextPipeline.normalize_final(TextPipeline.standardize_units(row["Name"]))
    return {
        **row,
        "clean_text": row.get("clean_text", clean),
        "clean_no_weights": row.get("clean_no_weights", TextPipeline.strip_weights(clean)),
    }


def _make_matcher(catalog_rows: list[dict], classifier=None, domain: str = "market"):
    catalog_df = pd.DataFrame([_norm_row(r) for r in catalog_rows])
    mock_cache = MagicMock()
    mock_vs = MagicMock()
    mock_cache.manage_catalog_cache.return_value = (catalog_df, mock_vs)
    mock_ner = MagicMock()
    mock_ner.batch_extract_entities.side_effect = lambda texts: [{} for _ in texts]
    mock_embed = MagicMock()
    logic_gates = LogicGates(embed_engine=mock_embed)
    matcher = SKUMatcher(
        catalog_df=catalog_df,
        brands_df=pd.DataFrame(),
        ner_engine=mock_ner,
        embed_engine=mock_embed,
        cache_manager=mock_cache,
        logic_gates=logic_gates,
        domain=domain,
        classifier=classifier,
    )
    return matcher, mock_embed, mock_vs, mock_ner


def test_matcher_skips_onnx_on_unambiguous_exact_and_multisize_fuzzy_bypass():
    """Items 4 & 11: Unambiguous single-BT exact and multi-size fuzzy bypasses skip ONNX embed + BT."""
    catalog_rows = [
        {
            "Name": "Coca Cola Can 330ml",
            "Brand": "Coca Cola",
            "basictype": "Soft Drinks",
            "Generic keywords": "Cola, Soda",
            "weight_val": (330.0, "ml", "liquid"),
            "entities": {"brand": {"coca cola"}, "flavor": set()},
        },
        {
            "Name": "Smak Hot & Spicy Mixture 200g",
            "Brand": "Smak",
            "basictype": "Snacks",
            "Generic keywords": "Mixture",
            "weight_val": (200.0, "g", "solid"),
            "entities": {"brand": {"smak"}, "flavor": {"hot", "spicy"}},
        },
        {
            "Name": "Smak Hot & Spicy Mixture 150g",
            "Brand": "Smak",
            "basictype": "Snacks",
            "Generic keywords": "Mixture",
            "weight_val": (150.0, "g", "solid"),
            "entities": {"brand": {"smak"}, "flavor": {"hot", "spicy"}},
        },
    ]
    mock_classifier = MagicMock()
    matcher, mock_embed, mock_vs, _ = _make_matcher(catalog_rows, classifier=mock_classifier)

    input_df = pd.DataFrame(
        [
            {"Name": "Coca Cola Can 330ML"},  # Exact match
            {
                "Name": "Smak Spicy Hot Mixture 150G"
            },  # Multi-size token-sorted fuzzy match (same BT)
        ]
    )
    res = matcher.process_inputs(input_df)

    # Zero ONNX embedding, zero ArcFace BT classification, zero Qdrant searches
    mock_embed.embed_weighted_sku.assert_not_called()
    mock_classifier.batch_predict_bt.assert_not_called()
    mock_vs.search_batch_from_vectors.assert_not_called()

    assert res.iloc[0]["Matched Catalog Name"] == "Coca Cola Can 330ml"
    assert res.iloc[0]["Final Score"] == 1.0
    assert res.iloc[0]["Status"] == "High Confidence"
    assert res.iloc[0]["Logic Notes"] == "Exact Text Match"

    assert res.iloc[1]["Matched Catalog Name"] == "Smak Hot & Spicy Mixture 150g"
    assert res.iloc[1]["Final Score"] == 1.0
    assert res.iloc[1]["Status"] == "High Confidence"
    assert "Fuzzy Match (100%)" in res.iloc[1]["Logic Notes"]


def test_matcher_multi_bt_bypass_runs_bt_only_for_ambiguous_rows():
    """Item 4: Ambiguous multi-BT exact/fuzzy collisions run BT tie-breaking accurately."""
    catalog_rows = [
        {
            "Name": "Classic Mint 100g",
            "basictype": "Confectionery",
            "Generic keywords": "Mint",
            "weight_val": (100.0, "g", "solid"),
        },
        {
            "Name": "Classic Mint 100g",
            "basictype": "Herbs",
            "Generic keywords": "Fresh Herbs",
            "weight_val": (100.0, "g", "solid"),
        },
        {
            "Name": "Pure Honey 500g",
            "basictype": "Spreads",
            "Generic keywords": "Honey",
            "weight_val": (500.0, "g", "solid"),
        },
    ]
    mock_classifier = MagicMock()
    mock_classifier.batch_predict_bt.return_value = [("Herbs", 0.95, "arcface", None)]
    matcher, mock_embed, _, _ = _make_matcher(catalog_rows, classifier=mock_classifier)
    mock_embed.embed_weighted_sku.return_value = {
        "dense": np.ones((1, 8), dtype=np.float32),
        "sparse": [{1: 0.8}],
    }

    input_df = pd.DataFrame(
        [
            {"Name": "Pure Honey 500g"},  # Unambiguous -> skipped from ONNX
            {"Name": "Classic Mint 100g"},  # Multi-BT -> encoded & classified
        ]
    )
    res = matcher.process_inputs(input_df)

    expected_clean = TextPipeline.normalize_final(
        TextPipeline.standardize_units("Classic Mint 100g")
    )
    assert mock_embed.embed_weighted_sku.call_count == 1
    assert mock_embed.embed_weighted_sku.call_args[0][0] == [expected_clean]
    assert res.iloc[0]["BasicType"] == "Spreads"
    assert res.iloc[1]["BasicType"] == "Herbs"


def test_matcher_candidate_evaluation_matches_dataframe_reference_and_skips_noop_strip():
    """Item 11: Dict candidate evaluation & brand-strip guard match exact legacy DataFrame behavior."""
    catalog_rows = [
        {
            "Name": "Acme Almond Milk 1L",
            "basictype": "Dairy Alternatives",
            "Generic keywords": "Milk, Almond, Vanilla",
            "category": "Beverages",
            "region": "Global",
            "weight_val": (1000.0, "ml", "liquid"),
            "entities": {"brand": {"acme"}, "flavor": {"vanilla"}},
        }
    ]
    mock_classifier = MagicMock()
    mock_classifier.batch_predict_bt.return_value = [
        ("Dairy Alternatives", 0.92, "arcface", None),
        ("Snacks", 0.88, "arcface", None),
    ]
    matcher, mock_embed, mock_vs, mock_ner = _make_matcher(catalog_rows, classifier=mock_classifier)

    # SKU 0 has brand "acme" present in clean_input -> stripped != clean_input (runs stripped search)
    # SKU 1 has brand "ghostbrand" NOT in clean_input -> stripped == clean_input (skips no-op stripped search)
    mock_ner.batch_extract_entities.side_effect = None
    mock_ner.batch_extract_entities.return_value = [
        {"brand": {"acme"}, "flavor": {"vanilla"}},
        {"brand": {"ghostbrand"}, "flavor": {"vanilla"}},
    ]

    mock_embed.embed_weighted_sku.side_effect = [
        {
            "dense": np.ones((2, 8), dtype=np.float32),
            "sparse": [{1: 0.9}, {2: 0.7}],
        },
        {
            "dense": np.ones((1, 8), dtype=np.float32),
            "sparse": [{3: 0.8}],
        },
    ]

    cand_hit = {
        **_norm_row(catalog_rows[0]),
        "_qdrant_score_": 0.91,
    }
    cand_alt = {
        **_norm_row(
            {
                "Name": "Snack Bar 50g",
                "basictype": "Snacks",
                "Generic keywords": "Bar, Chocolate",
                "category": "Snacks",
                "region": "Global",
                "weight_val": (50.0, "g", "solid"),
                "entities": {"brand": set(), "flavor": {"chocolate"}},
            }
        ),
        "_qdrant_score_": 0.72,
    }

    mock_vs.search_batch_from_vectors.return_value = [
        [cand_hit],  # SKU 0 filtered
        [cand_hit],  # SKU 0 unfiltered
        [cand_hit],  # SKU 0 stripped
        [cand_alt],  # SKU 1 filtered (low score -> fallback)
        [cand_alt, cand_hit],  # SKU 1 unfiltered
    ]
    mock_embed.score_cross_encoder.side_effect = lambda pairs: np.array(
        [4.5 if "almond" in p[1] else -2.0 for p in pairs], dtype=np.float32
    )

    input_df = pd.DataFrame(
        [
            {"Name": "Acme Vanilla Almond Milk 1L"},
            {"Name": "Vanilla Almond Beverage 1L"},
        ]
    )
    res = matcher.process_inputs(input_df)

    expected_stripped = TextPipeline.normalize_final(
        TextPipeline.standardize_units("Vanilla Almond Milk 1L")
    )
    assert mock_embed.embed_weighted_sku.call_count == 2
    stripped_call_names = mock_embed.embed_weighted_sku.call_args_list[1][0][0]
    assert stripped_call_names == [expected_stripped]
    assert len(mock_vs.search_batch_from_vectors.call_args[0][0]) == 5

    assert res.iloc[0]["Matched Catalog Name"] == "Acme Almond Milk 1L"
    assert res.iloc[0]["Status"] == "High Confidence"
    assert res.iloc[0]["Logic Notes"].startswith("BT-Filtered (Dairy Alternatives) |")
    assert res.iloc[0]["GenericKeywords"] == "Milk, Almond"

    assert res.iloc[1]["Matched Catalog Name"] == "Acme Almond Milk 1L"
    assert res.iloc[1]["Logic Notes"].startswith("BT-Fallback (BT: Snacks,")


def test_cross_encoder_fp32_cache_vs_int8_exact_batch_preservation():
    """Item 11: FP32 Cross-Encoder deduplicates + caches; INT8 preserves exact batch composition."""
    eng = EmbeddingEngine.__new__(EmbeddingEngine)
    eng.cross_encoder_fallback = None
    eng.rerank_tokenizer = MagicMock(
        side_effect=lambda t1, t2, **kw: {
            "input_ids": np.ones((len(t1), 4), dtype=np.int64),
            "attention_mask": np.ones((len(t1), 4), dtype=np.int64),
        }
    )
    eng.cross_session = MagicMock()

    # 1. INT8 mode: preserves exact dynamic quantization batch composition
    eng._cross_is_int8 = True
    eng.cross_session.run.side_effect = lambda _, inp: [
        np.arange(len(inp["input_ids"]), dtype=np.float32)
    ]
    pairs = [["sku a", "cand 1"], ["sku a", "cand 1"], ["sku a", "cand 2"]]
    out_int8 = eng.score_cross_encoder(pairs, batch_size=64)
    assert len(out_int8) == 3
    assert eng.rerank_tokenizer.call_args[0][0] == ["sku a", "sku a", "sku a"]

    # 2. FP32 mode: deduplicates duplicate pairs and serves subsequent calls from LRU cache
    eng._cross_is_int8 = False
    eng.rerank_tokenizer.reset_mock()
    eng.cross_session.reset_mock()
    eng.cross_session.run.side_effect = lambda _, inp: [
        np.array(
            [1.25 if idx == 0 else -0.75 for idx in range(len(inp["input_ids"]))],
            dtype=np.float32,
        )
    ]

    out_fp32_first = eng.score_cross_encoder(pairs, batch_size=64)
    assert np.allclose(out_fp32_first, np.array([1.25, 1.25, -0.75], dtype=np.float32))
    assert len(eng.rerank_tokenizer.call_args[0][0]) == 2

    out_fp32_second = eng.score_cross_encoder(pairs, batch_size=64)
    assert np.allclose(out_fp32_second, out_fp32_first)
    assert eng.cross_session.run.call_count == 1


def test_ner_engine_deduplicates_and_caches_food_prepass_to_tagger():
    """Item 5: processor.py food prepass and tagger.py share identical NER cache keys (zero duplicate GLiNER)."""
    ner = NEREngine.__new__(NEREngine)
    ner.domain = config.DOMAIN_FOOD
    ner.labels = ["flavor"]
    ner.target_label = "flavor"
    ner._ner_cache = OrderedDict()
    ner._ner_cache_max = 50000
    ner._get_dict_entities = lambda _txt: (set(), set())
    ner._resolve_entity_text = lambda txt: {txt.lower()}

    mock_model = MagicMock()
    mock_model.inference.side_effect = lambda texts, labels, threshold=0.25, batch_size=32: [
        [{"label": "flavor", "text": "peri peri"}] for _ in texts
    ]
    ner.model = mock_model

    raw_name, raw_desc, raw_cat = "Grilled Peri Peri Platter", "Served hot", "Mains"
    prepass_txt = TextPipeline.prep_for_ner(
        TextPipeline.build_ner_input(raw_name, raw_desc, raw_cat)
    )
    tagger_txt = TextPipeline.prep_for_ner(
        TextPipeline.build_ner_input(raw_name, raw_desc, raw_cat)
    )
    assert prepass_txt == tagger_txt

    # Pass 1 with duplicate rows in same batch -> GLiNER runs once on 1 unique text
    res1 = ner.batch_extract_entities([prepass_txt, prepass_txt])
    assert mock_model.inference.call_count == 1
    assert len(mock_model.inference.call_args[0][0]) == 1
    assert res1[0]["flavor"] == {"peri peri"}
    assert res1[1]["flavor"] == {"peri peri"}

    # Mutating returned set must not corrupt cached sets
    res1[0]["flavor"].add("mutated")

    # Pass 2 (Tagger) -> 100% cache hit, 0 GLiNER calls, clean unmutated entities
    res2 = ner.batch_extract_entities([tagger_txt])
    assert mock_model.inference.call_count == 1
    assert res2[0]["flavor"] == {"peri peri"}


def test_vector_store_retry_rebinds_client_and_falls_back_from_grpc():
    """Items 2, 9, 10: _call_with_retry rebinds bound client methods, falls back on gRPC UNAVAILABLE, skips NOT_FOUND."""
    vs = VectorStore.__new__(VectorStore)

    client_1 = MagicMock(name="client_1")
    client_2 = MagicMock(name="client_2")
    client_3 = MagicMock(name="client_3")

    class FakeUnavailableRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

    client_1.get_collections.side_effect = FakeUnavailableRpcError("grpc port 6334 refused")
    client_2.get_collections.return_value = SimpleNamespace(collections=["ok"])
    client_3.get_collections.return_value = SimpleNamespace(collections=["reconnected"])

    orig_client = VectorStore._client
    orig_override = VectorStore._prefer_grpc_override
    try:
        VectorStore._client = client_1
        VectorStore._prefer_grpc_override = None
        vs.client = client_1

        with (
            patch("engine.data_pipeline.vector_store.time.sleep"),
            patch(
                "engine.data_pipeline.vector_store.QdrantClient", return_value=client_2
            ) as mock_qc,
        ):
            out = vs._call_with_retry("get_collections", vs.client.get_collections)

        assert out.collections == ["ok"]
        assert VectorStore._prefer_grpc_override is False
        assert mock_qc.call_args.kwargs["prefer_grpc"] is False
        client_1.close.assert_called_once()
        client_2.get_collections.assert_called_once()

        # Also verify attempt >= 2 transient network reconnect rebinds bound method
        client_2.get_collections.side_effect = ConnectionError("connection reset")
        with (
            patch("engine.data_pipeline.vector_store.time.sleep"),
            patch("engine.data_pipeline.vector_store.QdrantClient", return_value=client_3),
        ):
            out2 = vs._call_with_retry("get_collections", vs.client.get_collections)
        assert out2.collections == ["reconnected"]
        client_2.close.assert_called_once()
        client_3.get_collections.assert_called_once()

        # Non-transient NOT_FOUND error must NOT retry
        class FakeNotFoundError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.NOT_FOUND

        client_3.get_collection.side_effect = FakeNotFoundError("not found")
        with pytest.raises(FakeNotFoundError):
            vs._call_with_retry("get_collection", vs.client.get_collection, "missing_col")
        assert client_3.get_collection.call_count == 1
    finally:
        VectorStore._client = orig_client
        VectorStore._prefer_grpc_override = orig_override


def test_vector_store_search_hybrid_tags_uses_single_batch_query_and_minimal_payload():
    """Item 10: search_hybrid_tags combines dense+sparse into 1 query_batch_points call with with_payload=['tag']."""
    vs = VectorStore.__new__(VectorStore)
    vs._prefer_grpc = True
    vs._verified_collections = {"market_tags"}
    vs.client = MagicMock()
    dense_pt = SimpleNamespace(payload={"tag": "Beverages"}, score=0.91)
    sparse_pt = SimpleNamespace(payload={"tag": "Soft Drinks"}, score=0.84)
    vs.client.query_batch_points.return_value = [
        SimpleNamespace(points=[dense_pt]),
        SimpleNamespace(points=[sparse_pt]),
    ]

    dense_hits, sparse_hits = vs.search_hybrid_tags(
        dense_query=np.ones(8, dtype=np.float32),
        sparse_query={10: 0.5},
        limit=5,
        filter_dict_type="BT",
        domain="market",
    )

    vs.client.query_batch_points.assert_called_once()
    call_kwargs = vs.client.query_batch_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "market_tags"
    assert len(call_kwargs["requests"]) == 2
    assert all(req.with_payload == ["tag"] for req in call_kwargs["requests"])
    assert dense_hits == [dense_pt]
    assert sparse_hits == [sparse_pt]


def test_worker_exec_gate_releases_before_outbound_webhook():
    """Item 3: _execute_job releases _exec_lock before callback_url POST so next job starts inference immediately."""
    webhook_started = threading.Event()
    allow_webhook_finish = threading.Event()
    second_worker_acquired_lock = threading.Event()

    def slow_webhook(*args, **kwargs):
        webhook_started.set()
        allow_webhook_finish.wait(timeout=3.0)
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        return resp

    payload = {
        "job_id": "job-1",
        "skus": [{"Name": "Coca Cola 330ml"}],
        "task": "matcher",
        "domain": "market",
        "callback_url": "https://example.invalid/webhook",
    }

    with (
        patch.object(worker_runner, "sync_rules_version"),
        patch.object(worker_runner, "snapshot_rules", return_value=SimpleNamespace(version="v1")),
        patch.object(worker_runner, "pinned_rules"),
        patch.object(
            worker_runner,
            "process_request",
            return_value={"results": [{"status": "High Confidence"}]},
        ),
        patch.object(worker_runner._http_session, "post", side_effect=slow_webhook),
        patch.object(worker_runner, "_get_backend_stub") as mock_stub_fn,
    ):
        mock_stub = MagicMock()
        mock_stub_fn.return_value = mock_stub

        def run_job_1():
            worker_runner._exec_lock.acquire()
            try:
                worker_runner._execute_job(payload)
            finally:
                worker_runner._exec_lock.release()

        t1 = threading.Thread(target=run_job_1)
        t1.start()

        assert webhook_started.wait(timeout=2.0), "Webhook should have started"

        def run_job_2_probe():
            with worker_runner._exec_lock:
                second_worker_acquired_lock.set()

        t2 = threading.Thread(target=run_job_2_probe)
        t2.start()
        assert second_worker_acquired_lock.wait(timeout=1.0), (
            "_exec_lock remained held during outbound webhook POST!"
        )
        t2.join(timeout=1.0)

        allow_webhook_finish.set()
        t1.join(timeout=2.0)


def test_backend_enrich_job_rows_skips_redis_unless_running():
    """Item 14: 0 Redis calls when no jobs are running; 1 pipelined batch call when multiple are running."""
    non_running = [
        {"id": "1", "status": "completed", "total_items": 10},
        {"id": "2", "status": "failed", "total_items": 10},
        {"id": "3", "status": "cancelled", "total_items": 10},
        {"id": "4", "status": "queued", "total_items": 100},
    ]
    with (
        patch("backend.app.api.endpoints.jobs.get_bulk_job_progress_states") as mock_bulk,
        patch("backend.app.api.endpoints.jobs.get_job_progress_state") as mock_single,
        patch.dict("backend.app.api.endpoints.jobs._job_progress", {"2": 30.0, "3": 50.0}),
    ):
        enriched = _enrich_job_rows(non_running)
        mock_bulk.assert_not_called()
        mock_single.assert_not_called()
        assert enriched[0]["progress_pct"] == 100.0
        assert enriched[1]["progress_pct"] == 30.0
        assert enriched[2]["progress_pct"] == 50.0
        assert enriched[3]["progress_pct"] == 0.0

    mixed_with_running = [
        {"id": "10", "status": "running", "total_items": 50},
        {"id": "11", "status": "running", "total_items": 80},
        {"id": "12", "status": "completed", "total_items": 20},
    ]
    with (
        patch(
            "backend.app.api.endpoints.jobs.get_bulk_job_progress_states",
            return_value={"10": (42.5, 12), "11": (85.0, 4)},
        ) as mock_bulk,
        patch("backend.app.api.endpoints.jobs.get_job_progress_state") as mock_single,
    ):
        enriched_run = _enrich_job_rows(mixed_with_running)
        mock_bulk.assert_called_once_with(["10", "11"])
        mock_single.assert_not_called()
        assert enriched_run[0]["progress_pct"] == 42.5
        assert enriched_run[0]["eta_seconds"] == 12
        assert enriched_run[1]["progress_pct"] == 85.0
        assert enriched_run[1]["eta_seconds"] == 4


def test_streaming_csv_export_matches_buffered_csv_and_sanitizes_formulas():
    """Item 16: Streaming CSV export matches buffered DictWriter byte-for-byte and validates format upfront."""
    sample_rows = [
        {
            "id": "sku-1",
            "sku_name": "Coca Cola 330ml",
            "bt": "Soft Drinks",
            "confidence": 0.99,
            "formula_col": "=CMD|' /C calc'!A0",
        },
        {
            "id": "sku-2",
            "sku_name": "+Plus Energy 250ml",
            "bt": "Energy Drinks",
            "confidence": 0.87,
            "formula_col": "@SUM(1,2)",
        },
    ]

    mock_mappings = MagicMock()
    mock_mappings.partitions.return_value = [[sample_rows[0]], [sample_rows[1]]]
    mock_result = MagicMock()
    mock_result.mappings.return_value = mock_mappings
    mock_conn = MagicMock()
    mock_conn.execution_options.return_value = mock_conn
    mock_conn.execute.return_value = mock_result

    principal = Principal(username="tester", role="admin")

    with pytest.raises(HTTPException) as exc_info:
        export_history(ids=None, format="xlsx", db=mock_conn, principal=principal)
    assert exc_info.value.status_code == 400

    with patch("backend.app.api.endpoints.history.StreamingResponse") as mock_sr:
        export_history(ids="sku-1,sku-2", format="csv", db=mock_conn, principal=principal)
        gen = mock_sr.call_args[0][0]
        streamed_csv = "".join(list(gen))
    mock_result.close.assert_called_once()

    ref_buf = io.StringIO()
    ref_writer = csv.DictWriter(ref_buf, fieldnames=sample_rows[0].keys())
    ref_writer.writeheader()
    for row in sample_rows:
        ref_writer.writerow({k: _sanitize_csv_cell(v) for k, v in row.items()})

    assert streamed_csv == ref_buf.getvalue()
    assert "\t=CMD|" in streamed_csv
    assert "\t+Plus Energy 250ml" in streamed_csv
    assert "\t@SUM(1,2)" in streamed_csv


def test_graceful_shutdown_lease_release_and_fast_reclaim():
    """SIGTERM shutdown releases active idempotent leases immediately (0 attempts/0s); hard-crash first reclaim has 0s delay."""
    import json

    from engine.pipeline import task_queue
    from engine.pipeline.task_queue import INFLIGHT, REALTIME, TaskQueue

    fake_r = MagicMock()
    fake_pipe = MagicMock()
    fake_r.pipeline.return_value = fake_pipe
    fake_r.register_script.return_value = MagicMock()

    q = TaskQueue(fake_r, max_attempts=3, lease_seconds=30.0, backoff_seconds=5.0)

    # 1. Graceful shutdown releases pipeline task immediately without burning attempts
    active_task = {
        "task_id": "t-1244",
        "job_id": "1244",
        "queue": REALTIME,
        "attempts": 0,
        "max_attempts": 3,
        "payload": {"job_id": "1244", "task": "pipeline"},
    }
    upload_task = {
        "task_id": "t-up",
        "job_id": "1245",
        "queue": REALTIME,
        "attempts": 0,
        "max_attempts": 1,
        "payload": {"job_id": "1245", "task": "upload"},
    }
    fake_r.zrem.return_value = 1
    try:
        with task_queue._active_lock:
            task_queue._active_tasks["t-1244"] = (q, active_task)
            task_queue._active_tasks["t-up"] = (q, upload_task)

        worker_runner.shutdown_workers()
        assert worker_runner._worker_stop.is_set()
        fake_r.zrem.assert_called_once_with(INFLIGHT, "t-1244")
        fake_r.lpush.assert_called_once_with(REALTIME, "t-1244")
        assert active_task["attempts"] == 0
    finally:
        with task_queue._active_lock:
            task_queue._active_tasks.clear()
        worker_runner._worker_stop.clear()

    # 2. Hard crash: first lease-expiry reclaim uses 0.0s delay; second uses exponential backoff
    fake_r.reset_mock()
    fake_pipe.reset_mock()
    fake_r.zrangebyscore.return_value = ["t-1244"]
    fake_r.zrem.return_value = 1
    fake_r.hget.return_value = json.dumps(active_task)

    with patch("engine.pipeline.task_queue._now", return_value=1000.0):
        q.reclaim_expired()
    first_zadd_mapping = fake_pipe.zadd.call_args[0][1]
    assert first_zadd_mapping["t-1244"] == 1000.0  # 0.0s delay on attempt 1!

    saved_after_1 = json.loads(fake_pipe.hset.call_args[0][2])
    assert saved_after_1["attempts"] == 1

    fake_pipe.reset_mock()
    fake_r.hget.return_value = json.dumps(saved_after_1)
    with patch("engine.pipeline.task_queue._now", return_value=1000.0):
        q.reclaim_expired()
    second_zadd_mapping = fake_pipe.zadd.call_args[0][1]
    assert second_zadd_mapping["t-1244"] == 1010.0  # 5.0 * 2**(2-1) = 10.0s backoff on attempt 2!
