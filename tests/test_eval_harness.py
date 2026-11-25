"""End-to-end synthetic & unit tests for evaluation/metrics.py, evaluation/run_eval.py, and evaluation/report.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from engine.core import config
from engine.inference.reranker_model import RerankerModel
from evaluation import metrics as M
from evaluation import model_manager, report, run_eval


def test_metrics_classification_multilabel_ranking_and_reranker_lift():
    gt_bt = ["Burger", "Burger", "Fried Rice", "Pizza"]
    pred_bt = ["Burger", "Pizza", "Fried Rice", "Pizza"]
    confs = [0.95, 0.45, 0.88, 0.91]
    statuses = ["AUTO", "REVIEW", "AUTO", "AUTO"]
    topk = [
        ["Burger", "Sandwich", "Wrap"],
        ["Pizza", "Burger", "Bun"],
        ["Fried Rice", "Noodles", "Kottu"],
        ["Pizza", "Pasta", "Calzone"],
    ]
    cls_m = M.compute_multiclass_metrics(
        gt_bt,
        pred_bt,
        confidences=confs,
        statuses=statuses,
        topk_preds=topk,
        train_class_counts={"Burger": 20, "Fried Rice": 5, "Pizza": 0},
        seen_sku_mask=[True, False, True, False],
    )
    assert cls_m["accuracy_pct"] == 75.0
    assert cls_m["top3_acc_pct"] == 100.0
    assert cls_m["by_status"]["AUTO"]["accuracy_pct"] == 100.0
    assert cls_m["by_frequency_tier"]["warm_ge_15"]["accuracy_pct"] == 50.0
    assert cls_m["by_name_seen"]["seen_name"]["accuracy_pct"] == 100.0

    # Multi-label GK with 1 null-GT row excluded cleanly
    gk_m = M.compute_gk_metrics(
        ["Burger, Chicken Burger", "", "Fried Rice"],
        ["Burger", "Extra Tag", "Fried Rice, Rice"],
        train_seen_gks={"burger", "fried rice"},
    )
    assert gk_m["eval_rows"] == 2
    assert gk_m["null_gt_rows"] == 1
    assert gk_m["null_gt_predicted_empty_pct"] == 0.0
    assert gk_m["sample_precision_pct"] == 75.0
    assert gk_m["sample_recall_pct"] == 75.0

    # Reranker lift check: post-rerank moves relevant item from rank 2 to rank 1
    r_diag = M.compute_reranker_diagnostics(
        pre_binary_rels=[[0, 1, 0], [1, 0, 0]],
        post_binary_rels=[[1, 0, 0], [1, 0, 0]],
        total_relevant_per_query=[1, 1],
        pair_labels=[0, 1, 0, 1, 0, 0],
        pair_scores=[0.2, 0.9, 0.1, 0.85, 0.3, 0.15],
        top1_scores=[0.9, 0.85],
        top1_correct=[1, 1],
    )
    assert r_diag["before_rerank"]["hit@1_pct"] == 50.0
    assert r_diag["after_rerank"]["hit@1_pct"] == 100.0
    assert r_diag["lift"]["hit@1_pct"] == 50.0
    assert r_diag["win_tie_loss"] == {"wins": 1, "ties": 1, "losses": 0}
    assert r_diag["pair_scores"]["pairwise_auroc"] == 1.0


class _FakeEvalEngine:
    """Deterministic lightweight replacement for EvalEmbeddingEngine in unit tests."""

    def __init__(
        self,
        embedding_model=None,
        reranker_model=None,
        emb_precision="int8",
        rerank_precision="int8",
        **_kwargs,
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.emb_precision = emb_precision
        self.rerank_precision = rerank_precision
        self.embedding_dim = 8
        self.has_learned_sparse = False
        self.emb_backend = f"fake_{emb_precision}"
        self.rerank_backend = f"fake_{rerank_precision}"
        self.cross_session = object() if reranker_model else None
        if embedding_model:
            config.EMBEDDING_DIM = self.embedding_dim

    @staticmethod
    def _vec_for(text: str) -> np.ndarray:
        tl = str(text or "").lower()
        v = np.zeros(8, dtype=np.float32)
        if "burger" in tl:
            v[0] = 1.0
        elif "pizza" in tl:
            v[1] = 1.0
        elif "rice" in tl:
            v[2] = 1.0
        else:
            v[3] = 1.0
        v[4] = (sum(ord(c) for c in tl) % 17) * 0.01
        return v / np.linalg.norm(v)

    @staticmethod
    def compute_bm25_tf(text: str) -> dict[int, float]:
        return {abs(hash(tok)) % 100000 + 1: 1.0 for tok in str(text or "").lower().split() if tok}

    def encode(self, texts: list[str], batch_size: int = 32) -> dict:
        if not texts:
            return {"dense": np.array([]), "bm25": [], "sparse": []}
        dense = np.vstack([self._vec_for(t) for t in texts]).astype(np.float32)
        bm25 = [self.compute_bm25_tf(t) for t in texts]
        return {"dense": dense, "bm25": bm25, "sparse": []}

    def embed_weighted_sku(
        self,
        names: list[str],
        descriptions: list[str],
        categories: list[str],
        weights=(1.0, 0.3, 0.2),
    ) -> dict:
        texts = [f"{n} {d} {c}".strip() for n, d, c in zip(names, descriptions, categories)]
        return self.encode(texts)

    def embed_dictionary_incremental(self, domain: str, dict_key: str, keywords: list[str]) -> dict:
        return self.encode(keywords)

    def score_cross_encoder(self, pairs: list[list[str]], progress_callback=None) -> np.ndarray:
        scores = []
        for q, d in pairs:
            ql, dl = q.lower(), d.lower()
            overlap = len(set(ql.split()) & set(dl.split()))
            scores.append(min(0.99, 0.2 + 0.25 * overlap))
        return np.array(scores, dtype=np.float32)

    @staticmethod
    def predict_centroids_temperature(vecs: np.ndarray, prototypes: np.ndarray, tau: float = 0.05):
        from engine.inference.centroid_model import compute_temperature_centroids

        return compute_temperature_centroids(vecs, prototypes, tau=tau)


class _FakeNERModel:
    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.25):
        return []


def test_run_eval_end_to_end_smoke_and_checkpoint_resume(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval_data"
    staged_dir = eval_dir / "fixtures" / "staged_sheets"
    staged_dir.mkdir(parents=True)
    work_dir = tmp_path / "runs"

    train_rows = []
    for i in range(6):
        train_rows.append(
            {
                "Name": f"Spicy Chicken Burger {i}",
                "Description": "Grilled chicken burger",
                "Flavor": "Chicken",
                "Price": 1200 + i * 10,
                "SellerCategory": "BURGERS",
                "GenericKeywords": "Burger, Chicken Burger",
                "BasicType": "Burger",
                "Region": "Western",
            }
        )
        train_rows.append(
            {
                "Name": f"Classic Cheese Pizza {i}",
                "Description": "Wood fired pizza",
                "Flavor": "",
                "Price": 2100 + i * 10,
                "SellerCategory": "PIZZAS",
                "GenericKeywords": "Pizza, Cheese Pizza",
                "BasicType": "Pizza",
                "Region": "Italian",
            }
        )
    test_rows = [
        {
            "Name": "Spicy Chicken Burger 0",  # seen name
            "Description": "Grilled chicken burger",
            "Price": 1200,
            "SellerCategory": "BURGERS",
            "GenericKeywords": "Burger, Chicken Burger",
            "BasicType": "Burger",
            "Region": "Western",
        },
        {
            "Name": "Crispy Margherita Pizza",  # unseen name
            "Description": "Mozzarella pizza",
            "Price": 2250,
            "SellerCategory": "PIZZAS",
            "GenericKeywords": "Pizza, Cheese Pizza",
            "BasicType": "Pizza",
            "Region": "Italian",
        },
    ]

    pd.DataFrame(train_rows).to_csv(eval_dir / "train_food.csv", index=False)
    pd.DataFrame(test_rows).to_csv(eval_dir / "test_food.csv", index=False)

    pd.DataFrame(
        [
            {
                "Flavor Name": "Chicken",
                "Aliases": "chicken",
                "Is_Meat": True,
                "Is_Vegetable": False,
                "Is_Seafood": False,
            }
        ]
    ).to_csv(staged_dir / "Food_Flavors.csv", index=False)
    (staged_dir / "Food_BT.csv").write_text("Burger\nPizza\n", encoding="utf-8")
    (staged_dir / "Food_GK.csv").write_text(
        "Burger\nChicken Burger\nPizza\nCheese Pizza\n", encoding="utf-8"
    )
    (staged_dir / "Food_Region.csv").write_text("Western\nItalian\n", encoding="utf-8")

    rules_path = eval_dir / "fixtures" / "rules_snapshot.json"
    rules_path.write_text(json.dumps({"version": "v1", "rules": []}), encoding="utf-8")

    monkeypatch.setattr(run_eval, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(run_eval, "STAGED_FIXTURES_DIR", staged_dir)
    monkeypatch.setattr(run_eval, "RULES_SNAPSHOT_PATH", rules_path)
    monkeypatch.setattr(run_eval, "EvalEmbeddingEngine", _FakeEvalEngine)
    monkeypatch.setattr(config, "GLINER_ONNX", tmp_path / "gliner.onnx")
    (tmp_path / "gliner.onnx").write_bytes(b"fake")
    monkeypatch.setattr(
        "engine.pipelines.runtime_pipeline._ner_engine", SimpleNamespace(model=_FakeNERModel())
    )

    orig_cache = config.CACHE_DIR
    orig_arc = config.ARCFACE_DIR
    orig_food_col = config.QDRANT_COLLECTION_FOOD
    orig_mkt_col = config.QDRANT_COLLECTION_MARKET
    orig_prefix = config.QDRANT_TAGS_PREFIX
    orig_emb_dim = config.EMBEDDING_DIM
    orig_emb_model = config.EMBEDDING_MODEL
    orig_rr_model = config.RERANKER_MODEL
    orig_load_dicts = run_eval.DataIngestion.load_classifier_dictionaries
    orig_get_clf = run_eval.resource_loader.get_classifier
    orig_get_pipe = run_eval.resource_loader.get_pipeline
    orig_get_ner = run_eval.resource_loader.get_ner_engine
    try:
        args = run_eval.build_parser().parse_args(
            [
                "--domains",
                "food",
                "--embeddings",
                "BAAI/bge-m3",
                "--rerankers",
                "BAAI/bge-reranker-v2-m3",
                "--precisions",
                "int8",
                "--modes",
                "matcher,classifier,pipeline",
                "--heads",
                "arcface,logreg",
                "--work-dir",
                str(work_dir),
                "--epochs",
                "2",
                "--arcface-batch-size",
                "4",
            ]
        )
        run_eval.run_evaluation(args)

        reg = json.loads((work_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))
        combo_id = "food__bge-m3__bge-reranker-v2-m3__int8"
        assert reg["stage_c"][combo_id]["done"] is True

        summary = json.loads(
            (work_dir / "results" / combo_id / "summary.json").read_text(encoding="utf-8")
        )
        assert "pipeline__arcface__templates_on__rules_on" in summary["runs"]
        assert "pipeline__logreg__templates_off__rules_off" in summary["runs"]
        assert (
            summary["embedding_standalone"]["catalog_retrieval"]["hybrid_rrf"]["hit@1_pct"] == 100.0
        )
        assert (work_dir / "reports" / "leaderboard.csv").exists()
        assert (work_dir / "reports" / "report.md").exists()

        # Verify resume skips completed combo cleanly without running stage C again
        with patch.object(
            _FakeEvalEngine, "score_cross_encoder", side_effect=AssertionError("Should skip!")
        ):
            run_eval.run_evaluation(args)

        # Verify report aggregation standalone
        rep_paths = report.generate_reports(work_dir)
        lb = pd.read_csv(rep_paths["leaderboard.csv"])
        assert len(lb) >= 2
    finally:
        config.CACHE_DIR = orig_cache
        config.ARCFACE_DIR = orig_arc
        config.QDRANT_COLLECTION_FOOD = orig_food_col
        config.QDRANT_COLLECTION_MARKET = orig_mkt_col
        config.QDRANT_TAGS_PREFIX = orig_prefix
        config.EMBEDDING_DIM = orig_emb_dim
        config.EMBEDDING_MODEL = orig_emb_model
        config.RERANKER_MODEL = orig_rr_model
        run_eval.DataIngestion.load_classifier_dictionaries = orig_load_dicts
        run_eval.DataIngestion.clear_mem_cache()
        run_eval.resource_loader.get_classifier = orig_get_clf
        run_eval.resource_loader.get_pipeline = orig_get_pipe
        run_eval.resource_loader.get_ner_engine = orig_get_ner
        run_eval.resource_loader.reset_statuses(clear_caches=True)
        run_eval.resource_loader._embed_engine = None
        run_eval.resource_loader._ner_engine = None
        run_eval.resource_loader._vector_store = None
        if run_eval.QdrantClientManager._client is not None:
            run_eval.QdrantClientManager._client.close()
            run_eval.QdrantClientManager._client = None
        run_eval.QdrantClientManager._checked_collections.clear()


def test_eval_engine_length_bucketing_and_oom_backoff(monkeypatch):
    eng = model_manager.EvalEmbeddingEngine(
        embedding_model=None, reranker_model=None, rerank_precision="fp16", batch_size=4
    )
    eng.rerank_backend = "onnx_cuda_fp16"

    pairs = [
        ["q_long_" + ("x" * 100), "d_long_" + ("y" * 100)],  # idx 0 (long -> OOM at bs > 1)
        ["q1", "d1"],  # idx 1 (short)
        ["q_med_" + ("a" * 30), "d_med_" + ("b" * 30)],  # idx 2 (medium -> OOM at bs > 2)
        ["q2", "d2"],  # idx 3 (short)
        ["q_long2_" + ("z" * 120), "d_long2_" + ("w" * 120)],  # idx 4 (longest -> OOM at bs > 1)
    ]
    seen_calls: list[list[str]] = []

    def _fake_super_rerank(self, batch, batch_size=32, progress_callback=None):
        seen_calls.append([p[0] for p in batch])
        max_len = max(len(p[0]) + len(p[1]) for p in batch)
        # Simulate BFC Arena OOM when long sequences are batched together.
        if len(batch) > 2 and max_len > 60:
            raise RuntimeError(
                "onnxruntime_pybind11_state.Fail: bfc_arena.cc:360 Failed to allocate memory for requested buffer of size 1092874752"
            )
        if len(batch) > 1 and max_len > 200:
            raise RuntimeError("CUDA out of memory.")
        return np.array([float(len(p[0]) + len(p[1])) for p in batch], dtype=np.float32)

    monkeypatch.setattr(RerankerModel, "score_cross_encoder", _fake_super_rerank)

    progress_events: list[tuple[int, int]] = []
    scores = eng.score_cross_encoder(
        pairs, progress_callback=lambda d, t: progress_events.append((d, t))
    )
    expected = np.array([float(len(p[0]) + len(p[1])) for p in pairs], dtype=np.float32)
    np.testing.assert_allclose(scores, expected)
    # First batch (batch_size=4) had indices [1, 3, 2, 0] sorted by length, which hit OOM -> halved to 2 -> [1, 3] succeeded,
    # [2, 0] hit OOM (>200 chars with len>1) -> halved to 1 -> [2], [0], [4] each succeeded individually!
    assert seen_calls[0] == ["q1", "q2", pairs[2][0], pairs[0][0]]
    assert seen_calls[1] == ["q1", "q2"]
    assert progress_events[0] == (0, 5)
    assert progress_events[-1] == (5, 5)

    # INT8 must preserve exact original pair order (no length sorting) for dynamic activation quantization.
    eng_int8 = model_manager.EvalEmbeddingEngine(
        embedding_model=None, reranker_model=None, rerank_precision="int8", batch_size=3
    )
    eng_int8.rerank_backend = "onnx_cpu_int8"
    int8_calls: list[list[str]] = []

    def _fake_int8_rerank(self, batch, batch_size=32, progress_callback=None):
        int8_calls.append([p[0] for p in batch])
        return np.array([1.0] * len(batch), dtype=np.float32)

    monkeypatch.setattr(RerankerModel, "score_cross_encoder", _fake_int8_rerank)
    eng_int8.score_cross_encoder(pairs[:4])
    assert int8_calls == [[pairs[0][0], pairs[1][0], pairs[2][0]], [pairs[3][0]]]

    # CUDA ORT sessions must use kSameAsRequested arena extension to avoid power-of-two VRAM inflation.
    monkeypatch.setattr(
        model_manager.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    captured: dict[str, object] = {}

    class _DummySession:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _fake_ort_sess(path, opts, providers=None):
        captured["providers"] = providers
        return _DummySession()

    monkeypatch.setattr(model_manager.ort, "InferenceSession", _fake_ort_sess)
    _, tag = model_manager._create_ort_session(
        model_manager.Path("dummy.onnx"), "fp16", device="cuda"
    )
    assert tag == "onnx_cuda_fp16"
    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        "CPUExecutionProvider",
    ]
