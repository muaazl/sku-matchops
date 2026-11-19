import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import joblib
import pytest

from engine.core import startup_checks


def test_check_active_embedding_model_missing(tmp_path: Path):
    with patch.object(startup_checks.config, "CACHE_DIR", tmp_path):
        errs = startup_checks.check_active_embedding_model()
    assert len(errs) == 1
    assert "scripts.sync" in errs[0]


def test_check_active_embedding_model_mismatch(tmp_path: Path):
    (tmp_path / "active_embedding_model.txt").write_text("jinaai/jina-embeddings-v5-text-small", encoding="utf-8")
    with (
        patch.object(startup_checks.config, "CACHE_DIR", tmp_path),
        patch.object(startup_checks.config, "EMBEDDING_MODEL", "BAAI/bge-m3"),
    ):
        errs = startup_checks.check_active_embedding_model()
    assert len(errs) == 1
    assert "Embedding model mismatch" in errs[0]
    assert "jinaai/jina-embeddings-v5-text-small" in errs[0]
    assert "BAAI/bge-m3" in errs[0]


def test_check_active_embedding_model_match(tmp_path: Path):
    (tmp_path / "active_embedding_model.txt").write_text("BAAI/bge-m3", encoding="utf-8")
    with (
        patch.object(startup_checks.config, "CACHE_DIR", tmp_path),
        patch.object(startup_checks.config, "EMBEDDING_MODEL", "BAAI/bge-m3"),
    ):
        assert startup_checks.check_active_embedding_model() == []


def test_check_qdrant_detects_same_dim_model_mismatch_and_empty_collection(tmp_path: Path):
    (tmp_path / "qdrant_embedding_models.json").write_text(
        json.dumps({"market": "jinaai/jina-embeddings-v5-text-small", "food": "BAAI/bge-m3"}),
        encoding="utf-8",
    )

    fake_vs = MagicMock()
    fake_vs.client.get_collections.return_value = []
    fake_vs.client.collection_exists.return_value = True
    fake_vs.client.get_collection.side_effect = [
        SimpleNamespace(points_count=100),  # market
        SimpleNamespace(points_count=0),    # food (empty)
    ]
    fake_vs.get_collection_schema_mismatch.return_value = None

    with (
        patch.object(startup_checks.config, "CACHE_DIR", tmp_path),
        patch.object(startup_checks.config, "EMBEDDING_MODEL", "BAAI/bge-m3"),
        patch("engine.data_pipeline.vector_store.VectorStore", return_value=fake_vs),
    ):
        errs = startup_checks.check_qdrant()

    assert any("market_catalog" in e and "jinaai/jina-embeddings-v5-text-small" in e for e in errs)
    assert any("food_catalog" in e and "empty" in e for e in errs)


def test_check_arcface_detects_embedding_model_mismatch(tmp_path: Path):
    arc_dir = tmp_path / "arcface"
    arc_dir.mkdir()
    (arc_dir / "food_bt_arcface_int8.onnx").write_bytes(b"fake-onnx")
    (arc_dir / "food_bt_arcface_meta.joblib").write_bytes(b"fake-meta")
    (arc_dir / "food_bt_arcface_labels.json").write_text(
        json.dumps(
            {
                "classes": ["Pizza", "Burger"],
                "input_dim": 1025,
                "embedding_model": "jinaai/jina-embeddings-v5-text-small",
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.object(startup_checks.config, "get_arcface_dir", return_value=arc_dir),
        patch.object(
            startup_checks.config,
            "get_bt_arcface_onnx_path",
            return_value=arc_dir / "food_bt_arcface_int8.onnx",
        ),
        patch.object(
            startup_checks.config,
            "get_bt_arcface_labels_path",
            return_value=arc_dir / "food_bt_arcface_labels.json",
        ),
        patch.object(startup_checks.config, "EMBEDDING_DIM", 1024),
        patch.object(startup_checks.config, "EMBEDDING_MODEL", "BAAI/bge-m3"),
    ):
        errs = startup_checks._check_arcface_domain("food")

    assert any("jinaai/jina-embeddings-v5-text-small" in e and "BAAI/bge-m3" in e for e in errs)


def test_check_logreg_detects_embedding_model_mismatch(tmp_path: Path):
    joblib.dump(
        {"embedding_model": "jinaai/jina-embeddings-v5-text-small", "bt_clf": "dummy"},
        tmp_path / "market_classifier_model.pkl",
    )
    with (
        patch.object(startup_checks.config, "CACHE_DIR", tmp_path),
        patch.object(startup_checks.config, "EMBEDDING_MODEL", "BAAI/bge-m3"),
    ):
        errs = startup_checks._check_logreg_domain("market")

    assert len(errs) == 1
    assert "jinaai/jina-embeddings-v5-text-small" in errs[0]


def test_check_redis_fails_when_unreachable():
    with patch("redis.Redis.from_url", side_effect=ConnectionError("refused")):
        errs = startup_checks.check_redis()
    assert len(errs) == 1
    assert "Redis is required" in errs[0]


def test_classifier_load_arcface_model_fails_on_embedding_model_mismatch(tmp_path: Path):
    from engine.classification.classifier import TaxonomyClassifier

    onnx_path = tmp_path / "food_bt_arcface_int8.onnx"
    labels_path = tmp_path / "food_bt_arcface_labels.json"
    onnx_path.write_bytes(b"fake")
    labels_path.write_text(
        json.dumps(
            {
                "classes": ["Pizza"],
                "input_dim": 1025,
                "embedding_model": "jinaai/jina-embeddings-v5-text-small",
            }
        ),
        encoding="utf-8",
    )

    clf = TaxonomyClassifier.__new__(TaxonomyClassifier)
    clf.domain = "food"

    with (
        patch("engine.classification.classifier.config.get_bt_arcface_onnx_path", return_value=onnx_path),
        patch("engine.classification.classifier.config.get_bt_arcface_labels_path", return_value=labels_path),
        patch("engine.classification.classifier.config.EMBEDDING_MODEL", "BAAI/bge-m3"),
        patch("engine.classification.classifier.config.EMBEDDING_DIM", 1024),
    ):
        with pytest.raises(RuntimeError, match="trained with 'jinaai/jina-embeddings-v5-text-small'"):
            clf._load_arcface_model()
