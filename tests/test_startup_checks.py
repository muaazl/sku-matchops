import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pytest

from engine.pipelines import startup_check_pipeline as startup_checks


def test_check_active_embedding_model_missing(tmp_path: Path):
    with patch.object(startup_checks.config, "CACHE_DIR", tmp_path):
        errs = startup_checks.check_active_embedding_model()
    assert len(errs) == 1
    assert "scripts.sync" in errs[0]


def test_check_active_embedding_model_mismatch(tmp_path: Path):
    (tmp_path / "active_embedding_model.txt").write_text(
        "jinaai/jina-embeddings-v5-text-small", encoding="utf-8"
    )
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
    fake_vs.verify_connection.return_value = []
    fake_vs.check_collection_exists.return_value = True
    fake_vs.count_collection_points.side_effect = [100, 0]
    fake_vs.get_collection_schema_mismatch.return_value = None

    with (
        patch.object(startup_checks.config, "CACHE_DIR", tmp_path),
        patch.object(startup_checks.config, "EMBEDDING_MODEL", "BAAI/bge-m3"),
        patch("engine.vector_store.vector_store.VectorStore", return_value=fake_vs),
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
    from engine.pipelines.classification_pipeline import TaxonomyClassifier

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
        patch(
            "engine.core.config.get_bt_arcface_onnx_path",
            return_value=onnx_path,
        ),
        patch(
            "engine.core.config.get_bt_arcface_labels_path",
            return_value=labels_path,
        ),
        patch("engine.core.config.EMBEDDING_MODEL", "BAAI/bge-m3"),
        patch("engine.core.config.EMBEDDING_DIM", 1024),
    ):
        with pytest.raises(
            RuntimeError, match="trained with 'jinaai/jina-embeddings-v5-text-small'"
        ):
            clf._load_arcface_model()


def test_check_domain_synced_reports_unsynced_qdrant_and_tags(tmp_path: Path):
    import pandas as pd

    (tmp_path / "active_embedding_model.txt").write_text("BAAI/bge-m3", encoding="utf-8")
    (tmp_path / "qdrant_embedding_models.json").write_text(
        json.dumps({"food": "BAAI/bge-m3"}), encoding="utf-8"
    )
    (tmp_path / "food_classifier_dicts.json").write_text(
        json.dumps({"gk": ["Seafood"], "bt": ["Burger"], "region": ["Asian"]}),
        encoding="utf-8",
    )
    pd.DataFrame([{"Name": "Fish Burger"}, {"Name": "Chicken Rice"}]).to_feather(
        tmp_path / "food_catalog_mmap.feather"
    )
    pd.DataFrame([{"Flavor Name": "Fish"}]).to_feather(tmp_path / "food_brands_mmap.feather")
    joblib.dump(
        {"embedding_model": "BAAI/bge-m3", "classes": ["Burger"]},
        tmp_path / "food_warm_centroids.pkl",
    )
    joblib.dump(
        pd.DataFrame(
            [
                {"clean_text": "fish burger", "entities": {}, "clean_no_weights": "fish burger"},
                {"clean_text": "chicken rice", "entities": {}, "clean_no_weights": "chicken rice"},
            ]
        ),
        tmp_path / "food_catalog_metadata.pkl",
    )

    fake_vs = MagicMock()
    fake_vs.resolve_catalog_collection_name.return_value = "food_catalog"
    fake_vs.check_collection_exists.return_value = True
    fake_vs.count_collection_points.return_value = 1  # fewer than 2 catalog rows
    fake_vs.get_collection_schema_mismatch.return_value = None
    fake_vs.check_tags_synced.side_effect = [True, False, True]

    with (
        patch.object(startup_checks.config, "CACHE_DIR", tmp_path),
        patch.object(startup_checks.config, "EMBEDDING_MODEL", "BAAI/bge-m3"),
        patch.object(startup_checks, "check_onnx_models", return_value=[]),
        patch.object(startup_checks, "_check_arcface_domain", return_value=[]),
        patch.object(startup_checks, "_check_logreg_domain", return_value=[]),
        patch("engine.vector_store.vector_store.VectorStore", return_value=fake_vs),
    ):
        errs = startup_checks.check_domain_synced("food")

    assert any("food_catalog" in e and "expected at least 2" in e for e in errs)
    assert any("Qdrant 'bt' tags are not synced" in e for e in errs)
