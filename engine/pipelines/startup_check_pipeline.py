"""
Pre-flight startup validation for the SKU MatchOps ML Engine.

Verifies that all required models, ONNX artifacts, Qdrant collections, Redis,
data caches, and classifier heads (ArcFace / LogReg) are present, healthy,
and synchronized with the active EMBEDDING_MODEL before the engine boots.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib

from engine.core import config

logger = logging.getLogger("matchops.engine.startup_checks")


class StartupCheckError(RuntimeError):
    """Raised when one or more pre-flight engine startup checks fail."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def check_config() -> list[str]:
    errors: list[str] = []
    if not (config.EMBEDDING_MODEL or "").strip():
        errors.append("EMBEDDING_MODEL is not configured.")
    if not (config.RERANKER_MODEL or "").strip():
        errors.append("RERANKER_MODEL is not configured.")
    for domain in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        bt_model = config.get_bt_model(domain)
        if bt_model not in ("arcface", "logreg"):
            errors.append(
                f"{domain.upper()}_BT_MODEL='{bt_model}' is invalid; must be 'arcface' or 'logreg'."
            )
    return errors


def check_onnx_models() -> list[str]:
    errors: list[str] = []
    required_files = [
        ("Embedding INT8 ONNX", Path(config.EMBEDDING_ONNX_INT8)),
        ("Embedding manifest", Path(config.EMBEDDING_MANIFEST)),
        ("Reranker INT8 ONNX", Path(config.RERANKER_ONNX_INT8)),
        ("Reranker manifest", Path(config.RERANKER_MANIFEST)),
        ("GLiNER ONNX", Path(config.GLINER_ONNX)),
    ]
    for label, path in required_files:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(
                f"{label} missing or empty at '{path}'. Run 'python -m scripts.sync --models sync' first."
            )

    emb_manifest_path = Path(config.EMBEDDING_MANIFEST)
    if emb_manifest_path.exists():
        try:
            manifest = _read_json(emb_manifest_path)
            dim = manifest.get("embedding_dim")
            if dim is not None and int(dim) != config.EMBEDDING_DIM:
                errors.append(
                    f"Embedding manifest dim ({dim}) != config.EMBEDDING_DIM ({config.EMBEDDING_DIM}). "
                    "Run 'python -m scripts.sync --models rebuild --model-target embedding'."
                )
        except Exception as e:
            errors.append(f"Corrupt embedding manifest '{emb_manifest_path}': {e}")

    rerank_manifest_path = Path(config.RERANKER_MANIFEST)
    if rerank_manifest_path.exists():
        try:
            _read_json(rerank_manifest_path)
        except Exception as e:
            errors.append(f"Corrupt reranker manifest '{rerank_manifest_path}': {e}")

    return errors


def check_active_embedding_model() -> list[str]:
    errors: list[str] = []
    marker = Path(config.CACHE_DIR) / "active_embedding_model.txt"
    if not marker.exists():
        errors.append(
            f"Missing '{marker}'. Sync has not been run yet. Run 'python -m scripts.sync' first."
        )
        return errors

    recorded = marker.read_text(encoding="utf-8").strip()
    if not recorded:
        errors.append(
            f"'{marker}' is empty. Run 'python -m scripts.sync --rebuild' to re-sync all artifacts."
        )
    elif recorded != config.EMBEDDING_MODEL:
        errors.append(
            f"Embedding model mismatch: caches/models were synced with '{recorded}', "
            f"but active EMBEDDING_MODEL is '{config.EMBEDDING_MODEL}'. "
            f"Either revert EMBEDDING_MODEL to '{recorded}' or run 'python -m scripts.sync --rebuild'."
        )
    return errors


def check_redis() -> list[str]:
    errors: list[str] = []
    try:
        import redis

        client = redis.Redis.from_url(
            config.REDIS_URL,
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=True,
        )
        if not client.ping():
            errors.append(f"Redis ping returned False at '{config.REDIS_URL}'.")
    except Exception as e:
        errors.append(f"Redis is required and unreachable at '{config.REDIS_URL}': {e}")
    return errors


def check_qdrant() -> list[str]:
    errors: list[str] = []
    try:
        from engine.vector_store.vector_store import VectorStore

        vs = VectorStore()
        vs.verify_connection()
    except Exception as e:
        return [f"Qdrant is unreachable at '{config.QDRANT_URL}': {e}"]

    qdrant_state_path = Path(config.CACHE_DIR) / "qdrant_embedding_models.json"
    qdrant_state: dict[str, Any] = {}
    if qdrant_state_path.exists():
        try:
            qdrant_state = _read_json(qdrant_state_path)
        except Exception as e:
            errors.append(f"Corrupt '{qdrant_state_path}': {e}")
    else:
        errors.append(
            f"Missing '{qdrant_state_path}'. Run 'python -m scripts.sync --qdrant rebuild' first."
        )

    collections = (
        (config.DOMAIN_MARKET, config.QDRANT_COLLECTION_MARKET),
        (config.DOMAIN_FOOD, config.QDRANT_COLLECTION_FOOD),
    )
    for domain, col_name in collections:
        try:
            if not vs.check_collection_exists(col_name):
                errors.append(
                    f"Qdrant collection '{col_name}' does not exist. "
                    f"Run 'python -m scripts.sync --qdrant rebuild --domain {domain}'."
                )
                continue

            points_count = vs.count_collection_points(col_name)
            if points_count <= 0:
                errors.append(
                    f"Qdrant collection '{col_name}' is empty (0 points). "
                    f"Run 'python -m scripts.sync --qdrant rebuild --domain {domain}'."
                )

            mismatch = vs.get_collection_schema_mismatch(col_name)
            if mismatch:
                errors.append(
                    f"Qdrant collection '{col_name}' schema mismatch ({mismatch}). "
                    f"Run 'python -m scripts.sync --qdrant rebuild --domain {domain}'."
                )

            recorded_model = str(qdrant_state.get(domain, "") or "")
            if recorded_model and recorded_model != config.EMBEDDING_MODEL:
                errors.append(
                    f"Qdrant collection '{col_name}' was indexed with '{recorded_model}', "
                    f"but active EMBEDDING_MODEL is '{config.EMBEDDING_MODEL}'. "
                    f"Revert EMBEDDING_MODEL to '{recorded_model}' or run "
                    f"'python -m scripts.sync --qdrant rebuild --domain {domain}'."
                )
            elif qdrant_state_path.exists() and not recorded_model:
                errors.append(
                    f"Qdrant collection '{col_name}' has no recorded embedding model in '{qdrant_state_path}'. "
                    f"Run 'python -m scripts.sync --qdrant rebuild --domain {domain}'."
                )
        except Exception as e:
            errors.append(f"Failed to inspect Qdrant collection '{col_name}': {e}")

    return errors


def _check_arcface_domain(domain: str) -> list[str]:
    errors: list[str] = []
    onnx_path = Path(config.get_bt_arcface_onnx_path(domain))
    labels_path = Path(config.get_bt_arcface_labels_path(domain))
    meta_path = Path(config.get_arcface_dir()) / f"{domain}_bt_arcface_meta.joblib"

    for label, p in (
        ("ArcFace ONNX", onnx_path),
        ("ArcFace labels JSON", labels_path),
        ("ArcFace meta joblib", meta_path),
    ):
        if not p.exists() or p.stat().st_size == 0:
            errors.append(
                f"[{domain.upper()}] {label} missing at '{p}'. "
                f"Run 'python -m scripts.sync --arcface rebuild --domain {domain}'."
            )

    if errors:
        return errors

    classes: list[Any] = []
    try:
        meta = _read_json(labels_path)
        classes = meta.get("classes") or []
        if not isinstance(classes, list) or not classes:
            errors.append(f"[{domain.upper()}] ArcFace labels '{labels_path}' has no classes.")

        expected_dim = config.EMBEDDING_DIM + 1
        actual_dim = meta.get("input_dim")
        if actual_dim != expected_dim:
            errors.append(
                f"[{domain.upper()}] ArcFace input_dim ({actual_dim}) != expected ({expected_dim}). "
                f"Run 'python -m scripts.sync --arcface rebuild --domain {domain}'."
            )

        trained_emb = meta.get("embedding_model")
        if not trained_emb:
            errors.append(
                f"[{domain.upper()}] ArcFace labels '{labels_path}' is missing 'embedding_model'. "
                f"Run 'python -m scripts.sync --arcface rebuild --domain {domain}'."
            )
        elif trained_emb != config.EMBEDDING_MODEL:
            errors.append(
                f"[{domain.upper()}] ArcFace was trained with '{trained_emb}', "
                f"but active EMBEDDING_MODEL is '{config.EMBEDDING_MODEL}'. "
                f"Revert EMBEDDING_MODEL to '{trained_emb}' or run "
                f"'python -m scripts.sync --arcface rebuild --domain {domain}'."
            )
    except Exception as e:
        errors.append(f"[{domain.upper()}] Failed to parse '{labels_path}': {e}")
        return errors

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_shape = sess.get_inputs()[0].shape
        out_shape = sess.get_outputs()[0].shape
        if len(in_shape) >= 2 and isinstance(in_shape[1], int):
            if in_shape[1] != config.EMBEDDING_DIM + 1:
                errors.append(
                    f"[{domain.upper()}] ArcFace ONNX input dim ({in_shape[1]}) != "
                    f"EMBEDDING_DIM + 1 ({config.EMBEDDING_DIM + 1})."
                )
        if len(out_shape) >= 2 and isinstance(out_shape[1], int) and classes:
            if out_shape[1] != len(classes):
                errors.append(
                    f"[{domain.upper()}] ArcFace ONNX output classes ({out_shape[1]}) != "
                    f"labels count ({len(classes)})."
                )
    except Exception as e:
        errors.append(f"[{domain.upper()}] Failed to load ArcFace ONNX '{onnx_path}': {e}")

    return errors


def _check_logreg_domain(domain: str) -> list[str]:
    errors: list[str] = []
    clf_path = Path(config.CACHE_DIR) / f"{domain}_classifier_model.pkl"
    if not clf_path.exists() or clf_path.stat().st_size == 0:
        return [
            f"[{domain.upper()}] LogReg classifier cache missing at '{clf_path}'. "
            f"Run 'python -m scripts.sync --cache rebuild --domain {domain}'."
        ]
    try:
        cached = joblib.load(clf_path)
        trained_emb = cached.get("embedding_model") if isinstance(cached, dict) else None
        if trained_emb != config.EMBEDDING_MODEL:
            errors.append(
                f"[{domain.upper()}] LogReg classifier was trained with '{trained_emb}', "
                f"but active EMBEDDING_MODEL is '{config.EMBEDDING_MODEL}'. "
                f"Revert EMBEDDING_MODEL to '{trained_emb}' or run "
                f"'python -m scripts.sync --cache rebuild --domain {domain}'."
            )
        if not isinstance(cached, dict) or cached.get("bt_clf") is None:
            errors.append(f"[{domain.upper()}] Corrupt LogReg classifier cache '{clf_path}'.")
    except Exception as e:
        errors.append(f"[{domain.upper()}] Failed to load LogReg cache '{clf_path}': {e}")
    return errors


def check_domain_caches_and_classifiers() -> list[str]:
    errors: list[str] = []
    cache_dir = Path(config.CACHE_DIR)

    for domain in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        required_caches = (
            cache_dir / f"{domain}_catalog_mmap.feather",
            cache_dir / f"{domain}_brands_mmap.feather",
            cache_dir / f"{domain}_classifier_dicts.json",
            cache_dir / f"{domain}_warm_centroids.pkl",
        )
        for p in required_caches:
            if not p.exists() or p.stat().st_size == 0:
                errors.append(
                    f"[{domain.upper()}] Required cache '{p.name}' is missing or empty. "
                    f"Run 'python -m scripts.sync --cache rebuild --domain {domain}'."
                )

        centroids_path = cache_dir / f"{domain}_warm_centroids.pkl"
        if centroids_path.exists() and centroids_path.stat().st_size > 0:
            try:
                c_data = joblib.load(centroids_path)
                c_emb = c_data.get("embedding_model") if isinstance(c_data, dict) else None
                if c_emb and c_emb != config.EMBEDDING_MODEL:
                    errors.append(
                        f"[{domain.upper()}] Warm centroids cached with '{c_emb}', "
                        f"but active EMBEDDING_MODEL is '{config.EMBEDDING_MODEL}'. "
                        f"Run 'python -m scripts.sync --cache rebuild --domain {domain}'."
                    )
            except Exception as e:
                errors.append(f"[{domain.upper()}] Corrupt warm centroids '{centroids_path}': {e}")

        weighted_path = cache_dir / f"{domain}_weighted_skus_cache.pkl"
        if weighted_path.exists() and weighted_path.stat().st_size > 0:
            try:
                w_data = joblib.load(weighted_path)
                w_emb = w_data.get("__embedding_model__") if isinstance(w_data, dict) else None
                if w_emb and w_emb != config.EMBEDDING_MODEL:
                    errors.append(
                        f"[{domain.upper()}] Weighted SKU cache built with '{w_emb}', "
                        f"but active EMBEDDING_MODEL is '{config.EMBEDDING_MODEL}'. "
                        f"Run 'python -m scripts.sync --cache rebuild --domain {domain}'."
                    )
            except Exception as e:
                errors.append(
                    f"[{domain.upper()}] Corrupt weighted SKU cache '{weighted_path}': {e}"
                )

        if config.get_bt_model(domain) == "arcface":
            errors.extend(_check_arcface_domain(domain))
        else:
            errors.extend(_check_logreg_domain(domain))

    return errors


def run_startup_checks() -> None:
    """Runs all pre-flight engine startup checks and raises StartupCheckError on any failure."""
    checks = (
        ("Configuration", check_config),
        ("ONNX models", check_onnx_models),
        ("Active embedding model", check_active_embedding_model),
        ("Redis", check_redis),
        ("Qdrant collections", check_qdrant),
        ("Domain caches & classifiers", check_domain_caches_and_classifiers),
    )
    all_errors: list[str] = []
    for name, fn in checks:
        errs = fn()
        if errs:
            print(f"[STARTUP CHECK][FAIL] {name}:", flush=True)
            for err in errs:
                print(f"  - {err}", flush=True)
            all_errors.extend(errs)
        else:
            print(f"[STARTUP CHECK][OK] {name}", flush=True)

    if all_errors:
        raise StartupCheckError(
            f"{len(all_errors)} startup check(s) failed. Fix the errors above before starting the engine."
        )


def main() -> int:
    print("=" * 60, flush=True)
    print("Running SKU MatchOps Engine Pre-Flight Startup Checks...", flush=True)
    print("=" * 60, flush=True)
    try:
        run_startup_checks()
    except StartupCheckError as e:
        print("=" * 60, flush=True)
        print(f"FATAL: {e}", flush=True)
        print("=" * 60, flush=True)
        return 1
    print("=" * 60, flush=True)
    print("All engine pre-flight startup checks passed!", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
