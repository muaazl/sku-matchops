"""Scikit-learn LogisticRegression model wrapper for BT, Third-Tag, and GK predictions."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from tqdm import tqdm

from engine.core import config

logger = logging.getLogger("matchops.inference.logreg")


@contextlib.contextmanager
def track_joblib_progress(total: int, desc: str):
    """Attaches a tqdm progress bar to joblib parallel classifier fitting."""
    pbar = tqdm(total=total, desc=desc, unit="clf", mininterval=1.0, leave=True)

    class _TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCallback
    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        pbar.close()


class LogregClassifierModel:
    """Manages sklearn LogisticRegression heads, price scaling, and disk serialization."""

    def __init__(self) -> None:
        self.trained = False
        self.price_scaler = None
        self.bt_enc = None
        self.bt_clf = None
        self.third_tag_enc = None
        self.third_tag_clf = None
        self.gk_enc = None
        self.gk_clf = None

    def preprocess_prices(self, prices_list: list[float], is_training: bool = False) -> np.ndarray:
        """Log-scales prices and fits/applies `StandardScaler` with zero-price neutralization."""
        prices = np.array(prices_list, dtype=np.float32).reshape(-1, 1)
        prices = np.clip(prices, 0.0, None)
        log_prices = np.log1p(prices)

        if is_training:
            from sklearn.preprocessing import StandardScaler

            self.price_scaler = StandardScaler()
            return self.price_scaler.fit_transform(log_prices)

        if self.price_scaler is not None:
            scaled_prices = self.price_scaler.transform(log_prices)
            zero_mask = (prices <= 0.0).flatten()
            scaled_prices[zero_mask] = 0.0
            return scaled_prices
        return np.zeros_like(log_prices)

    def load_cached_models(self, cache_file: Path, expected_embedding_model: str) -> dict[str, Any]:
        """Reads serialized sklearn models from disk if compatible with the active embedding model."""
        if not cache_file.exists():
            return {}
        try:
            cached = joblib.load(cache_file)
            if not isinstance(cached, dict):
                return {}
            cached_emb = cached.get("embedding_model")
            has_model_marker = (cache_file.parent / "qdrant_embedding_models.json").exists() or (
                cache_file.parent / "active_embedding_model.txt"
            ).exists()
            emb_matches = cached_emb == expected_embedding_model or (
                cached_emb is None and not has_model_marker
            )
            if emb_matches:
                return cached
            logger.info(
                "[TRAIN] Cached sklearn classifier built with '%s' (!= '%s'); ignoring.",
                cached_emb,
                expected_embedding_model,
            )
        except Exception as exc:
            logger.warning("[TRAIN] Failed loading sklearn cache %s: %s", cache_file, exc)
        return {}

    def apply_cached_state(self, cached: dict[str, Any]) -> None:
        """Restores encoder and classifier weights from a loaded cache dictionary."""
        self.bt_enc = cached["bt_enc"]
        self.bt_clf = cached["bt_clf"]
        self.third_tag_enc = cached["third_tag_enc"]
        self.third_tag_clf = cached["third_tag_clf"]
        self.gk_enc = cached["gk_enc"]
        self.gk_clf = cached["gk_clf"]
        self.price_scaler = cached.get("price_scaler")
        self.trained = True

    def fit_classifiers(
        self,
        X: np.ndarray,
        y_bt: np.ndarray,
        y_third_tag: np.ndarray,
        y_gk: np.ndarray,
        domain: str,
        target_col: str,
        old_models: dict[str, Any] | None = None,
    ) -> None:
        """Fits BT, Third-Tag, and GK LogisticRegression / OneVsRestClassifier heads."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        old_models = old_models or {}
        old_bt_clf = old_models.get("bt_clf")
        old_bt_enc = old_models.get("bt_enc")
        old_third_clf = old_models.get("third_tag_clf")
        old_third_enc = old_models.get("third_tag_enc")

        logger.info(
            "[TRAIN] [%s] Fitting BT LogisticRegression (%d classes)...",
            domain.upper(),
            len(self.bt_enc.classes_),
        )
        can_warm_start_bt = (
            old_bt_clf is not None
            and old_bt_enc is not None
            and hasattr(old_bt_clf, "coef_")
            and list(old_bt_enc.classes_) == list(self.bt_enc.classes_)
        )
        if can_warm_start_bt:
            self.bt_clf = old_bt_clf
            self.bt_clf.warm_start = True
            self.bt_clf.fit(X, y_bt)
        else:
            self.bt_clf = LogisticRegression(max_iter=1000, C=5.0).fit(X, y_bt)

        logger.info(
            "[TRAIN] [%s] Fitting %s LogisticRegression (%d classes)...",
            domain.upper(),
            target_col.capitalize(),
            len(self.third_tag_enc.classes_),
        )
        can_warm_start_third = (
            old_third_clf is not None
            and old_third_enc is not None
            and hasattr(old_third_clf, "coef_")
            and list(old_third_enc.classes_) == list(self.third_tag_enc.classes_)
        )
        if can_warm_start_third:
            self.third_tag_clf = old_third_clf
            self.third_tag_clf.warm_start = True
            self.third_tag_clf.fit(X, y_third_tag)
        else:
            self.third_tag_clf = LogisticRegression(
                max_iter=1000, C=5.0, class_weight="balanced"
            ).fit(X, y_third_tag)

        n_gk_tags = len(self.gk_enc.classes_)
        logger.info(
            "[TRAIN] [%s] Fitting GK OneVsRestClassifier (%d tags, n_jobs=%d)...",
            domain.upper(),
            n_gk_tags,
            config.MAX_CPU_CORES,
        )
        base_clf = LogisticRegression(max_iter=250, C=5.0, class_weight="balanced")
        with (
            joblib.parallel_backend("threading", n_jobs=config.MAX_CPU_CORES),
            track_joblib_progress(n_gk_tags, desc=f"[TRAIN] {domain} GK OneVsRest"),
        ):
            self.gk_clf = OneVsRestClassifier(base_clf, n_jobs=config.MAX_CPU_CORES).fit(X, y_gk)

        self.trained = True

    def save_cached_models(self, cache_file: Path, embedding_model: str) -> None:
        """Persists sklearn encoders, classifiers, and scaler to disk."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "embedding_model": embedding_model,
                "bt_enc": self.bt_enc,
                "bt_clf": self.bt_clf,
                "third_tag_enc": self.third_tag_enc,
                "third_tag_clf": self.third_tag_clf,
                "gk_enc": self.gk_enc,
                "gk_clf": self.gk_clf,
                "price_scaler": self.price_scaler,
            },
            cache_file,
        )

    def predict_bt_batch(
        self, vecs: np.ndarray, prices: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns `(best_indices, confidences)` from the fitted BT LogisticRegression model."""
        scaled_p = self.preprocess_prices(prices, is_training=False)
        vecs_with_price = np.hstack([vecs, scaled_p])
        probas = self.bt_clf.predict_proba(vecs_with_price)
        return np.argmax(probas, axis=1), np.max(probas, axis=1)

    def predict_gk_batch(self, vecs: np.ndarray, prices: list[float]) -> np.ndarray:
        """Returns multi-label GK probabilities of shape `(N, num_gks)`."""
        scaled_p = self.preprocess_prices(prices, is_training=False)
        vecs_with_price = np.hstack([vecs, scaled_p])
        return self.gk_clf.predict_proba(vecs_with_price)

    def predict_third_tag_batch(
        self, vecs: np.ndarray, prices: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns `(best_indices, confidences)` from the fitted third-tag LogisticRegression."""
        scaled_p = self.preprocess_prices(prices, is_training=False)
        vecs_with_price = np.hstack([vecs, scaled_p])
        probas = self.third_tag_clf.predict_proba(vecs_with_price)
        return np.argmax(probas, axis=1), np.max(probas, axis=1)
