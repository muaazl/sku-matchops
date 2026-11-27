"""Temperature-scaled centroid prototype scoring model."""

from __future__ import annotations

import numpy as np

from engine.core import config


def compute_temperature_centroids(
    vecs: np.ndarray,
    prototypes: np.ndarray,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates temperature-scaled softmax class probabilities over normalized prototypes."""
    vecs_2d = np.atleast_2d(vecs)
    protos_2d = np.atleast_2d(prototypes)

    if vecs_2d.shape[0] == 0 or protos_2d.shape[0] == 0:
        return (
            np.empty((vecs_2d.shape[0], protos_2d.shape[0]), dtype=np.float32),
            np.empty(vecs_2d.shape[0], dtype=np.int64),
            np.empty(vecs_2d.shape[0], dtype=np.float32),
        )

    safe_tau = max(float(tau), 1e-6)
    logits = (vecs_2d @ protos_2d.T) / safe_tau

    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    sum_exp = np.where(sum_exp == 0, 1e-12, sum_exp)
    probas = (exp_logits / sum_exp).astype(np.float32)

    best_indices = np.argmax(probas, axis=-1)
    confidences = np.max(probas, axis=-1)
    return probas, best_indices, confidences


class CentroidPrototypeModel:
    """Predicts warm taxonomy classes using temperature-scaled cosine similarity prototypes."""

    def __init__(self, tau: float = config.COLD_START_TAU) -> None:
        self.tau = tau

    def predict_single(
        self,
        dense_vec: np.ndarray,
        prototypes: np.ndarray,
        classes: list[str],
    ) -> tuple[str, float, str]:
        """Scores one dense SKU vector against centroid prototypes."""
        if prototypes.shape[0] == 0:
            return "", 0.0, "trained"
        _, best_indices, confs = compute_temperature_centroids(dense_vec, prototypes, tau=self.tau)
        best_idx = int(best_indices[0])
        return classes[best_idx], float(confs[0]), "trained"

    def predict_batch(
        self,
        dense_vecs: np.ndarray,
        prototypes: np.ndarray,
        classes: list[str],
    ) -> list[tuple[str, float, str]]:
        """Batch-scores dense SKU vectors against centroid prototypes."""
        n = dense_vecs.shape[0]
        if n == 0:
            return []
        if prototypes.shape[0] == 0:
            return [("", 0.0, "trained")] * n

        _, best_indices, confs = compute_temperature_centroids(dense_vecs, prototypes, tau=self.tau)
        return [(classes[int(best_indices[i])], float(confs[i]), "trained") for i in range(n)]
