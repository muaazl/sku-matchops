"""ArcFace PyTorch architecture modules and ONNX inference wrapper."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from engine.core import config
from engine.core.observability import ONNX_INFERENCE_SECONDS
from engine.core.onnx_factory import create_onnx_session

logger = logging.getLogger("matchops.inference.arcface")


class ArcFaceHead(nn.Module):
    """ArcFace additive angular margin classification head."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.35,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Computes scaled cosine similarity logits, applying angular margin when training."""
        x_norm = F.normalize(x, p=2, dim=1, eps=self.eps)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=self.eps)
        cosine = F.linear(x_norm, w_norm)

        if labels is None or not self.training:
            return cosine * self.scale

        cosine_clamped = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)
        sine = torch.sqrt((1.0 - cosine_clamped * cosine_clamped).clamp(min=self.eps))
        phi = cosine_clamped * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_clamped > self.th, phi, cosine_clamped - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class BTArcFaceNet(nn.Module):
    """Deep metric learning network for BasicType classification over embedding + log-price."""

    def __init__(
        self,
        in_features: int = 1025,
        hidden_features: int = 512,
        num_classes: int = 899,
        scale: float = 30.0,
        margin: float = 0.35,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_classes = num_classes

        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.head = ArcFaceHead(
            in_features=hidden_features,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts projected 512-D latent representations."""
        return self.projector(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Projects fused SKU+price features and scores ArcFace logits."""
        latent = self.projector(x)
        return self.head(latent, labels=labels)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss emphasizing sparse/long-tail BasicType classes."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes focal cross-entropy loss from raw logits and integer class targets."""
        log_p = F.log_softmax(logits, dim=-1)
        target_log_p = log_p.gather(1, targets.unsqueeze(1).long()).squeeze(1)
        p_t = target_log_p.exp().clamp(min=self.eps, max=1.0)
        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = -focal_weight * target_log_p
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ArcFaceOnnxModel:
    """ONNX Runtime wrapper for BasicType classification with price scaling."""

    def __init__(self) -> None:
        self.session = None
        self.classes: list[str] = []
        self.price_scaler: StandardScaler | None = None
        self.input_name: str | None = None

    def load_arcface_artifacts(self, domain: str) -> None:
        """Loads domain ArcFace ONNX session, label mapping, and price scaler with validation."""
        onnx_path = Path(config.get_bt_arcface_onnx_path(domain))
        labels_path = Path(config.get_bt_arcface_labels_path(domain))

        if not onnx_path.exists() or not labels_path.exists():
            raise RuntimeError(
                f"[FAIL-FAST] Domain '{domain}' is configured with BT_MODEL='arcface', "
                f"but required ONNX model file '{onnx_path}' or label mapping '{labels_path}' is missing. "
                f"Please train and export the model using 'python -m scripts.sync --arcface rebuild --domain {domain}' "
                f"or configure {domain.upper()}_BT_MODEL=logreg in your environment."
            )

        try:
            with labels_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            trained_emb = meta.get("embedding_model")
            if trained_emb and trained_emb != config.EMBEDDING_MODEL:
                raise RuntimeError(
                    f"[FAIL-FAST] ArcFace model for '{domain}' was trained with "
                    f"'{trained_emb}', but active EMBEDDING_MODEL is '{config.EMBEDDING_MODEL}'. "
                    f"Revert EMBEDDING_MODEL to '{trained_emb}' or run "
                    f"'python -m scripts.sync --arcface rebuild --domain {domain}'."
                )
            expected_dim = config.EMBEDDING_DIM + 1
            actual_dim = meta.get("input_dim")
            if actual_dim is not None and actual_dim != expected_dim:
                raise RuntimeError(
                    f"[FAIL-FAST] ArcFace input_dim ({actual_dim}) != expected ({expected_dim}) "
                    f"for '{domain}'. Run 'python -m scripts.sync --arcface rebuild --domain {domain}'."
                )
            self.classes = meta["classes"]
            scaler_info = meta.get("price_scaler", {})
            scaler = StandardScaler()
            scaler.mean_ = np.array([scaler_info.get("mean", 0.0)], dtype=np.float32)
            scaler.scale_ = np.array([scaler_info.get("scale", 1.0)], dtype=np.float32)
            scaler.var_ = np.array([scaler_info.get("var", 1.0)], dtype=np.float32)
            self.price_scaler = scaler
        except Exception as exc:
            raise RuntimeError(
                f"[FAIL-FAST] Failed to parse ArcFace label metadata from '{labels_path}': {exc}"
            ) from exc

        try:
            self.session, _ = create_onnx_session(onnx_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(
                "[ARCFACE] [%s] Successfully loaded ArcFace ONNX model from %s with %d classes.",
                domain.upper(),
                onnx_path,
                len(self.classes),
            )
        except Exception as exc:
            raise RuntimeError(
                f"[FAIL-FAST] Failed to initialize ONNX InferenceSession from '{onnx_path}': {exc}"
            ) from exc

    def preprocess_prices(self, prices_list: list[float]) -> np.ndarray:
        """Applies clipped `log1p` normalization while neutralizing missing/zero prices to 0.0."""
        prices = np.array(prices_list, dtype=np.float32).reshape(-1, 1)
        prices = np.clip(prices, 0.0, None)
        log_prices = np.log1p(prices)
        if self.price_scaler is not None:
            scaled = self.price_scaler.transform(log_prices)
            zero_mask = (prices <= 0.0).flatten()
            scaled[zero_mask] = 0.0
            return scaled.astype(np.float32)
        return np.zeros_like(log_prices, dtype=np.float32)

    def predict_probas(
        self, vecs: np.ndarray, prices: list[float], batch_size: int = config.ONNX_BATCH_SIZE
    ) -> np.ndarray:
        """Runs batched ArcFace ONNX inference and converts logits to softmax probabilities."""
        if self.session is None:
            return np.empty((len(vecs), 0), dtype=np.float32)
        scaled_p = self.preprocess_prices(prices)
        vecs_with_price = np.hstack([vecs, scaled_p]).astype(np.float32)
        n = len(vecs_with_price)
        bs = max(1, int(batch_size))
        batches = []
        for i in range(0, n, bs):
            with ONNX_INFERENCE_SECONDS.labels(model="arcface").time():
                batches.append(
                    self.session.run(None, {self.input_name: vecs_with_price[i : i + bs]})[0]
                )
        logits = np.concatenate(batches, axis=0)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict_single(self, vec_2d: np.ndarray, price: float) -> tuple[str, float]:
        """Predicts `(class_name, confidence)` for a single 2D SKU vector and price."""
        probas = self.predict_probas(vec_2d, [price])[0]
        best = int(np.argmax(probas))
        return self.classes[best], float(probas[best])

    def predict_batch(self, vecs: np.ndarray, prices: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Returns `(best_indices, confidences)` across a batch of SKU vectors."""
        probas = self.predict_probas(vecs, prices)
        return np.argmax(probas, axis=1), np.max(probas, axis=1)
