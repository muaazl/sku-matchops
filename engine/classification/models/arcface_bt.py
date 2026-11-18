import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin) Metric Learning Classification Head.

    Applies additive angular margin penalty m to target classes during training:
        cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
    During evaluation / inference, outputs scaled angular similarity logits:
        logits = scale * (W^T x) / (||W|| * ||x||)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.35,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.eps = eps

        # Weight matrix: (num_classes, in_features)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for margin formula
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input feature tensor of shape (batch_size, in_features).
            labels: Optional ground-truth class label tensor of shape (batch_size,).
                    If provided (training), additive angular margin is applied.
                    If None (eval/inference/export), outputs raw scaled cosine similarities.
        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # L2-normalize input embeddings and class weight vectors
        x_norm = F.normalize(x, p=2, dim=1, eps=self.eps)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=self.eps)

        # Compute cosine similarity: cos(theta)
        cosine = F.linear(x_norm, w_norm)

        # If no labels or in evaluation mode, return scaled cosine logits
        if labels is None or not self.training:
            return cosine * self.scale

        # Numerical stability clamp for cosine
        cosine_clamped = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt((1.0 - cosine_clamped * cosine_clamped).clamp(min=self.eps))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine_clamped * self.cos_m - sine * self.sin_m

        # Keep monotonicity when theta + m > pi
        phi = torch.where(cosine_clamped > self.th, phi, cosine_clamped - self.mm)

        # One-hot mask for ground-truth labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply margin only to target classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return output


class BTArcFaceNet(nn.Module):
    """
    Deep Metric Learning PyTorch model for SKU BasicType (BT) classification.

    Input Dimension: EMBEDDING_DIM + 1
      - D-dimensional dense embedding (Name, Description, Category) from the configured EMBEDDING_MODEL
      - 1-D scaled log-price: StandardScaler(np.log1p(price))

    Architecture:
      - (EMBEDDING_DIM + 1) -> 512 dense projection
      - LayerNorm(512)
      - GELU activation
      - Dropout(0.2)
      - ArcFaceHead(512, num_classes, scale=30.0, margin=0.35)
    """

    def __init__(
        self,
        in_features: int = 1025,
        hidden_features: int = 512,
        num_classes: int = 899,
        scale: float = 30.0,
        margin: float = 0.35,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_classes = num_classes

        # Feature fusion / projection layer
        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # ArcFace Metric Learning Head
        self.head = ArcFaceHead(
            in_features=hidden_features,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts projected 512-D latent representations."""
        return self.projector(x)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 1025).
            labels: Optional ground-truth target label tensor of shape (batch_size,).
        Returns:
            Logits of shape (batch_size, num_classes).
        """
        latent = self.projector(x)
        logits = self.head(latent, labels=labels)
        return logits


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss over ArcFace margins:
        FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    where p_t is the model probability for the true class.
    Penalizes errors on long-tail / sparse classes.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        Args:
            logits: Predicted raw logits of shape (batch_size, num_classes).
            targets: Ground-truth class labels of shape (batch_size,).
        Returns:
            Scalar focal loss (or per-sample loss if reduction='none').
        """
        log_p = F.log_softmax(logits, dim=-1)
        target_log_p = log_p.gather(1, targets.unsqueeze(1).long()).squeeze(1)
        p_t = target_log_p.exp().clamp(min=self.eps, max=1.0)

        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = -focal_weight * target_log_p

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
