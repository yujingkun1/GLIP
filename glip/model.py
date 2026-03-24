"""Image/gene contrastive model used by GLIP."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from glip import config as CFG


def _build_resnet50(pretrained: bool) -> Tuple[nn.Module, int]:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Failed to initialize a pretrained ResNet50. "
                "Run with --pretrained false if you want random initialization."
            ) from exc
        backbone = models.resnet50(weights=None)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = CFG.PROJECTION_DIM,
        dropout: float = CFG.DROPOUT,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def soft_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_softmax = nn.LogSoftmax(dim=-1)
    return (-targets * log_softmax(predictions)).sum(dim=1)


class ContrastiveImageGeneModel(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        *,
        pretrained: bool = True,
        temperature: float = CFG.TEMPERATURE,
    ) -> None:
        super().__init__()
        self.image_encoder, image_feature_dim = _build_resnet50(pretrained=pretrained)
        self.image_projection = ProjectionHead(embedding_dim=image_feature_dim)
        self.gene_projection = ProjectionHead(embedding_dim=gene_dim)
        self.temperature = float(temperature)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        return self.image_projection(features)

    def encode_genes(self, expressions: torch.Tensor) -> torch.Tensor:
        return self.gene_projection(expressions)

    def forward(self, batch: dict) -> torch.Tensor:
        image_embeddings = self.encode_images(batch["image"])
        gene_embeddings = self.encode_genes(batch["expression"])

        logits = (gene_embeddings @ image_embeddings.T) / self.temperature
        image_similarity = image_embeddings @ image_embeddings.T
        gene_similarity = gene_embeddings @ gene_embeddings.T
        targets = F.softmax(((image_similarity + gene_similarity) / 2.0) / self.temperature, dim=-1)

        gene_loss = soft_cross_entropy(logits, targets)
        image_loss = soft_cross_entropy(logits.T, targets.T)
        return ((gene_loss + image_loss) / 2.0).mean()
