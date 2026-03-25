"""Image/gene contrastive model used by GLIP."""

from __future__ import annotations

import importlib
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from glip import config as CFG


UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"

MODEL_NAME_ALIASES = {
    "uni": UNI_MODEL_NAME,
    "uni2-h": UNI_MODEL_NAME,
}

UNI2_H_BACKBONE_NAME = "vit_giant_patch14_224"


def resolve_image_model_name(model_name: str) -> str:
    normalized = str(model_name).strip()
    if not normalized:
        raise ValueError("Image encoder model name cannot be empty.")
    if normalized.startswith("hf_hub:"):
        normalized = "hf-hub:" + normalized[len("hf_hub:"):]
    return MODEL_NAME_ALIASES.get(normalized.lower(), normalized)


def _import_timm():
    try:
        return importlib.import_module("timm")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "timm is required for non-ResNet image encoders. "
            "Install timm in the training environment or run with --model resnet50."
        ) from exc


def _infer_output_dim(model: nn.Module) -> int:
    output_dim = getattr(model, "num_features", None)
    if output_dim is not None:
        output_dim = int(output_dim)
        if output_dim > 0:
            return output_dim

    feature_info = getattr(model, "feature_info", None)
    if feature_info is not None and hasattr(feature_info, "channels"):
        channels = feature_info.channels()
        if channels:
            return int(channels[-1])

    raise AttributeError("Unable to infer the output dimension for the selected image encoder.")


def _load_local_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)}")

    model_state = model.state_dict()
    filtered_state_dict = {}

    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            filtered_state_dict[clean_key] = value

    model.load_state_dict(filtered_state_dict, strict=False)


def _build_resnet50(pretrained: bool, checkpoint_path: str = "") -> Tuple[nn.Module, int]:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained and not checkpoint_path else None
        backbone = models.resnet50(weights=weights)
    except Exception as exc:
        if pretrained and not checkpoint_path:
            raise RuntimeError(
                "Failed to initialize a pretrained ResNet50. "
                "Run with --pretrained false if you want random initialization."
            ) from exc
        backbone = models.resnet50(weights=None)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    if checkpoint_path:
        _load_local_checkpoint(backbone, checkpoint_path)
    return backbone, feature_dim


def _build_uni2_h(pretrained: bool, checkpoint_path: str) -> Tuple[nn.Module, int]:
    timm = _import_timm()
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    if pretrained and not checkpoint_path:
        model = timm.create_model(UNI_MODEL_NAME, pretrained=True, **timm_kwargs)
    else:
        model = timm.create_model(UNI2_H_BACKBONE_NAME, pretrained=False, **timm_kwargs)
        if checkpoint_path:
            _load_local_checkpoint(model, checkpoint_path)
    return model, _infer_output_dim(model)


def _build_timm_image_encoder(model_name: str, pretrained: bool, checkpoint_path: str) -> Tuple[nn.Module, int]:
    resolved_model_name = resolve_image_model_name(model_name)
    if resolved_model_name == UNI_MODEL_NAME:
        return _build_uni2_h(pretrained=pretrained, checkpoint_path=checkpoint_path)

    timm = _import_timm()
    model = timm.create_model(
        resolved_model_name,
        pretrained=pretrained and not checkpoint_path,
        num_classes=0,
        global_pool="avg",
    )
    if checkpoint_path:
        _load_local_checkpoint(model, checkpoint_path)
    return model, _infer_output_dim(model)


def _build_image_encoder(model_name: str, pretrained: bool, checkpoint_path: str) -> Tuple[nn.Module, int]:
    resolved_model_name = resolve_image_model_name(model_name)
    if resolved_model_name == "resnet50":
        return _build_resnet50(pretrained=pretrained, checkpoint_path=checkpoint_path)
    return _build_timm_image_encoder(
        model_name=resolved_model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )


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
        model_name: str = CFG.MODEL_NAME,
        pretrained: bool = True,
        image_encoder_checkpoint: str = CFG.IMAGE_ENCODER_CHECKPOINT,
        temperature: float = CFG.TEMPERATURE,
    ) -> None:
        super().__init__()
        self.model_name = resolve_image_model_name(model_name)
        self.image_encoder, image_feature_dim = _build_image_encoder(
            model_name=self.model_name,
            pretrained=pretrained,
            checkpoint_path=image_encoder_checkpoint,
        )
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
