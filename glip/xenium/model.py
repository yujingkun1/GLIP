"""Image/gene contrastive model used by GLIP."""

from __future__ import annotations

import importlib
import math
import os
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from . import config as CFG


UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"
H0MINI_MODEL_NAME = "hf-hub:bioptimus/H0-mini"

MODEL_NAME_ALIASES = {
    "uni": UNI_MODEL_NAME,
    "uni2-h": UNI_MODEL_NAME,
    "h0mini": H0MINI_MODEL_NAME,
    "h0-mini": H0MINI_MODEL_NAME,
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


def _build_h0mini(pretrained: bool, checkpoint_path: str, trainable: bool = False) -> Tuple[nn.Module, int]:
    """Build H0-mini encoder (ViT-B/14 reg4, 768-dim output)."""
    timm = _import_timm()

    if pretrained and not checkpoint_path:
        # Load from HuggingFace
        model = timm.create_model(H0MINI_MODEL_NAME, pretrained=True, num_classes=0)
    else:
        # Load from local checkpoint with correct architecture
        model = timm.create_model(
            "vit_base_patch14_reg4_dinov2",
            pretrained=False,
            num_classes=0,
            img_size=224,
            reg_tokens=4,
            dynamic_img_size=True,
            mlp_ratio=5.33334,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        if checkpoint_path:
            _load_local_checkpoint(model, checkpoint_path)

    # Set trainability
    for param in model.parameters():
        param.requires_grad = trainable

    return model, 768  # H0-mini output dimension


def _build_timm_image_encoder(model_name: str, pretrained: bool, checkpoint_path: str, trainable: bool = False) -> Tuple[nn.Module, int]:
    resolved_model_name = resolve_image_model_name(model_name)
    if resolved_model_name == UNI_MODEL_NAME:
        return _build_uni2_h(pretrained=pretrained, checkpoint_path=checkpoint_path)
    if resolved_model_name == H0MINI_MODEL_NAME:
        return _build_h0mini(pretrained=pretrained, checkpoint_path=checkpoint_path, trainable=trainable)

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


def _build_image_encoder(model_name: str, pretrained: bool, checkpoint_path: str, trainable: bool = False) -> Tuple[nn.Module, int]:
    resolved_model_name = resolve_image_model_name(model_name)
    if resolved_model_name == "resnet50":
        return _build_resnet50(pretrained=pretrained, checkpoint_path=checkpoint_path)
    return _build_timm_image_encoder(
        model_name=resolved_model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        trainable=trainable,
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


def _resolve_scfoundation_module(repo_dir: str):
    model_dir = os.path.join(os.path.abspath(repo_dir), "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"scFoundation model directory not found: {model_dir}")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    try:
        from load import convertconfig, gatherData  # type: ignore
        from pretrainmodels.select_model import select_model  # type: ignore
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", str(exc))
        raise RuntimeError(
            "scFoundation dependencies are incomplete. "
            f"Missing module: {missing_name}. Install the required packages, especially local_attention."
        ) from exc
    return convertconfig, gatherData, select_model


class ScFoundationGeneBackbone(nn.Module):
    def __init__(
        self,
        *,
        repo_dir: str,
        checkpoint_path: str,
        checkpoint_key: str = CFG.SCFOUNDATION_KEY,
        pool_type: str = CFG.SCFOUNDATION_POOL_TYPE,
        tgthighres: str = CFG.SCFOUNDATION_TGTHIGHRES,
    ) -> None:
        super().__init__()
        convertconfig, gather_data_fn, select_model = _resolve_scfoundation_module(repo_dir)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint_key not in checkpoint:
            raise KeyError(f"scFoundation checkpoint key not found: {checkpoint_key}")

        model_data = convertconfig(checkpoint[checkpoint_key])
        config = model_data["config"]
        if "qv_dim" not in config:
            if config.get("model") != "mae_autobin":
                if "dim_head" in config:
                    config["qv_dim"] = config["dim_head"]
                else:
                    config["qv_dim"] = 64
        if "ppi_edge" not in config:
            config["ppi_edge"] = None

        model = select_model(config)
        model.load_state_dict(model_data["model_state_dict"])

        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        self.config = config
        self.gather_data = gather_data_fn
        self.pool_type = str(pool_type).strip().lower()
        self.tgthighres = str(tgthighres).strip().lower()
        hidden_dim = int(config["encoder"]["hidden_dim"])
        self.output_dim = hidden_dim * 4 if self.pool_type == "all" else hidden_dim

    def _build_resolution_tokens(self, total_count: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        safe_total = total_count.clamp_min(1.0)
        source_token = torch.log10(safe_total)
        if not self.tgthighres:
            target_token = source_token
        elif self.tgthighres.startswith("f"):
            target_token = torch.log10(safe_total * float(self.tgthighres[1:]))
        elif self.tgthighres.startswith("a"):
            target_token = source_token + float(self.tgthighres[1:])
        elif self.tgthighres.startswith("t"):
            target_token = torch.full_like(source_token, float(self.tgthighres[1:]))
        else:
            raise ValueError("scFoundation tgthighres must start with f, a, or t")
        return target_token, source_token

    def forward(self, expressions: torch.Tensor) -> torch.Tensor:
        expressions = expressions.float().clamp_min(0.0)
        total_count = expressions.sum(dim=1, keepdim=True)
        normalized = torch.where(
            total_count > 0,
            torch.log1p(expressions / total_count.clamp_min(1e-6) * 1e4),
            torch.zeros_like(expressions),
        )
        target_token, source_token = self._build_resolution_tokens(total_count)
        model_input = torch.cat([normalized, target_token, source_token], dim=1)

        data_gene_ids = torch.arange(model_input.shape[1], device=model_input.device).unsqueeze(0).expand(model_input.shape[0], -1)
        value_labels = model_input > 0
        gathered_values, padding_labels = self.gather_data(model_input, value_labels, self.config["pad_token_id"])
        position_gene_ids, _ = self.gather_data(data_gene_ids, value_labels, self.config["pad_token_id"])

        token_embeddings = self.token_emb(gathered_values.unsqueeze(2).float(), output_weight=0)
        token_embeddings = token_embeddings + self.pos_emb(position_gene_ids)
        encoded = self.encoder(token_embeddings, padding_labels)

        if self.pool_type == "max":
            pooled, _ = torch.max(encoded, dim=1)
            return pooled
        if self.pool_type != "all":
            raise ValueError("scFoundation pool_type must be one of: all, max")

        token_target = encoded[:, -1, :]
        token_source = encoded[:, -2, :]
        token_max, _ = torch.max(encoded[:, :-2, :], dim=1)
        token_mean = torch.mean(encoded[:, :-2, :], dim=1)
        return torch.cat([token_target, token_source, token_max, token_mean], dim=1)


class ContrastiveImageGeneModel(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        *,
        model_name: str = CFG.MODEL_NAME,
        pretrained: bool = True,
        image_encoder_checkpoint: str = CFG.IMAGE_ENCODER_CHECKPOINT,
        temperature: float = CFG.TEMPERATURE,
        gene_encoder: str = CFG.GENE_ENCODER,
        scfoundation_repo_dir: str = CFG.SCFOUNDATION_REPO_DIR,
        scfoundation_checkpoint: str = CFG.SCFOUNDATION_CHECKPOINT,
        scfoundation_key: str = CFG.SCFOUNDATION_KEY,
        scfoundation_pool_type: str = CFG.SCFOUNDATION_POOL_TYPE,
        scfoundation_tgthighres: str = CFG.SCFOUNDATION_TGTHIGHRES,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = resolve_image_model_name(model_name)
        self.image_encoder, image_feature_dim = _build_image_encoder(
            model_name=self.model_name,
            pretrained=pretrained,
            checkpoint_path=image_encoder_checkpoint,
            trainable=trainable,
        )
        self.image_projection = ProjectionHead(embedding_dim=image_feature_dim)

        self.gene_encoder_type = str(gene_encoder).strip().lower()
        if self.gene_encoder_type == "projection":
            gene_feature_dim = int(gene_dim)
            self.gene_backbone = None
        elif self.gene_encoder_type == "scfoundation":
            self.gene_backbone = ScFoundationGeneBackbone(
                repo_dir=scfoundation_repo_dir,
                checkpoint_path=scfoundation_checkpoint,
                checkpoint_key=scfoundation_key,
                pool_type=scfoundation_pool_type,
                tgthighres=scfoundation_tgthighres,
            )
            gene_feature_dim = int(self.gene_backbone.output_dim)
        else:
            raise ValueError("gene_encoder must be one of: projection, scfoundation")

        self.gene_projection = ProjectionHead(embedding_dim=gene_feature_dim)
        self.temperature = float(temperature)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        return self.image_projection(features)

    def encode_gene_inputs(self, expressions: torch.Tensor) -> torch.Tensor:
        if self.gene_encoder_type == "projection":
            return expressions
        return self.gene_backbone(expressions)

    def encode_genes(self, expressions: torch.Tensor) -> torch.Tensor:
        features = self.encode_gene_inputs(expressions)
        return self.gene_projection(features)

    def compute_image_gene_loss(self, batch: dict, return_gene_embeddings: bool = False):
        image_embeddings = self.encode_images(batch["image"])
        gene_inputs = batch.get("encoder_expression", batch["expression"])
        gene_embeddings = self.encode_genes(gene_inputs)

        logits = (gene_embeddings @ image_embeddings.T) / self.temperature
        image_similarity = image_embeddings @ image_embeddings.T
        gene_similarity = gene_embeddings @ gene_embeddings.T
        targets = F.softmax(((image_similarity + gene_similarity) / 2.0) / self.temperature, dim=-1)

        gene_loss = soft_cross_entropy(logits, targets)
        image_loss = soft_cross_entropy(logits.T, targets.T)
        loss = ((gene_loss + image_loss) / 2.0).mean()
        if return_gene_embeddings:
            return loss, gene_embeddings
        return loss

    def forward(self, batch: dict) -> torch.Tensor:
        return self.compute_image_gene_loss(batch, return_gene_embeddings=False)
