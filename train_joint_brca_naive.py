#!/usr/bin/env python3
"""Stage 2 minimal naive joint training for BRCA Visium + Xenium pseudo-spots.

This script intentionally adds only one core idea beyond accepted baselines:
train a single shared image-expression retrieval model on pooled Visium and
Xenium pseudo-spot samples within a shared gene space.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from glip.utils import AvgMeter, compute_pearson_metrics, get_lr, parse_bool, save_json, seed_everything
from glip.visium.dataset import CLIPDataset
from glip.visium.models import CLIPModel, CLIPModel_CLIP, CLIPModel_UNI, CLIPModel_ViT, CLIPModel_ViT_L, CLIPModel_resnet101, CLIPModel_resnet152
from glip.xenium.pseudospot import XeniumPseudoSpotDataset, build_pseudospot_output_dir

UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"
MODEL_NAME_ALIASES = {"uni": UNI_MODEL_NAME, "uni2-h": UNI_MODEL_NAME}
DEFAULT_VISIUM_SAMPLE_IDS = [f"SPA{i}" for i in range(119, 155)]
DEFAULT_SHARED_GENE_FILE = "/data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt"


def parse_sample_ids(raw_sample_ids: str | Sequence[str] | None) -> List[str]:
    if raw_sample_ids is None:
        return []
    if isinstance(raw_sample_ids, str):
        return [sample_id.strip() for sample_id in raw_sample_ids.split(",") if sample_id.strip()]
    return [str(sample_id).strip() for sample_id in raw_sample_ids if str(sample_id).strip()]


def load_fixed_fold_manifest(manifest_path: str, fold_index: int) -> Dict:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    folds = payload.get("folds") if isinstance(payload, dict) else payload
    if not isinstance(folds, list):
        raise ValueError(f"Invalid fold manifest format: {manifest_path}")

    for fold in folds:
        if int(fold.get("fold_index", -1)) != int(fold_index):
            continue
        train_samples = parse_sample_ids(fold.get("train_samples"))
        test_samples = parse_sample_ids(fold.get("test_samples"))
        if not train_samples or not test_samples:
            raise ValueError(f"Fold {fold_index} in {manifest_path} is missing train/test samples")
        sample_ids = parse_sample_ids(payload.get("sample_ids")) if isinstance(payload, dict) else []
        if not sample_ids:
            sample_ids = sorted(set(train_samples + test_samples))
        return {
            "fold_index": int(fold_index),
            "split_name": payload.get("split_name", os.path.basename(manifest_path)) if isinstance(payload, dict) else os.path.basename(manifest_path),
            "sample_ids": sample_ids,
            "train_samples": train_samples,
            "test_samples": test_samples,
        }

    raise ValueError(f"Fold index {fold_index} not found in manifest: {manifest_path}")


@dataclass
class RunConfig:
    visium_heldout_sample: str
    visium_sample_ids: List[str]
    visium_train_samples: List[str]
    visium_test_samples: List[str]
    visium_fold_manifest: str
    visium_fold_index: int
    shared_gene_file: str
    xenium_sample_id: str
    xenium_reference_visium_sample_id: str
    xenium_test_fold: int
    xenium_num_position_folds: int
    model: str
    resolved_model_name: str
    pretrained: bool
    image_encoder_checkpoint: str
    batch_size: int
    eval_batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    top_k: int
    retrieval_chunk_size: int
    seed: int
    device: str
    num_workers: int
    max_visium_train_spots: int
    max_visium_test_spots: int
    max_xenium_train_spots: int
    max_xenium_test_spots: int
    eval_bank_mode: str
    module_naive_joint: bool
    module_platform_token: bool
    module_shared_private: bool
    keep_contrastive_loss: bool
    module_platform_image_encoder: bool
    shared_private_dim: int
    private_dim: int
    private_gate: float
    shared_align_weight: float
    orth_weight: float
    module_vae_decoder: bool
    vae_latent_dim: int
    vae_hidden_dim: int
    vae_recon_weight: float
    vae_kl_weight: float
    module_ot: bool
    module_image_ot: bool
    module_gene_ot: bool
    ot_transport: str
    ot_image_weight: float
    ot_gene_weight: float
    ot_sinkhorn_eps: float
    ot_sinkhorn_iters: int
    uot_marginal_weight: float
    module_gene_completion: bool
    module_cell_refine: bool
    note: str


class WrappedVisiumDataset(Dataset):
    def __init__(self, dataset: Dataset, source_name: str, platform_id: int = 0) -> None:
        self.dataset = dataset
        self.source_name = source_name
        self.platform_id = int(platform_id)
        self.num_features = getattr(dataset, "num_features", None)
        self.num_genes = getattr(dataset, "num_genes", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        sample = self.dataset[index]
        sample["source"] = self.source_name
        sample["platform_id"] = torch.tensor(self.platform_id, dtype=torch.long)
        return sample


class WrappedXeniumPseudoSpotDataset(Dataset):
    def __init__(self, dataset: XeniumPseudoSpotDataset, source_name: str, platform_id: int = 1) -> None:
        self.dataset = dataset
        self.source_name = source_name
        self.platform_id = int(platform_id)
        self.num_features = getattr(dataset, "num_features", None)
        self.num_genes = getattr(dataset, "num_features", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        sample = self.dataset[index]
        return {
            "image": sample["image"],
            "reduced_expression": sample["encoder_expression"].float(),
            "barcode": f"{self.source_name}_{int(sample['spot_id'])}",
            "sample_id": self.source_name,
            "spatial_coords": [float(sample["centroid_x"]), float(sample["centroid_y"])],
            "source": self.source_name,
            "platform_id": torch.tensor(self.platform_id, dtype=torch.long),
        }


def create_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
        persistent_workers=num_workers > 0,
    )


def move_batch_to_device(batch, device: torch.device):
    moved = {
        "image": batch["image"].to(device, non_blocking=True),
        "reduced_expression": batch["reduced_expression"].to(device, non_blocking=True),
    }
    if "platform_id" in batch:
        moved["platform_id"] = batch["platform_id"].to(device, non_blocking=True)
    return moved




class PlatformConditionedModel(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        use_platform_token: bool = False,
        num_platforms: int = 2,
        use_shared_private: bool = False,
        shared_private_dim: int = 256,
        private_dim: int = 64,
        private_gate: float = 0.25,
        shared_align_weight: float = 0.05,
        orth_weight: float = 0.01,
        use_vae_decoder: bool = False,
        keep_contrastive_loss: bool = False,
        use_platform_image_encoder: bool = False,
        vae_latent_dim: int = 128,
        vae_hidden_dim: int = 512,
        vae_recon_weight: float = 1.0,
        vae_kl_weight: float = 1e-4,
        output_gene_dim: int | None = None,
        use_image_ot: bool = False,
        use_gene_ot: bool = False,
        ot_transport: str = "ot",
        ot_image_weight: float = 0.0,
        ot_gene_weight: float = 0.0,
        ot_sinkhorn_eps: float = 0.05,
        ot_sinkhorn_iters: int = 50,
        uot_marginal_weight: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.use_platform_token = bool(use_platform_token)
        self.use_shared_private = bool(use_shared_private)
        self.use_vae_decoder = bool(use_vae_decoder)
        self.keep_contrastive_loss = bool(keep_contrastive_loss)
        self.use_platform_image_encoder = bool(use_platform_image_encoder)
        if self.use_shared_private and self.use_vae_decoder:
            raise ValueError("--module-shared-private and --module-vae-decoder are mutually exclusive.")
        self.use_image_ot = bool(use_image_ot)
        self.use_gene_ot = bool(use_gene_ot)
        self.ot_transport = str(ot_transport).strip().lower()
        if self.ot_transport not in {"ot", "uot"}:
            raise ValueError(f"Unsupported OT transport mode: {ot_transport}")
        self.ot_image_weight = float(ot_image_weight)
        self.ot_gene_weight = float(ot_gene_weight)
        self.shared_align_weight = float(shared_align_weight)
        self.orth_weight = float(orth_weight)
        self.vae_recon_weight = float(vae_recon_weight)
        self.vae_kl_weight = float(vae_kl_weight)
        self.ot_sinkhorn_eps = float(ot_sinkhorn_eps)
        self.ot_sinkhorn_iters = int(ot_sinkhorn_iters)
        self.uot_marginal_weight = float(uot_marginal_weight)
        self.temperature = getattr(base_model, "temperature")
        projection_dim = int(base_model.image_projection.projection.out_features)
        self.projection_dim = projection_dim
        self.shared_private_dim = int(shared_private_dim)
        self.private_dim = int(private_dim)
        self.private_gate = float(private_gate)
        self.vae_latent_dim = int(vae_latent_dim)
        self.vae_hidden_dim = int(vae_hidden_dim)
        self.output_gene_dim = int(output_gene_dim) if output_gene_dim is not None else None
        self.platform_token = torch.nn.Embedding(num_platforms, projection_dim) if self.use_platform_token else None
        self.xenium_image_encoder = copy.deepcopy(base_model.image_encoder) if self.use_platform_image_encoder else None
        if self.use_vae_decoder:
            if self.vae_latent_dim <= 0:
                raise ValueError("vae_latent_dim must be positive when VAE decoder is enabled.")
            if self.vae_hidden_dim <= 0:
                raise ValueError("vae_hidden_dim must be positive when VAE decoder is enabled.")
            if self.output_gene_dim is None or self.output_gene_dim <= 0:
                raise ValueError("output_gene_dim must be positive when VAE decoder is enabled.")
            self.vae_mu = torch.nn.Linear(projection_dim, self.vae_latent_dim)
            self.vae_logvar = torch.nn.Linear(projection_dim, self.vae_latent_dim)
            self.vae_decoder = torch.nn.Sequential(
                torch.nn.Linear(self.vae_latent_dim, self.vae_hidden_dim),
                torch.nn.GELU(),
                torch.nn.LayerNorm(self.vae_hidden_dim),
                torch.nn.Linear(self.vae_hidden_dim, self.output_gene_dim),
            )
        else:
            self.vae_mu = None
            self.vae_logvar = None
            self.vae_decoder = None
        if self.use_shared_private:
            if self.shared_private_dim <= 0:
                raise ValueError("shared_private_dim must be positive when shared/private is enabled.")
            if self.private_dim <= 0:
                raise ValueError("private_dim must be positive when shared/private is enabled.")
            self.image_shared_head = self._make_latent_head(projection_dim, self.shared_private_dim)
            self.gene_shared_head = self._make_latent_head(projection_dim, self.shared_private_dim)
            self.image_private_head = self._make_latent_head(projection_dim, self.private_dim)
            self.gene_private_head = self._make_latent_head(projection_dim, self.private_dim)
            self.image_private_adapter = torch.nn.Linear(self.private_dim, self.shared_private_dim)
            self.gene_private_adapter = torch.nn.Linear(self.private_dim, self.shared_private_dim)
            self.image_fused_norm = torch.nn.LayerNorm(self.shared_private_dim)
            self.gene_fused_norm = torch.nn.LayerNorm(self.shared_private_dim)
        else:
            self.image_shared_head = None
            self.gene_shared_head = None
            self.image_private_head = None
            self.gene_private_head = None
            self.image_private_adapter = None
            self.gene_private_adapter = None
            self.image_fused_norm = None
            self.gene_fused_norm = None

    def encode_image_base(self, images: torch.Tensor, platform_ids: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.encode_image_features(images, platform_ids)
        image_embeddings = self.base_model.image_projection(image_features)
        return self._apply_platform_token(image_embeddings, platform_ids)

    def encode_image_features(self, images: torch.Tensor, platform_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.xenium_image_encoder is None or platform_ids is None:
            return self.base_model.image_encoder(images)
        if images.size(0) != platform_ids.size(0):
            raise ValueError("images and platform_ids batch sizes do not match.")
        output = None
        vis_mask = platform_ids == 0
        xen_mask = platform_ids == 1
        other_mask = ~(vis_mask | xen_mask)
        for mask, encoder in (
            (vis_mask | other_mask, self.base_model.image_encoder),
            (xen_mask, self.xenium_image_encoder),
        ):
            if not bool(mask.any()):
                continue
            features = encoder(images[mask])
            if output is None:
                output = features.new_empty((images.size(0), features.size(1)))
            output[mask] = features
        if output is None:
            raise RuntimeError("Unable to encode empty image batch.")
        return output

    @staticmethod
    def _make_latent_head(input_dim: int, output_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(output_dim),
        )

    @property
    def image_encoder(self):
        return self.base_model.image_encoder

    @property
    def image_projection(self):
        return self.base_model.image_projection

    @property
    def spot_projection(self):
        return self.base_model.spot_projection

    def _apply_platform_token(self, embeddings: torch.Tensor, platform_ids: torch.Tensor | None) -> torch.Tensor:
        if self.platform_token is None or platform_ids is None:
            return embeddings
        return embeddings + self.platform_token(platform_ids)

    def encode_images(
        self,
        images: torch.Tensor,
        platform_ids: torch.Tensor | None = None,
        embedding_view: str = "fused",
    ) -> torch.Tensor:
        image_embeddings = self.encode_image_base(images, platform_ids=platform_ids if self.use_platform_image_encoder else None)
        if self.use_shared_private:
            return self._select_embedding_view(
                self._split_embeddings(image_embeddings, platform_ids, modality="image"),
                embedding_view=embedding_view,
            )
        return self._apply_platform_token(image_embeddings, platform_ids)

    def encode_spots(
        self,
        reduced_expression: torch.Tensor,
        platform_ids: torch.Tensor | None = None,
        embedding_view: str = "fused",
    ) -> torch.Tensor:
        spot_embeddings = self.base_model.spot_projection(reduced_expression)
        if self.use_shared_private:
            return self._select_embedding_view(
                self._split_embeddings(spot_embeddings, platform_ids, modality="gene"),
                embedding_view=embedding_view,
            )
        return self._apply_platform_token(spot_embeddings, platform_ids)

    def _split_embeddings(
        self,
        base_embeddings: torch.Tensor,
        platform_ids: torch.Tensor | None,
        *,
        modality: str,
    ) -> Dict[str, torch.Tensor]:
        if modality == "image":
            shared_head = self.image_shared_head
            private_head = self.image_private_head
            private_adapter = self.image_private_adapter
            fused_norm = self.image_fused_norm
        elif modality == "gene":
            shared_head = self.gene_shared_head
            private_head = self.gene_private_head
            private_adapter = self.gene_private_adapter
            fused_norm = self.gene_fused_norm
        else:
            raise ValueError(f"Unsupported shared/private modality: {modality}")
        if shared_head is None or private_head is None or private_adapter is None or fused_norm is None:
            raise RuntimeError("Shared/private heads are not initialized.")
        shared = shared_head(base_embeddings)
        private_input = self._apply_platform_token(base_embeddings, platform_ids)
        private = private_head(private_input)
        private_projected = private_adapter(private)
        fused = fused_norm(shared + self.private_gate * private_projected)
        return {
            "shared": shared,
            "private": private,
            "private_projected": private_projected,
            "fused": fused,
        }

    def _select_embedding_view(self, views: Dict[str, torch.Tensor], embedding_view: str) -> torch.Tensor:
        view = str(embedding_view or "fused").strip().lower()
        if view == "alignment":
            view = "shared"
        if view not in views:
            raise ValueError(f"Unsupported embedding_view={embedding_view!r}; expected one of {sorted(views)}")
        return views[view]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def predict_expression_from_images(
        self,
        images: torch.Tensor,
        platform_ids: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        if not self.use_vae_decoder or self.vae_mu is None or self.vae_logvar is None or self.vae_decoder is None:
            raise RuntimeError("VAE decoder is not enabled.")
        image_embeddings = self.encode_image_base(images, platform_ids=platform_ids)
        mu = self.vae_mu(image_embeddings)
        logvar = self.vae_logvar(image_embeddings).clamp(min=-12.0, max=12.0)
        z = self.reparameterize(mu, logvar)
        prediction = self.vae_decoder(z)
        if return_latent:
            return prediction, mu, logvar, z, image_embeddings
        return prediction

    def compute_vae_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_per_sample.mean()

    def compute_contrastive_loss(
        self,
        image_embeddings: torch.Tensor,
        spot_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1)
        spots_loss = F.cross_entropy(logits, targets, reduction='none') if targets.dtype in (torch.long, torch.int64) else (-targets * F.log_softmax(logits, dim=-1)).sum(1)
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none') if targets.dtype in (torch.long, torch.int64) else (-targets.T * F.log_softmax(logits.T, dim=-1)).sum(1)
        loss = (images_loss + spots_loss) / 2.0
        return loss.mean()

    def compute_embedding_ot_loss(
        self,
        embeddings: torch.Tensor,
        platform_ids: torch.Tensor | None,
        *,
        enabled: bool,
        weight: float,
    ) -> tuple[torch.Tensor, int]:
        zero = embeddings.new_zeros(())
        if (
            (not enabled)
            or weight <= 0.0
            or platform_ids is None
            or embeddings.size(0) < 2
        ):
            return zero, 0

        vis_mask = platform_ids == 0
        xen_mask = platform_ids == 1
        vis_count = int(vis_mask.sum().item())
        xen_count = int(xen_mask.sum().item())
        if vis_count == 0 or xen_count == 0:
            return zero, 0

        vis_embeddings = F.normalize(embeddings[vis_mask].float(), dim=1)
        xen_embeddings = F.normalize(embeddings[xen_mask].float(), dim=1)
        cost = 1.0 - torch.clamp(vis_embeddings @ xen_embeddings.T, min=-1.0, max=1.0)

        num_vis = vis_embeddings.size(0)
        num_xen = xen_embeddings.size(0)
        a = torch.full((num_vis,), 1.0 / float(num_vis), device=embeddings.device, dtype=embeddings.dtype)
        b = torch.full((num_xen,), 1.0 / float(num_xen), device=embeddings.device, dtype=embeddings.dtype)

        if self.ot_transport == "uot":
            return self.compute_embedding_uot_loss(cost, a, b), min(vis_count, xen_count)

        return self.compute_embedding_balanced_ot_loss(cost, a, b), min(vis_count, xen_count)

    def compute_embedding_balanced_ot_loss(
        self,
        cost: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        kernel = torch.exp(-cost / max(self.ot_sinkhorn_eps, 1e-6)).clamp_min(1e-8)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(max(self.ot_sinkhorn_iters, 1)):
            kv = kernel @ v
            u = a / kv.clamp_min(1e-8)
            ktu = kernel.T @ u
            v = b / ktu.clamp_min(1e-8)

        transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        return torch.sum(transport * cost)

    def compute_embedding_uot_loss(
        self,
        cost: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        eps = max(self.ot_sinkhorn_eps, 1e-6)
        rho = max(self.uot_marginal_weight, 1e-6)
        tau = rho / (rho + eps)
        kernel = torch.exp(-cost / eps).clamp_min(1e-8)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(max(self.ot_sinkhorn_iters, 1)):
            kv = kernel @ v
            u = torch.pow(a / kv.clamp_min(1e-8), tau)
            ktu = kernel.T @ u
            v = torch.pow(b / ktu.clamp_min(1e-8), tau)

        transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        row_mass = transport.sum(dim=1)
        col_mass = transport.sum(dim=0)
        marginal_penalty = rho * (
            self.kl_divergence_with_reference(row_mass, a) +
            self.kl_divergence_with_reference(col_mass, b)
        )
        return torch.sum(transport * cost) + marginal_penalty

    def kl_divergence_with_reference(
        self,
        mass: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        mass = mass.clamp_min(1e-8)
        reference = reference.clamp_min(1e-8)
        return torch.sum(mass * (torch.log(mass) - torch.log(reference)) - mass + reference)

    def compute_image_ot_loss(
        self,
        image_embeddings: torch.Tensor,
        platform_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        return self.compute_embedding_ot_loss(
            image_embeddings,
            platform_ids,
            enabled=self.use_image_ot,
            weight=self.ot_image_weight,
        )

    def compute_gene_ot_loss(
        self,
        spot_embeddings: torch.Tensor,
        platform_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        return self.compute_embedding_ot_loss(
            spot_embeddings,
            platform_ids,
            enabled=self.use_gene_ot,
            weight=self.ot_gene_weight,
        )

    def compute_cross_covariance_loss(
        self,
        shared_embeddings: torch.Tensor,
        private_embeddings: torch.Tensor,
        platform_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        zero = shared_embeddings.new_zeros(())
        if shared_embeddings.size(0) < 2:
            return zero

        losses = []
        if platform_ids is None:
            masks = [torch.ones(shared_embeddings.size(0), dtype=torch.bool, device=shared_embeddings.device)]
        else:
            masks = [platform_ids == platform_id for platform_id in torch.unique(platform_ids)]

        for mask in masks:
            if int(mask.sum().item()) < 2:
                continue
            shared = shared_embeddings[mask].float()
            private = private_embeddings[mask].float()
            shared = shared - shared.mean(dim=0, keepdim=True)
            private = private - private.mean(dim=0, keepdim=True)
            denom = max(shared.size(0) - 1, 1)
            cross_covariance = (shared.T @ private) / float(denom)
            losses.append(cross_covariance.pow(2).mean())

        if not losses:
            return zero
        return torch.stack(losses).mean()

    def compute_shared_private_orth_loss(
        self,
        image_views: Dict[str, torch.Tensor],
        gene_views: Dict[str, torch.Tensor],
        platform_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        image_loss = self.compute_cross_covariance_loss(image_views["shared"], image_views["private"], platform_ids)
        gene_loss = self.compute_cross_covariance_loss(gene_views["shared"], gene_views["private"], platform_ids)
        return 0.5 * (image_loss + gene_loss)

    def forward(self, batch: Dict[str, torch.Tensor], return_components: bool = False):
        platform_ids = batch.get("platform_id")
        image_features = self.encode_image_features(batch["image"], platform_ids=platform_ids)
        image_base_embeddings = self.base_model.image_projection(image_features)
        spot_base_embeddings = self.base_model.spot_projection(batch["reduced_expression"])

        zero = image_base_embeddings.new_zeros(())
        vae_recon_loss = zero
        vae_kl_loss = zero
        if self.use_vae_decoder and not self.keep_contrastive_loss:
            if self.vae_mu is None or self.vae_logvar is None or self.vae_decoder is None:
                raise RuntimeError("VAE decoder is not initialized.")
            image_embeddings = self._apply_platform_token(image_base_embeddings, platform_ids)
            mu = self.vae_mu(image_embeddings)
            logvar = self.vae_logvar(image_embeddings).clamp(min=-12.0, max=12.0)
            z = self.reparameterize(mu, logvar)
            predictions = self.vae_decoder(z)
            vae_recon_loss = F.mse_loss(predictions, batch["reduced_expression"])
            vae_kl_loss = self.compute_vae_kl_loss(mu, logvar)
            image_ot_loss, image_ot_pairs = self.compute_image_ot_loss(image_embeddings, platform_ids)
            gene_ot_loss = zero
            gene_ot_pairs = 0
            shared_image_align_loss = zero
            shared_gene_align_loss = zero
            shared_image_align_pairs = 0
            shared_gene_align_pairs = 0
            orth_loss = zero
            main_loss = vae_recon_loss
            total_loss = (
                self.vae_recon_weight * vae_recon_loss
                + self.vae_kl_weight * vae_kl_loss
                + self.ot_image_weight * image_ot_loss
            )
            if not return_components:
                return total_loss
            return {
                "total_loss": total_loss,
                "main_loss": main_loss,
                "image_ot_loss": image_ot_loss,
                "gene_ot_loss": gene_ot_loss,
                "shared_image_align_loss": shared_image_align_loss,
                "shared_gene_align_loss": shared_gene_align_loss,
                "orth_loss": orth_loss,
                "vae_recon_loss": vae_recon_loss,
                "vae_kl_loss": vae_kl_loss,
                "image_ot_pairs": int(image_ot_pairs),
                "gene_ot_pairs": int(gene_ot_pairs),
                "shared_image_align_pairs": int(shared_image_align_pairs),
                "shared_gene_align_pairs": int(shared_gene_align_pairs),
            }

        if self.use_shared_private:
            image_views = self._split_embeddings(image_base_embeddings, platform_ids, modality="image")
            gene_views = self._split_embeddings(spot_base_embeddings, platform_ids, modality="gene")
            image_embeddings = image_views["fused"]
            spot_embeddings = gene_views["fused"]
        else:
            image_views = {}
            gene_views = {}
            image_embeddings = self._apply_platform_token(image_base_embeddings, platform_ids)
            spot_embeddings = self._apply_platform_token(spot_base_embeddings, platform_ids)

        main_loss = self.compute_contrastive_loss(image_embeddings, spot_embeddings)
        if self.use_vae_decoder:
            if self.vae_mu is None or self.vae_logvar is None or self.vae_decoder is None:
                raise RuntimeError("VAE decoder is not initialized.")
            mu = self.vae_mu(image_embeddings)
            logvar = self.vae_logvar(image_embeddings).clamp(min=-12.0, max=12.0)
            z = self.reparameterize(mu, logvar)
            predictions = self.vae_decoder(z)
            vae_recon_loss = F.mse_loss(predictions, batch["reduced_expression"])
            vae_kl_loss = self.compute_vae_kl_loss(mu, logvar)
        image_ot_loss, image_ot_pairs = self.compute_image_ot_loss(image_embeddings, platform_ids)
        gene_ot_loss, gene_ot_pairs = self.compute_gene_ot_loss(spot_embeddings, platform_ids)
        shared_image_align_loss = zero
        shared_gene_align_loss = zero
        shared_image_align_pairs = 0
        shared_gene_align_pairs = 0
        orth_loss = zero
        if self.use_shared_private:
            shared_image_align_loss, shared_image_align_pairs = self.compute_embedding_ot_loss(
                image_views["shared"],
                platform_ids,
                enabled=True,
                weight=self.shared_align_weight,
            )
            shared_gene_align_loss, shared_gene_align_pairs = self.compute_embedding_ot_loss(
                gene_views["shared"],
                platform_ids,
                enabled=True,
                weight=self.shared_align_weight,
            )
            orth_loss = self.compute_shared_private_orth_loss(image_views, gene_views, platform_ids)

        total_loss = (
            main_loss
            + self.vae_recon_weight * vae_recon_loss
            + self.vae_kl_weight * vae_kl_loss
            + self.ot_image_weight * image_ot_loss
            + self.ot_gene_weight * gene_ot_loss
            + self.shared_align_weight * (shared_image_align_loss + shared_gene_align_loss)
            + self.orth_weight * orth_loss
        )
        if not return_components:
            return total_loss
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "image_ot_loss": image_ot_loss,
            "gene_ot_loss": gene_ot_loss,
            "shared_image_align_loss": shared_image_align_loss,
            "shared_gene_align_loss": shared_gene_align_loss,
            "orth_loss": orth_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kl_loss": vae_kl_loss,
            "image_ot_pairs": int(image_ot_pairs),
            "gene_ot_pairs": int(gene_ot_pairs),
            "shared_image_align_pairs": int(shared_image_align_pairs),
            "shared_gene_align_pairs": int(shared_gene_align_pairs),
        }


def train_epoch(model, train_loader: DataLoader, optimizer, device: torch.device) -> Dict[str, float]:
    total_loss_meter = AvgMeter("train_loss")
    main_loss_meter = AvgMeter("main_loss")
    image_ot_loss_meter = AvgMeter("image_ot_loss")
    gene_ot_loss_meter = AvgMeter("gene_ot_loss")
    shared_image_align_loss_meter = AvgMeter("shared_image_align_loss")
    shared_gene_align_loss_meter = AvgMeter("shared_gene_align_loss")
    orth_loss_meter = AvgMeter("orth_loss")
    vae_recon_loss_meter = AvgMeter("vae_recon_loss")
    vae_kl_loss_meter = AvgMeter("vae_kl_loss")
    image_ot_active_batch_meter = AvgMeter("image_ot_active_batches")
    gene_ot_active_batch_meter = AvgMeter("gene_ot_active_batches")
    shared_image_align_active_batch_meter = AvgMeter("shared_image_align_active_batches")
    shared_gene_align_active_batch_meter = AvgMeter("shared_gene_align_active_batches")
    image_ot_pairs_meter = AvgMeter("image_ot_pairs")
    gene_ot_pairs_meter = AvgMeter("gene_ot_pairs")
    shared_image_align_pairs_meter = AvgMeter("shared_image_align_pairs")
    shared_gene_align_pairs_meter = AvgMeter("shared_gene_align_pairs")
    tqdm_object = tqdm(train_loader, total=len(train_loader), desc="joint_train")
    model.train()
    for batch in tqdm_object:
        moved = move_batch_to_device(batch, device)
        loss_components = model(moved, return_components=True)
        loss = loss_components["total_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = moved["image"].size(0)
        total_loss_meter.update(float(loss_components["total_loss"].item()), count)
        main_loss_meter.update(float(loss_components["main_loss"].item()), count)
        image_ot_loss_meter.update(float(loss_components["image_ot_loss"].item()), count)
        gene_ot_loss_meter.update(float(loss_components["gene_ot_loss"].item()), count)
        shared_image_align_loss_meter.update(float(loss_components["shared_image_align_loss"].item()), count)
        shared_gene_align_loss_meter.update(float(loss_components["shared_gene_align_loss"].item()), count)
        orth_loss_meter.update(float(loss_components["orth_loss"].item()), count)
        vae_recon_loss_meter.update(float(loss_components["vae_recon_loss"].item()), count)
        vae_kl_loss_meter.update(float(loss_components["vae_kl_loss"].item()), count)
        image_ot_active_batch_meter.update(1.0 if int(loss_components["image_ot_pairs"]) > 0 else 0.0, 1)
        gene_ot_active_batch_meter.update(1.0 if int(loss_components["gene_ot_pairs"]) > 0 else 0.0, 1)
        shared_image_align_active_batch_meter.update(1.0 if int(loss_components["shared_image_align_pairs"]) > 0 else 0.0, 1)
        shared_gene_align_active_batch_meter.update(1.0 if int(loss_components["shared_gene_align_pairs"]) > 0 else 0.0, 1)
        image_ot_pairs_meter.update(float(loss_components["image_ot_pairs"]), 1)
        gene_ot_pairs_meter.update(float(loss_components["gene_ot_pairs"]), 1)
        shared_image_align_pairs_meter.update(float(loss_components["shared_image_align_pairs"]), 1)
        shared_gene_align_pairs_meter.update(float(loss_components["shared_gene_align_pairs"]), 1)
        tqdm_object.set_postfix(
            total=total_loss_meter.avg,
            main=main_loss_meter.avg,
            img_ot=image_ot_loss_meter.avg,
            gene_ot=gene_ot_loss_meter.avg,
            sp_align=shared_gene_align_loss_meter.avg,
            orth=orth_loss_meter.avg,
            vae=vae_recon_loss_meter.avg,
            lr=get_lr(optimizer),
        )
    return {
        "total_loss": float(total_loss_meter.avg),
        "main_loss": float(main_loss_meter.avg),
        "image_ot_loss": float(image_ot_loss_meter.avg),
        "gene_ot_loss": float(gene_ot_loss_meter.avg),
        "shared_image_align_loss": float(shared_image_align_loss_meter.avg),
        "shared_gene_align_loss": float(shared_gene_align_loss_meter.avg),
        "orth_loss": float(orth_loss_meter.avg),
        "vae_recon_loss": float(vae_recon_loss_meter.avg),
        "vae_kl_loss": float(vae_kl_loss_meter.avg),
        "image_ot_active_fraction": float(image_ot_active_batch_meter.avg),
        "gene_ot_active_fraction": float(gene_ot_active_batch_meter.avg),
        "shared_image_align_active_fraction": float(shared_image_align_active_batch_meter.avg),
        "shared_gene_align_active_fraction": float(shared_gene_align_active_batch_meter.avg),
        "image_ot_pairs_mean": float(image_ot_pairs_meter.avg),
        "gene_ot_pairs_mean": float(gene_ot_pairs_meter.avg),
        "shared_image_align_pairs_mean": float(shared_image_align_pairs_meter.avg),
        "shared_gene_align_pairs_mean": float(shared_gene_align_pairs_meter.avg),
    }


def eval_loss(model, loader: DataLoader, device: torch.device, tag: str) -> AvgMeter:
    loss_meter = AvgMeter(tag)
    tqdm_object = tqdm(loader, total=len(loader), desc=tag)
    model.eval()
    with torch.no_grad():
        for batch in tqdm_object:
            moved = move_batch_to_device(batch, device)
            loss = model(moved)
            count = moved["image"].size(0)
            loss_meter.update(loss.item(), count)
            tqdm_object.set_postfix(loss=loss_meter.avg)
    return loss_meter


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


@torch.no_grad()
def collect_spot_bank(model, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    embeddings = []
    expressions = []
    sample_ids = []
    barcodes = []
    model.eval()
    for batch in loader:
        reduced_expression = batch["reduced_expression"].to(device, non_blocking=True)
        platform_ids = batch.get("platform_id")
        if platform_ids is not None:
            platform_ids = platform_ids.to(device, non_blocking=True)
        spot_embeddings = model.encode_spots(reduced_expression, platform_ids=platform_ids)
        embeddings.append(spot_embeddings.detach().cpu())
        expressions.append(batch["reduced_expression"].detach().cpu())
        sample_ids.extend(list(batch["sample_id"]))
        barcodes.extend(list(batch["barcode"]))
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "expressions": torch.cat(expressions, dim=0),
        "sample_ids": sample_ids,
        "barcodes": barcodes,
    }


@torch.no_grad()
def collect_image_queries(model, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    embeddings = []
    expressions = []
    sample_ids = []
    barcodes = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        platform_ids = batch.get("platform_id")
        if platform_ids is not None:
            platform_ids = platform_ids.to(device, non_blocking=True)
        image_embeddings = model.encode_images(images, platform_ids=platform_ids)
        embeddings.append(image_embeddings.detach().cpu())
        expressions.append(batch["reduced_expression"].detach().cpu())
        sample_ids.extend(list(batch["sample_id"]))
        barcodes.extend(list(batch["barcode"]))
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "expressions": torch.cat(expressions, dim=0),
        "sample_ids": sample_ids,
        "barcodes": barcodes,
    }


def predict_expression_from_retrieval(train_spot_bank, test_image_queries, top_k: int = 1, chunk_size: int = 1024):
    train_embeddings = F.normalize(train_spot_bank["embeddings"].float(), dim=1)
    train_expressions = train_spot_bank["expressions"].float()
    query_embeddings = F.normalize(test_image_queries["embeddings"].float(), dim=1)
    if train_embeddings.size(0) == 0:
        raise RuntimeError("Train spot bank is empty; cannot run retrieval.")
    k = max(1, min(int(top_k), train_embeddings.size(0)))
    predictions = []
    for start in range(0, query_embeddings.size(0), max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), query_embeddings.size(0))
        query_chunk = query_embeddings[start:end]
        similarity = query_chunk @ train_embeddings.T
        top_values, top_indices = similarity.topk(k, dim=1)
        matched_expressions = train_expressions.index_select(0, top_indices.reshape(-1))
        matched_expressions = matched_expressions.view(query_chunk.size(0), k, -1)
        if k == 1:
            chunk_prediction = matched_expressions[:, 0, :]
        else:
            weights = torch.softmax(top_values, dim=1).unsqueeze(-1)
            chunk_prediction = (matched_expressions * weights).sum(dim=1)
        predictions.append(chunk_prediction.cpu())
    return torch.cat(predictions, dim=0).numpy()


@torch.no_grad()
def predict_expression_from_vae_decoder(model, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    predictions = []
    targets = []
    sample_ids = []
    barcodes = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        platform_ids = batch.get("platform_id")
        if platform_ids is not None:
            platform_ids = platform_ids.to(device, non_blocking=True)
        predicted = model.predict_expression_from_images(images, platform_ids=platform_ids)
        predictions.append(predicted.detach().cpu())
        targets.append(batch["reduced_expression"].detach().cpu())
        sample_ids.extend(list(batch["sample_id"]))
        barcodes.extend(list(batch["barcode"]))
    return {
        "predictions": torch.cat(predictions, dim=0).numpy(),
        "targets": torch.cat(targets, dim=0).numpy(),
        "sample_ids": sample_ids,
        "barcodes": barcodes,
    }


def compute_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    overall_pearson = safe_pearson(predictions.reshape(-1), targets.reshape(-1))
    gene_pearsons = [safe_pearson(predictions[:, gene_idx], targets[:, gene_idx]) for gene_idx in range(predictions.shape[1])]
    spot_pearsons = [safe_pearson(predictions[spot_idx], targets[spot_idx]) for spot_idx in range(predictions.shape[0])]
    gene_pearsons_np = np.asarray(gene_pearsons, dtype=np.float32)
    spot_pearsons_np = np.asarray(spot_pearsons, dtype=np.float32)
    return {
        "overall_pearson": float(overall_pearson),
        "mean_gene_pearson": float(gene_pearsons_np.mean()) if gene_pearsons_np.size else 0.0,
        "std_gene_pearson": float(gene_pearsons_np.std()) if gene_pearsons_np.size else 0.0,
        "mean_spot_pearson": float(spot_pearsons_np.mean()) if spot_pearsons_np.size else 0.0,
        "std_spot_pearson": float(spot_pearsons_np.std()) if spot_pearsons_np.size else 0.0,
        "num_genes": int(predictions.shape[1]),
        "num_spots": int(predictions.shape[0]),
    }


def build_model(
    model_name: str,
    resolved_model_name: str,
    spot_embedding_dim: int,
    pretrained: bool,
    checkpoint_path: str,
    use_platform_token: bool = False,
    use_shared_private: bool = False,
    keep_contrastive_loss: bool = False,
    use_platform_image_encoder: bool = False,
    shared_private_dim: int = 256,
    private_dim: int = 64,
    private_gate: float = 0.25,
    shared_align_weight: float = 0.05,
    orth_weight: float = 0.01,
    use_vae_decoder: bool = False,
    vae_latent_dim: int = 128,
    vae_hidden_dim: int = 512,
    vae_recon_weight: float = 1.0,
    vae_kl_weight: float = 1e-4,
    use_image_ot: bool = False,
    use_gene_ot: bool = False,
    ot_transport: str = "ot",
    ot_image_weight: float = 0.0,
    ot_gene_weight: float = 0.0,
    ot_sinkhorn_eps: float = 0.05,
    ot_sinkhorn_iters: int = 50,
    uot_marginal_weight: float = 1.0,
):
    choice = str(model_name).strip().lower()
    if choice == "clip":
        base_model = CLIPModel_CLIP(spot_embedding=spot_embedding_dim)
    elif choice == "vit":
        base_model = CLIPModel_ViT(spot_embedding=spot_embedding_dim)
    elif choice == "vit_l":
        base_model = CLIPModel_ViT_L(spot_embedding=spot_embedding_dim)
    elif choice == "resnet101":
        base_model = CLIPModel_resnet101(spot_embedding=spot_embedding_dim)
    elif choice == "resnet152":
        base_model = CLIPModel_resnet152(spot_embedding=spot_embedding_dim)
    elif choice == "uni" or resolved_model_name == UNI_MODEL_NAME:
        base_model = CLIPModel_UNI(spot_embedding=spot_embedding_dim, pretrained=pretrained, checkpoint_path=checkpoint_path)
    else:
        base_model = CLIPModel(spot_embedding=spot_embedding_dim, model_name=resolved_model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
    return PlatformConditionedModel(
        base_model,
        use_platform_token=use_platform_token,
        use_shared_private=use_shared_private,
        keep_contrastive_loss=keep_contrastive_loss,
        use_platform_image_encoder=use_platform_image_encoder,
        shared_private_dim=shared_private_dim,
        private_dim=private_dim,
        private_gate=private_gate,
        shared_align_weight=shared_align_weight,
        orth_weight=orth_weight,
        use_vae_decoder=use_vae_decoder,
        vae_latent_dim=vae_latent_dim,
        vae_hidden_dim=vae_hidden_dim,
        vae_recon_weight=vae_recon_weight,
        vae_kl_weight=vae_kl_weight,
        output_gene_dim=spot_embedding_dim,
        use_image_ot=use_image_ot,
        use_gene_ot=use_gene_ot,
        ot_transport=ot_transport,
        ot_image_weight=ot_image_weight,
        ot_gene_weight=ot_gene_weight,
        ot_sinkhorn_eps=ot_sinkhorn_eps,
        ot_sinkhorn_iters=ot_sinkhorn_iters,
        uot_marginal_weight=uot_marginal_weight,
    )


def resolve_model_name(model_name: str) -> str:
    normalized = str(model_name).strip()
    if normalized.startswith("hf_hub:"):
        normalized = "hf-hub:" + normalized[len("hf_hub:"):]
    return MODEL_NAME_ALIASES.get(normalized.lower(), normalized)


def maybe_subset(indices: np.ndarray, limit: int) -> np.ndarray:
    if not limit or limit <= 0 or len(indices) <= int(limit):
        return indices
    return indices[: int(limit)]


def build_visium_subsets(args, shared_gene_file: str):
    dataset = CLIPDataset(
        hest_data_dir=args.visium_hest_data_dir,
        sample_ids=args.visium_sample_ids,
        gene_file=shared_gene_file,
        max_spots_per_sample=args.max_spots_per_sample,
        is_train=False,
    )
    if args.visium_fold_manifest:
        fold_payload = load_fixed_fold_manifest(args.visium_fold_manifest, args.visium_fold_index)
        train_samples = fold_payload["train_samples"]
        test_samples = fold_payload["test_samples"]
        holdout_label = f"fold_{int(fold_payload['fold_index']):02d}"
    else:
        train_samples = [sample_id for sample_id in args.visium_sample_ids if sample_id != args.visium_heldout_sample]
        test_samples = [args.visium_heldout_sample]
        holdout_label = args.visium_heldout_sample

    train_sample_set = set(train_samples)
    test_sample_set = set(test_samples)
    train_indices = []
    test_indices = []
    for idx, entry in enumerate(dataset.entries):
        sample_id = entry["sample_id"]
        if sample_id in test_sample_set:
            test_indices.append(idx)
        elif sample_id in train_sample_set:
            train_indices.append(idx)
    train_indices = maybe_subset(np.asarray(train_indices, dtype=np.int64), args.max_visium_train_spots)
    test_indices = maybe_subset(np.asarray(test_indices, dtype=np.int64), args.max_visium_test_spots)
    if len(train_indices) == 0:
        raise RuntimeError(f"No train visium spots for split {holdout_label}")
    if len(test_indices) == 0:
        raise RuntimeError(f"No test visium spots for split {holdout_label}")
    return dataset, train_indices, test_indices, train_samples, test_samples, holdout_label


def build_xenium_datasets(args, shared_gene_file: str):
    shared_genes = [line.strip() for line in Path(shared_gene_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    pseudospot_dir = build_pseudospot_output_dir(args.pseudo_output_base_dir, args.xenium_sample_id, args.xenium_reference_visium_sample_id)
    train_dataset = XeniumPseudoSpotDataset(
        pseudospot_dir=pseudospot_dir,
        split='train',
        test_fold=args.xenium_test_fold,
        num_position_folds=args.xenium_num_position_folds,
        encoder_target_gene_names=shared_genes,
        encoder_use_raw_counts=False,
        max_spots=args.max_xenium_train_spots,
        include_image=True,
        augment=True,
        image_size=args.image_size,
    )
    train_eval_dataset = XeniumPseudoSpotDataset(
        pseudospot_dir=pseudospot_dir,
        split='train',
        test_fold=args.xenium_test_fold,
        num_position_folds=args.xenium_num_position_folds,
        encoder_target_gene_names=shared_genes,
        encoder_use_raw_counts=False,
        max_spots=args.max_xenium_train_spots,
        include_image=True,
        augment=False,
        image_size=args.image_size,
    )
    test_dataset = XeniumPseudoSpotDataset(
        pseudospot_dir=pseudospot_dir,
        split='test',
        test_fold=args.xenium_test_fold,
        num_position_folds=args.xenium_num_position_folds,
        encoder_target_gene_names=shared_genes,
        encoder_use_raw_counts=False,
        max_spots=args.max_xenium_test_spots,
        include_image=True,
        augment=False,
        image_size=args.image_size,
    )
    return train_dataset, train_eval_dataset, test_dataset, shared_genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stage 2 minimal naive joint BRCA training')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--visium-hest-data-dir', default='/data/yujk/hovernet2feature/HEST/hest_data')
    parser.add_argument('--xenium-hest-data-dir', default='/data/yujk/hovernet2feature/HEST/hest_data_Xenium')
    parser.add_argument('--pseudo-output-base-dir', default='/data/yujk/GLIP/processed/pseudospots')
    parser.add_argument('--visium-sample-ids', default=','.join(DEFAULT_VISIUM_SAMPLE_IDS))
    parser.add_argument('--visium-heldout-sample', default='SPA119')
    parser.add_argument('--visium-fold-manifest', default='')
    parser.add_argument('--visium-fold-index', type=int, default=-1)
    parser.add_argument('--xenium-sample-id', default='NCBI784')
    parser.add_argument('--xenium-reference-visium-sample-id', default='SPA124')
    parser.add_argument('--shared-gene-file', default=DEFAULT_SHARED_GENE_FILE)
    parser.add_argument('--xenium-test-fold', type=int, default=4)
    parser.add_argument('--xenium-num-position-folds', type=int, default=5)
    parser.add_argument('--model', default='uni')
    parser.add_argument('--pretrained', default='false')
    parser.add_argument('--image-encoder-checkpoint', default='/data/yujk/UNI2-h/pytorch_model.bin')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--retrieval-chunk-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--max-spots-per-sample', type=int, default=0)
    parser.add_argument('--max-visium-train-spots', type=int, default=0)
    parser.add_argument('--max-visium-test-spots', type=int, default=0)
    parser.add_argument('--max-xenium-train-spots', type=int, default=0)
    parser.add_argument('--max-xenium-test-spots', type=int, default=0)
    parser.add_argument('--eval-bank-mode', choices=['target', 'joint'], default='target')
    parser.add_argument('--module-naive-joint', default='true')
    parser.add_argument('--module-platform-token', default='false')
    parser.add_argument('--module-shared-private', default='false')
    parser.add_argument('--keep-contrastive-loss', default='false')
    parser.add_argument('--module-platform-image-encoder', default='false')
    parser.add_argument('--shared-private-dim', type=int, default=256)
    parser.add_argument('--private-dim', type=int, default=64)
    parser.add_argument('--private-gate', type=float, default=0.25)
    parser.add_argument('--shared-align-weight', type=float, default=0.05)
    parser.add_argument('--orth-weight', type=float, default=0.01)
    parser.add_argument('--module-vae-decoder', default='false')
    parser.add_argument('--vae-latent-dim', type=int, default=128)
    parser.add_argument('--vae-hidden-dim', type=int, default=512)
    parser.add_argument('--vae-recon-weight', type=float, default=1.0)
    parser.add_argument('--vae-kl-weight', type=float, default=1e-4)
    parser.add_argument('--module-ot', default='false')
    parser.add_argument('--module-image-ot', default='')
    parser.add_argument('--module-gene-ot', default='false')
    parser.add_argument('--ot-transport', choices=['ot', 'uot'], default='ot')
    parser.add_argument('--ot-image-weight', type=float, default=0.05)
    parser.add_argument('--ot-gene-weight', type=float, default=0.05)
    parser.add_argument('--ot-sinkhorn-eps', type=float, default=0.05)
    parser.add_argument('--ot-sinkhorn-iters', type=int, default=50)
    parser.add_argument('--uot-marginal-weight', type=float, default=1.0)
    parser.add_argument('--module-gene-completion', default='false')
    parser.add_argument('--module-cell-refine', default='false')
    parser.add_argument('--note', default='Stage2 minimal naive joint training on pooled Visium + Xenium pseudo-spots in a shared 227-gene space.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.visium_sample_ids = parse_sample_ids(args.visium_sample_ids) or list(DEFAULT_VISIUM_SAMPLE_IDS)
    args.visium_fold_manifest = os.path.abspath(os.path.expanduser(args.visium_fold_manifest.strip())) if args.visium_fold_manifest else ''
    args.pretrained = parse_bool(args.pretrained)
    args.resolved_model_name = resolve_model_name(args.model)
    args.module_naive_joint = parse_bool(args.module_naive_joint)
    args.module_platform_token = parse_bool(args.module_platform_token)
    args.module_shared_private = parse_bool(args.module_shared_private)
    args.keep_contrastive_loss = parse_bool(args.keep_contrastive_loss)
    args.module_platform_image_encoder = parse_bool(args.module_platform_image_encoder)
    args.module_vae_decoder = parse_bool(args.module_vae_decoder)
    args.module_ot = parse_bool(args.module_ot)
    args.module_image_ot = parse_bool(args.module_image_ot) if str(args.module_image_ot).strip() else args.module_ot
    args.module_gene_ot = parse_bool(args.module_gene_ot)
    args.ot_transport = str(args.ot_transport).strip().lower()
    args.module_gene_completion = parse_bool(args.module_gene_completion)
    args.module_cell_refine = parse_bool(args.module_cell_refine)

    if not args.module_naive_joint:
        raise ValueError("This script is the Stage2 naive joint implementation; --module-naive-joint must remain true.")
    if args.module_shared_private and args.module_vae_decoder:
        raise ValueError("--module-shared-private and --module-vae-decoder are mutually exclusive.")
    if args.module_vae_decoder and args.module_gene_ot and not args.keep_contrastive_loss:
        raise ValueError("--module-gene-ot is not supported with --module-vae-decoder because the decoder path has no trained gene retrieval embedding.")
    if args.keep_contrastive_loss and not args.module_vae_decoder:
        raise ValueError("--keep-contrastive-loss requires --module-vae-decoder true.")
    unsupported_modules = {
        'platform_token': args.module_platform_token,
        'gene_completion': args.module_gene_completion,
        'cell_refine': args.module_cell_refine,
    }
    enabled_unsupported = [name for name, enabled in unsupported_modules.items() if enabled and name != 'platform_token']
    if enabled_unsupported:
        raise NotImplementedError(
            "The following module toggles are reserved for later stages and are not implemented in Stage2: "
            + ", ".join(enabled_unsupported)
        )

    os.makedirs(args.run_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')

    visium_dataset, vis_train_indices, vis_test_indices, visium_train_samples, visium_test_samples, visium_holdout_label = build_visium_subsets(args, args.shared_gene_file)
    x_train_ds, x_train_eval_ds, x_test_ds, shared_genes = build_xenium_datasets(args, args.shared_gene_file)

    vis_train = WrappedVisiumDataset(Subset(visium_dataset, vis_train_indices.tolist()), source_name='visium_train', platform_id=0)
    vis_train_eval = WrappedVisiumDataset(Subset(visium_dataset, vis_train_indices.tolist()), source_name='visium_train', platform_id=0)
    vis_test = WrappedVisiumDataset(Subset(visium_dataset, vis_test_indices.tolist()), source_name=f'visium_test_{visium_holdout_label}', platform_id=0)
    xen_train = WrappedXeniumPseudoSpotDataset(x_train_ds, source_name='xenium_train', platform_id=1)
    xen_train_eval = WrappedXeniumPseudoSpotDataset(x_train_eval_ds, source_name='xenium_train', platform_id=1)
    xen_test = WrappedXeniumPseudoSpotDataset(x_test_ds, source_name=f'xenium_test_{args.xenium_sample_id}', platform_id=1)

    combined_train = ConcatDataset([vis_train, xen_train])
    combined_train_eval = ConcatDataset([vis_train_eval, xen_train_eval])
    vis_train_eval_loader = create_loader(vis_train_eval, args.eval_batch_size, args.num_workers, shuffle=False)
    xen_train_eval_loader = create_loader(xen_train_eval, args.eval_batch_size, args.num_workers, shuffle=False)
    vis_test_loader = create_loader(vis_test, args.eval_batch_size, args.num_workers, shuffle=False)
    xen_test_loader = create_loader(xen_test, args.eval_batch_size, args.num_workers, shuffle=False)
    train_loader = create_loader(combined_train, args.batch_size, args.num_workers, shuffle=True)
    train_eval_loader = create_loader(combined_train_eval, args.eval_batch_size, args.num_workers, shuffle=False)

    model = build_model(
        args.model,
        args.resolved_model_name,
        len(shared_genes),
        args.pretrained,
        args.image_encoder_checkpoint,
        use_platform_token=args.module_platform_token,
        use_shared_private=args.module_shared_private,
        keep_contrastive_loss=args.keep_contrastive_loss,
        use_platform_image_encoder=args.module_platform_image_encoder,
        shared_private_dim=args.shared_private_dim,
        private_dim=args.private_dim,
        private_gate=args.private_gate,
        shared_align_weight=args.shared_align_weight,
        orth_weight=args.orth_weight,
        use_vae_decoder=args.module_vae_decoder,
        vae_latent_dim=args.vae_latent_dim,
        vae_hidden_dim=args.vae_hidden_dim,
        vae_recon_weight=args.vae_recon_weight,
        vae_kl_weight=args.vae_kl_weight,
        use_image_ot=args.module_image_ot,
        use_gene_ot=args.module_gene_ot,
        ot_transport=args.ot_transport,
        ot_image_weight=args.ot_image_weight,
        ot_gene_weight=args.ot_gene_weight,
        ot_sinkhorn_eps=args.ot_sinkhorn_eps,
        ot_sinkhorn_iters=args.ot_sinkhorn_iters,
        uot_marginal_weight=args.uot_marginal_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = RunConfig(
        visium_heldout_sample=visium_holdout_label,
        visium_sample_ids=args.visium_sample_ids,
        visium_train_samples=visium_train_samples,
        visium_test_samples=visium_test_samples,
        visium_fold_manifest=args.visium_fold_manifest,
        visium_fold_index=int(args.visium_fold_index),
        shared_gene_file=args.shared_gene_file,
        xenium_sample_id=args.xenium_sample_id,
        xenium_reference_visium_sample_id=args.xenium_reference_visium_sample_id,
        xenium_test_fold=args.xenium_test_fold,
        xenium_num_position_folds=args.xenium_num_position_folds,
        model=args.model,
        resolved_model_name=args.resolved_model_name,
        pretrained=args.pretrained,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        top_k=args.top_k,
        retrieval_chunk_size=args.retrieval_chunk_size,
        seed=args.seed,
        device=str(device),
        num_workers=args.num_workers,
        max_visium_train_spots=args.max_visium_train_spots,
        max_visium_test_spots=args.max_visium_test_spots,
        max_xenium_train_spots=args.max_xenium_train_spots,
        max_xenium_test_spots=args.max_xenium_test_spots,
        eval_bank_mode=args.eval_bank_mode,
        module_naive_joint=args.module_naive_joint,
        module_platform_token=args.module_platform_token,
        module_shared_private=args.module_shared_private,
        keep_contrastive_loss=args.keep_contrastive_loss,
        module_platform_image_encoder=args.module_platform_image_encoder,
        shared_private_dim=args.shared_private_dim,
        private_dim=args.private_dim,
        private_gate=args.private_gate,
        shared_align_weight=args.shared_align_weight,
        orth_weight=args.orth_weight,
        module_vae_decoder=args.module_vae_decoder,
        vae_latent_dim=args.vae_latent_dim,
        vae_hidden_dim=args.vae_hidden_dim,
        vae_recon_weight=args.vae_recon_weight,
        vae_kl_weight=args.vae_kl_weight,
        module_ot=args.module_ot,
        module_image_ot=args.module_image_ot,
        module_gene_ot=args.module_gene_ot,
        ot_transport=args.ot_transport,
        ot_image_weight=args.ot_image_weight,
        ot_gene_weight=args.ot_gene_weight,
        ot_sinkhorn_eps=args.ot_sinkhorn_eps,
        ot_sinkhorn_iters=args.ot_sinkhorn_iters,
        uot_marginal_weight=args.uot_marginal_weight,
        module_gene_completion=args.module_gene_completion,
        module_cell_refine=args.module_cell_refine,
        note=args.note,
    )
    save_json(asdict(config), os.path.join(args.run_dir, 'joint_config.json'))
    save_json({
        'manual_baseline_override': True,
        'baseline_source': 'user-provided manual single-platform results',
        'shared_gene_count': len(shared_genes),
        'shared_gene_file': os.path.abspath(args.shared_gene_file),
        'visium_holdout_label': visium_holdout_label,
        'visium_fold_manifest': args.visium_fold_manifest,
        'visium_fold_index': int(args.visium_fold_index),
        'visium_train_samples': visium_train_samples,
        'visium_test_samples': visium_test_samples,
        'visium_train_spots': len(vis_train),
        'visium_test_spots': len(vis_test),
        'xenium_train_spots': len(xen_train),
        'xenium_test_spots': len(xen_test),
        'eval_bank_mode': args.eval_bank_mode,
        'module_toggles': {
            'naive_joint': args.module_naive_joint,
            'platform_token': args.module_platform_token,
            'shared_private': args.module_shared_private,
            'keep_contrastive_loss': args.keep_contrastive_loss,
            'platform_image_encoder': args.module_platform_image_encoder,
            'shared_private_dim': args.shared_private_dim,
            'private_dim': args.private_dim,
            'private_gate': args.private_gate,
            'shared_align_weight': args.shared_align_weight,
            'orth_weight': args.orth_weight,
            'vae_decoder': args.module_vae_decoder,
            'vae_latent_dim': args.vae_latent_dim,
            'vae_hidden_dim': args.vae_hidden_dim,
            'vae_recon_weight': args.vae_recon_weight,
            'vae_kl_weight': args.vae_kl_weight,
            'ot': args.module_ot,
            'image_ot': args.module_image_ot,
            'gene_ot': args.module_gene_ot,
            'ot_transport': args.ot_transport,
            'ot_image_weight': args.ot_image_weight,
            'ot_gene_weight': args.ot_gene_weight,
            'ot_sinkhorn_eps': args.ot_sinkhorn_eps,
            'ot_sinkhorn_iters': args.ot_sinkhorn_iters,
            'uot_marginal_weight': args.uot_marginal_weight,
            'gene_completion': args.module_gene_completion,
            'cell_refine': args.module_cell_refine,
        },
        'no_extra_datasets_used': True,
    }, os.path.join(args.run_dir, 'split_manifest.json'))

    history = []
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_stats = train_epoch(model, train_loader, optimizer, device)
        visium_val = eval_loss(model, vis_test_loader, device, 'visium_val_loss')
        xenium_val = eval_loss(model, xen_test_loader, device, 'xenium_val_loss')
        combined_val = float((visium_val.avg + xenium_val.avg) / 2.0)
        epoch_record = {
            'epoch': epoch,
            'joint_train_loss': float(train_stats['total_loss']),
            'joint_train_main_loss': float(train_stats['main_loss']),
            'joint_train_image_ot_loss': float(train_stats['image_ot_loss']),
            'joint_train_gene_ot_loss': float(train_stats['gene_ot_loss']),
            'joint_train_shared_image_align_loss': float(train_stats['shared_image_align_loss']),
            'joint_train_shared_gene_align_loss': float(train_stats['shared_gene_align_loss']),
            'joint_train_orth_loss': float(train_stats['orth_loss']),
            'joint_train_vae_recon_loss': float(train_stats['vae_recon_loss']),
            'joint_train_vae_kl_loss': float(train_stats['vae_kl_loss']),
            'joint_train_image_ot_active_fraction': float(train_stats['image_ot_active_fraction']),
            'joint_train_gene_ot_active_fraction': float(train_stats['gene_ot_active_fraction']),
            'joint_train_shared_image_align_active_fraction': float(train_stats['shared_image_align_active_fraction']),
            'joint_train_shared_gene_align_active_fraction': float(train_stats['shared_gene_align_active_fraction']),
            'joint_train_image_ot_pairs_mean': float(train_stats['image_ot_pairs_mean']),
            'joint_train_gene_ot_pairs_mean': float(train_stats['gene_ot_pairs_mean']),
            'joint_train_shared_image_align_pairs_mean': float(train_stats['shared_image_align_pairs_mean']),
            'joint_train_shared_gene_align_pairs_mean': float(train_stats['shared_gene_align_pairs_mean']),
            'visium_val_loss': float(visium_val.avg),
            'xenium_val_loss': float(xenium_val.avg),
            'combined_val_loss': combined_val,
        }
        history.append(epoch_record)
        save_json({'history': history}, os.path.join(args.run_dir, 'history.json'))
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'history': history, 'config': asdict(config)}, os.path.join(args.run_dir, 'last.pt'))
        if combined_val < best_val:
            best_val = combined_val
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'history': history, 'config': asdict(config)}, os.path.join(args.run_dir, 'best.pt'))
            print(f'Saved best checkpoint with combined val loss {best_val:.6f}')

    best_payload = torch.load(os.path.join(args.run_dir, 'best.pt'), map_location=device)
    model.load_state_dict(best_payload['model_state_dict'])

    decoder_vis_metrics = None
    decoder_xen_metrics = None
    if args.module_vae_decoder and not args.keep_contrastive_loss:
        visium_bank = {"embeddings": torch.empty((0, 0))}
        xenium_bank = {"embeddings": torch.empty((0, 0))}
        joint_bank = {"embeddings": torch.empty((0, 0))}
        vis_queries = predict_expression_from_vae_decoder(model, vis_test_loader, device)
        xen_queries = predict_expression_from_vae_decoder(model, xen_test_loader, device)
        vis_query_count = int(vis_queries["predictions"].shape[0])
        xen_query_count = int(xen_queries["predictions"].shape[0])
        vis_metrics = compute_metrics(vis_queries["predictions"], vis_queries["targets"])
        xen_metrics = compute_metrics(xen_queries["predictions"], xen_queries["targets"])
        vis_joint_bank_metrics = dict(vis_metrics)
        xen_joint_bank_metrics = dict(xen_metrics)
    else:
        visium_bank = collect_spot_bank(model, vis_train_eval_loader, device)
        xenium_bank = collect_spot_bank(model, xen_train_eval_loader, device)
        joint_bank = collect_spot_bank(model, train_eval_loader, device)
        vis_queries = collect_image_queries(model, vis_test_loader, device)
        xen_queries = collect_image_queries(model, xen_test_loader, device)
        vis_query_count = int(vis_queries['embeddings'].shape[0])
        xen_query_count = int(xen_queries['embeddings'].shape[0])

        if args.eval_bank_mode == 'target':
            visium_primary_bank = visium_bank
            xenium_primary_bank = xenium_bank
        elif args.eval_bank_mode == 'joint':
            visium_primary_bank = joint_bank
            xenium_primary_bank = joint_bank
        else:
            raise ValueError(f"Unsupported eval bank mode: {args.eval_bank_mode}")

        vis_predictions = predict_expression_from_retrieval(visium_primary_bank, vis_queries, top_k=args.top_k, chunk_size=args.retrieval_chunk_size)
        xen_predictions = predict_expression_from_retrieval(xenium_primary_bank, xen_queries, top_k=args.top_k, chunk_size=args.retrieval_chunk_size)
        vis_predictions_joint_bank = predict_expression_from_retrieval(joint_bank, vis_queries, top_k=args.top_k, chunk_size=args.retrieval_chunk_size)
        xen_predictions_joint_bank = predict_expression_from_retrieval(joint_bank, xen_queries, top_k=args.top_k, chunk_size=args.retrieval_chunk_size)

        vis_metrics = compute_metrics(vis_predictions, vis_queries['expressions'].numpy())
        xen_metrics = compute_metrics(xen_predictions, xen_queries['expressions'].numpy())
        vis_joint_bank_metrics = compute_metrics(vis_predictions_joint_bank, vis_queries['expressions'].numpy())
        xen_joint_bank_metrics = compute_metrics(xen_predictions_joint_bank, xen_queries['expressions'].numpy())
        if args.module_vae_decoder:
            decoder_vis_queries = predict_expression_from_vae_decoder(model, vis_test_loader, device)
            decoder_xen_queries = predict_expression_from_vae_decoder(model, xen_test_loader, device)
            decoder_vis_metrics = compute_metrics(decoder_vis_queries["predictions"], decoder_vis_queries["targets"])
            decoder_xen_metrics = compute_metrics(decoder_xen_queries["predictions"], decoder_xen_queries["targets"])
            decoder_vis_metrics["top_k"] = 0
            decoder_xen_metrics["top_k"] = 0
            decoder_vis_metrics["heldout_sample"] = visium_holdout_label
            decoder_vis_metrics["heldout_samples"] = visium_test_samples
            decoder_xen_metrics["xenium_sample_id"] = args.xenium_sample_id
    vis_metrics['top_k'] = int(args.top_k)
    xen_metrics['top_k'] = int(args.top_k)
    vis_metrics['heldout_sample'] = visium_holdout_label
    vis_metrics['heldout_samples'] = visium_test_samples
    xen_metrics['xenium_sample_id'] = args.xenium_sample_id
    vis_joint_bank_metrics['top_k'] = int(args.top_k)
    xen_joint_bank_metrics['top_k'] = int(args.top_k)
    vis_joint_bank_metrics['heldout_sample'] = visium_holdout_label
    vis_joint_bank_metrics['heldout_samples'] = visium_test_samples
    xen_joint_bank_metrics['xenium_sample_id'] = args.xenium_sample_id

    summary = {
        'history': history,
        'shared_gene_count': len(shared_genes),
        'shared_genes_preview': shared_genes[:50],
        'best_epoch': int(best_payload['epoch']),
        'visium_bank_size': int(visium_bank['embeddings'].shape[0]),
        'xenium_bank_size': int(xenium_bank['embeddings'].shape[0]),
        'joint_bank_size': int(joint_bank['embeddings'].shape[0]),
        'eval_bank_mode': args.eval_bank_mode,
        'prediction_mode': 'hybrid_retrieval_with_vae_decoder' if args.keep_contrastive_loss else ('vae_decoder' if args.module_vae_decoder else 'retrieval'),
        'visium_test_metrics': vis_metrics,
        'xenium_test_metrics': xen_metrics,
        'visium_test_metrics_joint_bank': vis_joint_bank_metrics,
        'xenium_test_metrics_joint_bank': xen_joint_bank_metrics,
        'visium_test_metrics_decoder': decoder_vis_metrics,
        'xenium_test_metrics_decoder': decoder_xen_metrics,
        'visium_test_queries': vis_query_count,
        'xenium_test_queries': xen_query_count,
        'manual_baseline_override': True,
        'no_extra_datasets_used': True,
    }
    save_json(summary, os.path.join(args.run_dir, 'metrics.json'))

    rows = [
        {'target': 'visium', **vis_metrics},
        {'target': 'xenium_pseudospot', **xen_metrics},
        {'target': 'visium_joint_bank_diag', **vis_joint_bank_metrics},
        {'target': 'xenium_joint_bank_diag', **xen_joint_bank_metrics},
    ]
    if decoder_vis_metrics is not None and decoder_xen_metrics is not None:
        rows.extend([
            {'target': 'visium_decoder_diag', **decoder_vis_metrics},
            {'target': 'xenium_pseudospot_decoder_diag', **decoder_xen_metrics},
        ])
    csv_path = os.path.join(args.run_dir, 'metrics_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)

    print('Saved outputs to', args.run_dir)
    print('Visium test metrics:', vis_metrics)
    print('Xenium test metrics:', xen_metrics)


if __name__ == '__main__':
    main()
