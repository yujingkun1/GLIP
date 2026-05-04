#!/usr/bin/env python3
"""Evaluate a trained joint VAE decoder on held-out Xenium single cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from glip.utils import compute_pearson_metrics, parse_bool, save_json, seed_everything
from glip.xenium.data import XeniumSingleCellDataset
from train_joint_brca_naive import build_model, resolve_model_name


DEFAULT_GENE_FILE = "/data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt"
DEFAULT_CHECKPOINT = "/data/yujk/BLEEP/checkpoints/resnet50_a1_0-14fe96d1.pth"


def load_gene_names_from_text_file(path: str) -> List[str]:
    genes: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            gene = raw_line.strip()
            if not gene or gene in seen:
                continue
            seen.add(gene)
            genes.append(gene)
    if not genes:
        raise RuntimeError(f"No genes found in gene file: {path}")
    return genes


def create_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def get_config_value(config: Dict, name: str, fallback):
    value = config.get(name, fallback)
    return fallback if value is None else value


def build_decoder_model_from_config(args: argparse.Namespace, checkpoint_config: Dict, gene_dim: int):
    model_name = args.model or str(get_config_value(checkpoint_config, "model", "resnet50"))
    resolved_model_name = resolve_model_name(model_name)
    pretrained = parse_bool(args.pretrained)
    image_encoder_checkpoint = args.image_encoder_checkpoint or str(
        get_config_value(checkpoint_config, "image_encoder_checkpoint", DEFAULT_CHECKPOINT)
    )

    return build_model(
        model_name,
        resolved_model_name,
        gene_dim,
        pretrained,
        image_encoder_checkpoint,
        use_platform_token=bool(get_config_value(checkpoint_config, "module_platform_token", False)),
        use_shared_private=False,
        shared_private_dim=int(get_config_value(checkpoint_config, "shared_private_dim", 256)),
        private_dim=int(get_config_value(checkpoint_config, "private_dim", 64)),
        private_gate=float(get_config_value(checkpoint_config, "private_gate", 0.25)),
        shared_align_weight=float(get_config_value(checkpoint_config, "shared_align_weight", 0.05)),
        orth_weight=float(get_config_value(checkpoint_config, "orth_weight", 0.01)),
        use_vae_decoder=True,
        vae_latent_dim=int(get_config_value(checkpoint_config, "vae_latent_dim", 128)),
        vae_hidden_dim=int(get_config_value(checkpoint_config, "vae_hidden_dim", 512)),
        vae_recon_weight=float(get_config_value(checkpoint_config, "vae_recon_weight", 1.0)),
        vae_kl_weight=float(get_config_value(checkpoint_config, "vae_kl_weight", 1e-4)),
        use_image_ot=bool(get_config_value(checkpoint_config, "module_image_ot", False)),
        use_gene_ot=False,
        ot_transport=str(get_config_value(checkpoint_config, "ot_transport", "ot")),
        ot_image_weight=float(get_config_value(checkpoint_config, "ot_image_weight", 0.05)),
        ot_gene_weight=float(get_config_value(checkpoint_config, "ot_gene_weight", 0.05)),
        ot_sinkhorn_eps=float(get_config_value(checkpoint_config, "ot_sinkhorn_eps", 0.05)),
        ot_sinkhorn_iters=int(get_config_value(checkpoint_config, "ot_sinkhorn_iters", 50)),
        uot_marginal_weight=float(get_config_value(checkpoint_config, "uot_marginal_weight", 1.0)),
    )


@torch.no_grad()
def evaluate_decoder(model, loader: DataLoader, device: torch.device) -> Dict:
    predictions = []
    targets = []
    cell_ids: List[int] = []
    fold_ids: List[int] = []

    model.eval()
    for batch in tqdm(loader, total=len(loader), desc="decoder_cell_eval"):
        images = batch["image"].to(device, non_blocking=True)
        platform_ids = torch.ones(images.size(0), dtype=torch.long, device=device)
        predicted = model.predict_expression_from_images(images, platform_ids=platform_ids)
        predictions.append(predicted.detach().cpu())
        targets.append(batch["encoder_expression"].detach().cpu())
        cell_ids.extend([int(cell_id) for cell_id in batch["cell_id"]])
        fold_ids.extend([int(fold_id) for fold_id in batch["fold_id"]])

    predictions_np = torch.cat(predictions, dim=0).numpy()
    targets_np = torch.cat(targets, dim=0).numpy()
    metrics = compute_pearson_metrics(predictions_np, targets_np, entity_label="cell")
    return {
        "metrics": metrics,
        "cell_ids": cell_ids,
        "fold_ids": fold_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a joint VAE decoder on Xenium single cells")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hest-data-dir", default="/data/yujk/hovernet2feature/HEST/hest_data_Xenium")
    parser.add_argument("--processed-dir", default="/data/yujk/GLIP/processed")
    parser.add_argument("--sample-id", default="NCBI784")
    parser.add_argument("--gene-file", default=DEFAULT_GENE_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--pretrained", default="false")
    parser.add_argument("--image-encoder-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--wsi-level", type=int, default=0)
    parser.add_argument("--test-fold", type=int, default=4)
    parser.add_argument("--num-position-folds", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-test-cells", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    seed_everything(args.seed)

    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    gene_names = load_gene_names_from_text_file(args.gene_file)
    requested_device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: requested {requested_device}, but CUDA is not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = payload.get("config", {})
    if not checkpoint_config:
        config_path = Path(checkpoint_path).with_name("joint_config.json")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                checkpoint_config = json.load(handle)

    model = build_decoder_model_from_config(args, checkpoint_config, len(gene_names)).to(device)
    model.load_state_dict(payload["model_state_dict"])

    test_dataset = XeniumSingleCellDataset(
        processed_dir=args.processed_dir,
        hest_data_dir=args.hest_data_dir,
        sample_id=args.sample_id,
        split="test",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size=args.crop_size,
        image_size=args.image_size,
        wsi_level=args.wsi_level,
        augment=False,
        max_cells=args.max_test_cells,
        encoder_target_gene_names=gene_names,
        encoder_use_raw_counts=False,
    )
    loader = create_loader(test_dataset, args.eval_batch_size, args.num_workers)
    result = evaluate_decoder(model, loader, device)
    metrics = result["metrics"]
    metrics["fold_idx"] = int(args.test_fold)
    metrics["xenium_sample_id"] = args.sample_id
    metrics["prediction_mode"] = "vae_decoder_cell_eval"
    metrics["checkpoint"] = checkpoint_path

    summary = {
        "sample_id": args.sample_id,
        "checkpoint": checkpoint_path,
        "device": str(device),
        "num_genes": len(gene_names),
        "test_fold": int(args.test_fold),
        "num_position_folds": int(args.num_position_folds),
        "prediction_mode": "vae_decoder_cell_eval",
        "final_test_metrics": metrics,
    }
    save_json(summary, os.path.join(args.run_dir, "metrics.json"))
    save_json(
        {
            "test_cells": result["cell_ids"],
            "test_fold_ids": result["fold_ids"],
            "num_final_test_queries": int(len(result["cell_ids"])),
            "gene_file": os.path.abspath(args.gene_file),
            "checkpoint": checkpoint_path,
        },
        os.path.join(args.run_dir, "split_manifest.json"),
    )
    print(f"Final test overall Pearson: {metrics['overall_pearson']:.4f}")
    print(f"Saved outputs to {args.run_dir}")


if __name__ == "__main__":
    main()
