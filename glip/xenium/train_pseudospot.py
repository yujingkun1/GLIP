#!/usr/bin/env python3
"""Train GLIP on Xenium-derived pseudo-spots."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from . import config as CFG
from .data import load_gene_names_from_tsv
from .model import ContrastiveImageGeneModel, resolve_image_model_name
from .pseudospot import (
    XeniumPseudoSpotDataset,
    build_pseudospot_output_dir,
    build_pseudospot_paths,
    prepare_pseudospot_dataset,
)
from glip.utils import (
    compute_pearson_metrics,
    get_lr,
    parse_bool,
    sample_indices,
    save_json,
    seed_everything,
)
from .train import configure_hf_hub, train_epoch


def create_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
        persistent_workers=num_workers > 0,
    )


def resolve_scfoundation_gene_list_path(repo_dir: str) -> str:
    candidates = [
        os.path.join(repo_dir, "OS_scRNA_gene_index.19264.tsv"),
        os.path.join(repo_dir, "model", "OS_scRNA_gene_index.19264.tsv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Unable to find scFoundation reference gene list under {repo_dir}")


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def collect_embeddings(
    model: ContrastiveImageGeneModel,
    loader: DataLoader,
    device: torch.device,
    *,
    encode_images: bool,
) -> Dict[str, torch.Tensor]:
    model.eval()
    embeddings: List[torch.Tensor] = []
    expressions: List[torch.Tensor] = []
    spot_ids: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            if encode_images:
                images = batch["image"].to(device, non_blocking=True)
                batch_embeddings = model.encode_images(images)
            else:
                expressions_batch = batch.get("encoder_expression", batch["expression"]).to(device, non_blocking=True)
                batch_embeddings = model.encode_genes(expressions_batch)
            embeddings.append(batch_embeddings.detach().cpu())
            expressions.append(batch["expression"].detach().cpu())
            spot_ids.append(torch.as_tensor(batch["cell_id"], dtype=torch.int64))

    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "expressions": torch.cat(expressions, dim=0),
        "spot_ids": torch.cat(spot_ids, dim=0),
    }


def predict_expression_from_retrieval(
    *,
    bank: Dict[str, torch.Tensor],
    queries: Dict[str, torch.Tensor],
    top_k: int,
    chunk_size: int,
    exclude_self: bool = False,
) -> np.ndarray:
    bank_embeddings = F.normalize(bank["embeddings"].float(), dim=1)
    bank_expressions = bank["expressions"].float()
    query_embeddings = F.normalize(queries["embeddings"].float(), dim=1)
    bank_spot_ids = bank["spot_ids"].cpu().numpy().astype(np.int64, copy=False)
    query_spot_ids = queries["spot_ids"].cpu().numpy().astype(np.int64, copy=False)

    effective_top_k = max(1, min(int(top_k), bank_embeddings.size(0)))
    bank_id_to_index = {int(spot_id): idx for idx, spot_id in enumerate(bank_spot_ids)} if exclude_self else {}
    predictions: List[torch.Tensor] = []

    for start in range(0, query_embeddings.size(0), max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), query_embeddings.size(0))
        query_chunk = query_embeddings[start:end]
        similarity = query_chunk @ bank_embeddings.T

        if exclude_self:
            for local_idx, spot_id in enumerate(query_spot_ids[start:end]):
                bank_idx = bank_id_to_index.get(int(spot_id))
                if bank_idx is not None:
                    similarity[local_idx, bank_idx] = -float("inf")

        top_values, top_indices = similarity.topk(effective_top_k, dim=1)
        matched = bank_expressions.index_select(0, top_indices.reshape(-1)).view(query_chunk.size(0), effective_top_k, -1)
        if effective_top_k == 1:
            predictions.append(matched[:, 0, :].cpu())
        else:
            weights = torch.softmax(top_values, dim=1).unsqueeze(-1)
            predictions.append((matched * weights).sum(dim=1).cpu())
    return torch.cat(predictions, dim=0).numpy()


def build_eval_subset(dataset, max_spots: int, seed: int):
    if max_spots <= 0 or len(dataset) <= max_spots:
        return dataset, len(dataset)
    selected = sample_indices(total_size=len(dataset), max_items=max_spots, seed=seed)
    return Subset(dataset, selected.tolist()), int(selected.shape[0])


def evaluate_retrieval(
    *,
    model: ContrastiveImageGeneModel,
    bank_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
    top_k: int,
    chunk_size: int,
    exclude_self: bool = False,
) -> Dict[str, float]:
    bank = collect_embeddings(model, bank_loader, device, encode_images=False)
    queries = collect_embeddings(model, query_loader, device, encode_images=True)
    predictions = predict_expression_from_retrieval(
        bank=bank,
        queries=queries,
        top_k=top_k,
        chunk_size=chunk_size,
        exclude_self=exclude_self,
    )
    targets = queries["expressions"].numpy()
    metrics = compute_pearson_metrics(predictions, targets, entity_label="spot")
    metrics["top_k"] = int(top_k)
    metrics["num_queries"] = int(targets.shape[0])
    metrics["num_bank_spots"] = int(bank["expressions"].shape[0])
    metrics["exclude_self"] = bool(exclude_self)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GLIP on Xenium-derived pseudo-spots")
    parser.add_argument(
        "--xenium-hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data_Xenium",
        help="HEST Xenium root directory",
    )
    parser.add_argument(
        "--visium-hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data",
        help="HEST Visium/ST root directory",
    )
    parser.add_argument("--processed-dir", default="/data/yujk/GLIP/processed", help="Processed Xenium cell cache directory")
    parser.add_argument(
        "--pseudo-output-base-dir",
        default="/data/yujk/GLIP/processed/pseudospots",
        help="Base directory where pseudo-spot folders are created",
    )
    parser.add_argument("--sample-id", default="NCBI784", help="Xenium sample id")
    parser.add_argument(
        "--reference-visium-sample-id",
        default="SPA124",
        help="Reference Visium/ST sample used to define pseudo-spot size",
    )
    parser.add_argument("--run-dir", default="/data/yujk/GLIP/runs_xenium/ncbi784_pseudospot", help="Training output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--lr", type=float, default=CFG.LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=CFG.WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--temperature", type=float, default=CFG.TEMPERATURE, help="Contrastive temperature")
    parser.add_argument("--model", default=CFG.MODEL_NAME, help="Image encoder backbone, e.g. resnet50 or uni")
    parser.add_argument("--pretrained", default="true", help="Use pretrained image encoder weights when available")
    parser.add_argument(
        "--image-encoder-checkpoint",
        default=CFG.IMAGE_ENCODER_CHECKPOINT,
        help="Optional local image encoder checkpoint path",
    )
    parser.add_argument("--hf-endpoint", default="", help="Optional Hugging Face Hub endpoint")
    parser.add_argument("--hf-hub-download-timeout", type=int, default=0, help="Optional HF Hub download timeout")
    parser.add_argument("--hf-hub-etag-timeout", type=int, default=0, help="Optional HF Hub metadata timeout")
    parser.add_argument("--device", default="", help="Torch device, empty means auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--crop-size-um",
        type=float,
        default=0.0,
        help="Pseudo-spot crop width in microns, 0 means use the cached default based on the reference spacing",
    )
    parser.add_argument("--image-size", type=int, default=CFG.IMAGE_SIZE, help="Model input size")
    parser.add_argument("--max-train-spots", type=int, default=0, help="Optional cap on train pseudo-spots")
    parser.add_argument("--max-test-spots", type=int, default=0, help="Optional cap on test pseudo-spots")
    parser.add_argument("--test-fold", type=int, default=4, help="Held-out x-position fold id")
    parser.add_argument("--num-position-folds", type=int, default=5, help="Number of contiguous x-position folds")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k neighbors for retrieval-based expression prediction")
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024, help="Chunk size for retrieval similarity")
    parser.add_argument(
        "--epoch-eval-max-spots",
        type=int,
        default=1024,
        help="Max train/test query spots used for per-epoch Pearson; 0 means full split",
    )
    parser.add_argument(
        "--final-train-eval-max-spots",
        type=int,
        default=0,
        help="Max train query spots for final train Pearson; 0 means full train split",
    )
    parser.add_argument(
        "--final-test-eval-max-spots",
        type=int,
        default=0,
        help="Max test query spots for final test Pearson; 0 means full test split",
    )
    parser.add_argument("--force-rebuild-pseudospots", action="store_true", help="Rebuild pseudo-spot cache")
    parser.add_argument("--min-cells-per-spot", type=int, default=3, help="Minimum number of cells per pseudo-spot")
    parser.add_argument("--grid-layout", default="auto", choices=["auto", "square", "hex"], help="Pseudo-spot grid layout")
    parser.add_argument(
        "--use-reference-inter-spot-dist",
        default="true",
        help="Whether pseudo-spot center spacing should follow the reference inter-spot distance",
    )
    parser.add_argument(
        "--gene-encoder",
        default=CFG.GENE_ENCODER,
        choices=["projection", "scfoundation"],
        help="Gene encoder backbone for pseudo-spot expression inputs",
    )
    parser.add_argument("--scfoundation-repo-dir", default=CFG.SCFOUNDATION_REPO_DIR, help="Local scFoundation repository path")
    parser.add_argument("--scfoundation-checkpoint", default=CFG.SCFOUNDATION_CHECKPOINT, help="Local scFoundation checkpoint path")
    parser.add_argument("--scfoundation-key", default=CFG.SCFOUNDATION_KEY, help="Top-level key inside the scFoundation checkpoint")
    parser.add_argument(
        "--scfoundation-pool-type",
        default=CFG.SCFOUNDATION_POOL_TYPE,
        choices=["all", "max"],
        help="Pooling method for scFoundation pseudo-spot embeddings",
    )
    parser.add_argument("--scfoundation-tgthighres", default=CFG.SCFOUNDATION_TGTHIGHRES, help="scFoundation T token setting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pretrained = parse_bool(args.pretrained)
    args.use_reference_inter_spot_dist = parse_bool(args.use_reference_inter_spot_dist)
    args.gene_encoder = args.gene_encoder.strip().lower()
    args.model = args.model.strip()
    args.resolved_model_name = resolve_image_model_name(args.model)
    args.hf_endpoint = args.hf_endpoint.strip()
    args.image_encoder_checkpoint = os.path.abspath(os.path.expanduser(args.image_encoder_checkpoint.strip())) if args.image_encoder_checkpoint else ""
    args.scfoundation_repo_dir = os.path.abspath(os.path.expanduser(args.scfoundation_repo_dir.strip()))
    args.scfoundation_checkpoint = os.path.abspath(os.path.expanduser(args.scfoundation_checkpoint.strip())) if args.scfoundation_checkpoint else ""
    args.scfoundation_key = args.scfoundation_key.strip()
    configure_hf_hub(args)

    if args.image_encoder_checkpoint and not os.path.exists(args.image_encoder_checkpoint):
        raise FileNotFoundError(f"Local image encoder checkpoint not found: {args.image_encoder_checkpoint}")
    if args.image_encoder_checkpoint and args.pretrained:
        print("Local image encoder checkpoint provided; disabling remote pretrained download.")
        args.pretrained = False

    if args.gene_encoder == "scfoundation":
        if not os.path.isdir(args.scfoundation_repo_dir):
            raise FileNotFoundError(f"scFoundation repo not found: {args.scfoundation_repo_dir}")
        if not os.path.exists(args.scfoundation_checkpoint):
            raise FileNotFoundError(f"scFoundation checkpoint not found: {args.scfoundation_checkpoint}")

    os.makedirs(args.run_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Image encoder: {args.resolved_model_name}")
    print(f"Gene encoder:  {args.gene_encoder}")

    pseudospot_dir = build_pseudospot_output_dir(
        args.pseudo_output_base_dir,
        args.sample_id,
        args.reference_visium_sample_id,
    )
    prepare_pseudospot_dataset(
        xenium_hest_data_dir=args.xenium_hest_data_dir,
        visium_hest_data_dir=args.visium_hest_data_dir,
        processed_dir=args.processed_dir,
        pseudo_output_dir=pseudospot_dir,
        xenium_sample_id=args.sample_id,
        reference_sample_id=args.reference_visium_sample_id,
        min_cells_per_spot=args.min_cells_per_spot,
        num_position_folds=args.num_position_folds,
        grid_layout=args.grid_layout,
        use_reference_inter_spot_dist=args.use_reference_inter_spot_dist,
        force_rebuild=args.force_rebuild_pseudospots,
    )
    pseudospot_paths = build_pseudospot_paths(pseudospot_dir)
    with open(pseudospot_paths.manifest_path, "r", encoding="utf-8") as handle:
        pseudospot_manifest = json.load(handle)

    if args.gene_encoder == "scfoundation":
        encoder_target_gene_names = load_gene_names_from_tsv(resolve_scfoundation_gene_list_path(args.scfoundation_repo_dir))
        encoder_use_raw_counts = True
    else:
        with open(pseudospot_paths.genes_path, "r", encoding="utf-8") as handle:
            encoder_target_gene_names = list(json.load(handle)["genes"])
        encoder_use_raw_counts = False

    train_dataset = XeniumPseudoSpotDataset(
        pseudospot_dir=pseudospot_dir,
        split="train",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size_um=args.crop_size_um,
        image_size=args.image_size,
        augment=True,
        max_spots=args.max_train_spots,
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )
    train_eval_dataset = XeniumPseudoSpotDataset(
        pseudospot_dir=pseudospot_dir,
        split="train",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size_um=args.crop_size_um,
        image_size=args.image_size,
        augment=False,
        max_spots=args.max_train_spots,
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )
    train_bank_dataset = XeniumPseudoSpotDataset(
        pseudospot_dir=pseudospot_dir,
        split="train",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size_um=args.crop_size_um,
        image_size=args.image_size,
        augment=False,
        include_image=False,
        max_spots=args.max_train_spots,
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )
    test_dataset = XeniumPseudoSpotDataset(
        pseudospot_dir=pseudospot_dir,
        split="test",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size_um=args.crop_size_um,
        image_size=args.image_size,
        augment=False,
        max_spots=args.max_test_spots,
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )

    print(f"Pseudo-spot cache: {pseudospot_dir}")
    print(f"Train pseudo-spots: {len(train_dataset)}")
    print(f"Test pseudo-spots:  {len(test_dataset)}")
    print(f"Gene dim:           {train_dataset.num_features}")
    print(f"Encoder dim:        {len(encoder_target_gene_names)}")

    train_loader = create_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    bank_loader = create_loader(train_bank_dataset, args.eval_batch_size, args.num_workers, shuffle=False)

    model = ContrastiveImageGeneModel(
        gene_dim=len(encoder_target_gene_names),
        model_name=args.resolved_model_name,
        pretrained=args.pretrained,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        temperature=args.temperature,
        gene_encoder=args.gene_encoder,
        scfoundation_repo_dir=args.scfoundation_repo_dir,
        scfoundation_checkpoint=args.scfoundation_checkpoint,
        scfoundation_key=args.scfoundation_key,
        scfoundation_pool_type=args.scfoundation_pool_type,
        scfoundation_tgthighres=args.scfoundation_tgthighres,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_train_loss = float("inf")

    split_manifest = {
        "pseudospot_dir": os.path.abspath(pseudospot_dir),
        "sample_id": args.sample_id,
        "reference_visium_sample_id": args.reference_visium_sample_id,
        "num_pseudospots": int(pseudospot_manifest["num_pseudospots"]),
        "train_spots": int(len(train_dataset)),
        "test_spots": int(len(test_dataset)),
        "test_fold": int(args.test_fold),
        "num_position_folds": int(args.num_position_folds),
        "crop_size_um": float(train_dataset.crop_size_um),
        "default_crop_size_um": float(pseudospot_manifest["default_crop_size_um"]),
    }
    save_json(split_manifest, os.path.join(args.run_dir, "split_manifest.json"))

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_epoch(model, train_loader, optimizer, device)

        epoch_train_subset, num_epoch_train_queries = build_eval_subset(
            train_eval_dataset,
            max_spots=args.epoch_eval_max_spots,
            seed=args.seed + epoch,
        )
        epoch_test_subset, num_epoch_test_queries = build_eval_subset(
            test_dataset,
            max_spots=args.epoch_eval_max_spots,
            seed=args.seed + 1000 + epoch,
        )
        epoch_train_loader = create_loader(epoch_train_subset, args.eval_batch_size, args.num_workers, shuffle=False)
        epoch_test_loader = create_loader(epoch_test_subset, args.eval_batch_size, args.num_workers, shuffle=False)

        train_metrics = evaluate_retrieval(
            model=model,
            bank_loader=bank_loader,
            query_loader=epoch_train_loader,
            device=device,
            top_k=args.top_k,
            chunk_size=args.retrieval_chunk_size,
            exclude_self=True,
        )
        test_metrics = evaluate_retrieval(
            model=model,
            bank_loader=bank_loader,
            query_loader=epoch_test_loader,
            device=device,
            top_k=args.top_k,
            chunk_size=args.retrieval_chunk_size,
            exclude_self=False,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_stats["total_loss"]),
            "train_main_loss": float(train_stats["main_loss"]),
            "train_lr": float(get_lr(optimizer)),
            "epoch_train_num_queries": int(num_epoch_train_queries),
            "epoch_test_num_queries": int(num_epoch_test_queries),
            "epoch_train_overall_pearson": float(train_metrics["overall_pearson"]),
            "epoch_train_mean_gene_pearson": float(train_metrics["mean_gene_pearson"]),
            "epoch_train_mean_spot_pearson": float(train_metrics["mean_spot_pearson"]),
            "epoch_test_overall_pearson": float(test_metrics["overall_pearson"]),
            "epoch_test_mean_gene_pearson": float(test_metrics["mean_gene_pearson"]),
            "epoch_test_mean_spot_pearson": float(test_metrics["mean_spot_pearson"]),
        }
        history.append(epoch_record)
        save_json({"history": history}, os.path.join(args.run_dir, "metrics.json"))

        print(
            "Train: loss={loss:.4f} overall={overall:.4f} mean_gene={gene:.4f} mean_spot={spot:.4f}".format(
                loss=epoch_record["train_loss"],
                overall=train_metrics["overall_pearson"],
                gene=train_metrics["mean_gene_pearson"],
                spot=train_metrics["mean_spot_pearson"],
            )
        )
        print(
            "Test:  overall={overall:.4f} mean_gene={gene:.4f} mean_spot={spot:.4f}".format(
                overall=test_metrics["overall_pearson"],
                gene=test_metrics["mean_gene_pearson"],
                spot=test_metrics["mean_spot_pearson"],
            )
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
            "pseudospot_manifest": pseudospot_manifest,
        }
        torch.save(checkpoint_payload, os.path.join(args.run_dir, "last.pt"))
        if epoch_record["train_loss"] < best_train_loss:
            best_train_loss = epoch_record["train_loss"]
            torch.save(checkpoint_payload, os.path.join(args.run_dir, "best_train_loss.pt"))

    final_train_subset, _ = build_eval_subset(
        train_eval_dataset,
        max_spots=args.final_train_eval_max_spots,
        seed=args.seed + 5000,
    )
    final_test_subset, _ = build_eval_subset(
        test_dataset,
        max_spots=args.final_test_eval_max_spots,
        seed=args.seed + 6000,
    )
    final_train_loader = create_loader(final_train_subset, args.eval_batch_size, args.num_workers, shuffle=False)
    final_test_loader = create_loader(final_test_subset, args.eval_batch_size, args.num_workers, shuffle=False)
    final_train_metrics = evaluate_retrieval(
        model=model,
        bank_loader=bank_loader,
        query_loader=final_train_loader,
        device=device,
        top_k=args.top_k,
        chunk_size=args.retrieval_chunk_size,
        exclude_self=True,
    )
    final_test_metrics = evaluate_retrieval(
        model=model,
        bank_loader=bank_loader,
        query_loader=final_test_loader,
        device=device,
        top_k=args.top_k,
        chunk_size=args.retrieval_chunk_size,
        exclude_self=False,
    )

    final_payload = {
        "history": history,
        "final_train_metrics": final_train_metrics,
        "final_test_metrics": final_test_metrics,
        "pseudospot_dir": os.path.abspath(pseudospot_dir),
        "pseudospot_manifest": pseudospot_manifest,
        "run_args": vars(args),
    }
    save_json(final_payload, os.path.join(args.run_dir, "metrics.json"))
    print("\nFinal train metrics:", final_train_metrics)
    print("Final test metrics: ", final_test_metrics)


if __name__ == "__main__":
    main()
