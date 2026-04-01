#!/usr/bin/env python3
"""Train a BLEEP-style contrastive model at single-cell resolution on NCBI784."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from glip import config as CFG
from glip.data import (
    ScRNADataset,
    XeniumSingleCellDataset,
    build_processed_paths,
    load_gene_names_from_tsv,
    prepare_processed_dataset,
)
from glip.model import ContrastiveImageGeneModel, resolve_image_model_name
from glip.utils import (
    AvgMeter,
    compute_pearson_metrics,
    get_lr,
    parse_bool,
    sample_indices,
    save_json,
    seed_everything,
)


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


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def compute_xenium_scrna_knn_loss(
    xenium_embeddings: torch.Tensor,
    scrna_bank_embeddings: torch.Tensor,
    *,
    knn_percent: float,
    temperature: float,
) -> torch.Tensor:
    if scrna_bank_embeddings.numel() == 0:
        return xenium_embeddings.new_zeros(())

    normalized_xenium = F.normalize(xenium_embeddings.float(), dim=1)
    logits = (normalized_xenium @ scrna_bank_embeddings.T) / max(float(temperature), 1e-6)

    bank_size = int(scrna_bank_embeddings.size(0))
    knn_percent = min(max(float(knn_percent), 0.0), 100.0)
    knn_count = max(1, min(bank_size, int(math.ceil(bank_size * knn_percent / 100.0))))
    top_values, top_indices = logits.topk(knn_count, dim=1)
    target_weights = torch.softmax(top_values.detach(), dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_weights * log_probs.gather(1, top_indices)).sum(dim=1).mean()


def collect_scrna_embeddings(
    model: ContrastiveImageGeneModel,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="scrna_bank", leave=False):
            batch = move_batch_to_device(batch, device)
            batch_embeddings = model.encode_genes(batch["encoder_expression"])
            embeddings.append(batch_embeddings.detach().cpu())

    if not embeddings:
        return torch.empty((0, CFG.PROJECTION_DIM), dtype=torch.float32)
    return torch.cat(embeddings, dim=0)


def train_epoch(
    model: ContrastiveImageGeneModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scrna_bank_embeddings: Optional[torch.Tensor] = None,
    scrna_loss_weight: float = 0.0,
    scrna_knn_percent: float = 1.0,
    scrna_temperature: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss_meter = AvgMeter("train_loss")
    main_loss_meter = AvgMeter("main_loss")
    aux_loss_meter = AvgMeter("scrna_loss")
    progress = tqdm(loader, total=len(loader), desc="train", leave=False)

    normalized_scrna_bank = None
    if scrna_bank_embeddings is not None and scrna_bank_embeddings.numel() > 0 and scrna_loss_weight > 0:
        normalized_scrna_bank = F.normalize(scrna_bank_embeddings.to(device, non_blocking=True).float(), dim=1)

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        if normalized_scrna_bank is not None:
            main_loss, gene_embeddings = model.compute_image_gene_loss(batch, return_gene_embeddings=True)
            aux_loss = compute_xenium_scrna_knn_loss(
                gene_embeddings,
                normalized_scrna_bank,
                knn_percent=scrna_knn_percent,
                temperature=scrna_temperature,
            )
            loss = main_loss + float(scrna_loss_weight) * aux_loss
        else:
            main_loss = model(batch)
            aux_loss = main_loss.new_zeros(())
            loss = main_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = int(batch["image"].size(0))
        total_loss_meter.update(float(loss.item()), batch_size)
        main_loss_meter.update(float(main_loss.item()), batch_size)
        aux_loss_meter.update(float(aux_loss.item()), batch_size)
        progress.set_postfix(
            total=f"{total_loss_meter.avg:.4f}",
            main=f"{main_loss_meter.avg:.4f}",
            aux=f"{aux_loss_meter.avg:.4f}",
            lr=f"{get_lr(optimizer):.2e}",
        )

    return {
        "total_loss": float(total_loss_meter.avg),
        "main_loss": float(main_loss_meter.avg),
        "scrna_loss": float(aux_loss_meter.avg),
    }


def _collect_embeddings(
    model: ContrastiveImageGeneModel,
    loader: DataLoader,
    device: torch.device,
    *,
    encode_images: bool,
) -> Dict[str, torch.Tensor]:
    model.eval()
    embeddings: List[torch.Tensor] = []
    expressions: List[torch.Tensor] = []
    cell_ids: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="embed", leave=False):
            if encode_images:
                images = batch["image"].to(device, non_blocking=True)
                batch_embeddings = model.encode_images(images)
            else:
                expressions_batch = batch.get("encoder_expression", batch["expression"]).to(device, non_blocking=True)
                batch_embeddings = model.encode_genes(expressions_batch)
            embeddings.append(batch_embeddings.detach().cpu())
            expressions.append(batch["expression"].detach().cpu())
            cell_ids.append(torch.as_tensor(batch["cell_id"], dtype=torch.int64))

    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "expressions": torch.cat(expressions, dim=0),
        "cell_ids": torch.cat(cell_ids, dim=0),
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
    bank_cell_ids = bank["cell_ids"].cpu().numpy().astype(np.int64, copy=False)
    query_cell_ids = queries["cell_ids"].cpu().numpy().astype(np.int64, copy=False)

    if bank_embeddings.size(0) == 0:
        raise RuntimeError("Retrieval bank is empty.")

    effective_top_k = max(1, min(int(top_k), bank_embeddings.size(0)))
    predictions: List[torch.Tensor] = []
    bank_id_to_index = {int(cell_id): idx for idx, cell_id in enumerate(bank_cell_ids)} if exclude_self else {}

    for start in range(0, query_embeddings.size(0), max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), query_embeddings.size(0))
        query_chunk = query_embeddings[start:end]
        similarity = query_chunk @ bank_embeddings.T

        if exclude_self:
            for local_idx, cell_id in enumerate(query_cell_ids[start:end]):
                bank_idx = bank_id_to_index.get(int(cell_id))
                if bank_idx is not None:
                    similarity[local_idx, bank_idx] = -float("inf")

        top_values, top_indices = similarity.topk(effective_top_k, dim=1)
        matched = bank_expressions.index_select(0, top_indices.reshape(-1)).view(query_chunk.size(0), effective_top_k, -1)

        if effective_top_k == 1:
            chunk_prediction = matched[:, 0, :]
        else:
            weights = torch.softmax(top_values, dim=1).unsqueeze(-1)
            chunk_prediction = (matched * weights).sum(dim=1)
        predictions.append(chunk_prediction.cpu())

    return torch.cat(predictions, dim=0).numpy()


def build_eval_subset(dataset, max_cells: int, seed: int):
    if max_cells <= 0 or len(dataset) <= max_cells:
        return dataset, len(dataset)
    selected = sample_indices(total_size=len(dataset), max_items=max_cells, seed=seed)
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
    bank = _collect_embeddings(model, bank_loader, device, encode_images=False)
    queries = _collect_embeddings(model, query_loader, device, encode_images=True)
    predictions = predict_expression_from_retrieval(
        bank=bank,
        queries=queries,
        top_k=top_k,
        chunk_size=chunk_size,
        exclude_self=exclude_self,
    )
    targets = queries["expressions"].numpy()
    metrics = compute_pearson_metrics(predictions, targets, entity_label="cell")
    metrics["top_k"] = int(top_k)
    metrics["num_queries"] = int(targets.shape[0])
    metrics["num_bank_cells"] = int(bank["expressions"].shape[0])
    metrics["exclude_self"] = bool(exclude_self)
    return metrics


def configure_hf_hub(args: argparse.Namespace) -> None:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.hf_hub_download_timeout and args.hf_hub_download_timeout > 0:
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(int(args.hf_hub_download_timeout))

    if args.hf_hub_etag_timeout and args.hf_hub_etag_timeout > 0:
        os.environ["HF_HUB_ETAG_TIMEOUT"] = str(int(args.hf_hub_etag_timeout))


def resolve_scfoundation_gene_list_path(repo_dir: str) -> str:
    candidates = [
        os.path.join(repo_dir, "OS_scRNA_gene_index.19264.tsv"),
        os.path.join(repo_dir, "model", "OS_scRNA_gene_index.19264.tsv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Unable to find scFoundation reference gene list under {repo_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GLIP on NCBI784 single-cell Xenium data")
    parser.add_argument(
        "--hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data_Xenium",
        help="HEST Xenium root directory",
    )
    parser.add_argument("--processed-dir", default="/data/yujk/GLIP/processed", help="Processed cache directory")
    parser.add_argument("--sample-id", default="NCBI784", help="HEST Xenium sample id")
    parser.add_argument("--run-dir", default="/data/yujk/GLIP/runs_xenium/ncbi784_uni", help="Training output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
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
    parser.add_argument("--hf-endpoint", default="", help="Optional Hugging Face Hub endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--hf-hub-download-timeout", type=int, default=0, help="Optional HF Hub download timeout in seconds")
    parser.add_argument("--hf-hub-etag-timeout", type=int, default=0, help="Optional HF Hub metadata timeout in seconds")
    parser.add_argument("--device", default="", help="Torch device, empty means auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--crop-size", type=int, default=CFG.CROP_SIZE, help="Crop size in level-0 pixels")
    parser.add_argument("--image-size", type=int, default=CFG.IMAGE_SIZE, help="Model input size")
    parser.add_argument("--wsi-level", type=int, default=0, help="OpenSlide pyramid level used for reading")
    parser.add_argument("--max-train-cells", type=int, default=0, help="Optional cap on train cells, 0 means all")
    parser.add_argument("--max-test-cells", type=int, default=0, help="Optional cap on test cells, 0 means all")
    parser.add_argument("--test-fold", type=int, default=4, help="Held-out x-position fold id")
    parser.add_argument("--num-position-folds", type=int, default=5, help="Number of contiguous x-position folds")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k neighbors for retrieval-based expression prediction")
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024, help="Chunk size for retrieval similarity")
    parser.add_argument(
        "--epoch-eval-max-cells",
        type=int,
        default=2048,
        help="Max train/test query cells used for per-epoch Pearson; 0 means full split",
    )
    parser.add_argument(
        "--final-train-eval-max-cells",
        type=int,
        default=4096,
        help="Max train query cells for the final train Pearson; 0 means full train split",
    )
    parser.add_argument(
        "--final-test-eval-max-cells",
        type=int,
        default=0,
        help="Max test query cells for the final test Pearson; 0 means full test split",
    )
    parser.add_argument(
        "--remove-control-features",
        default="true",
        help="Remove BLANK / NegControl / antisense features during preprocessing",
    )
    parser.add_argument(
        "--nucleus-only",
        default="false",
        help="Only keep transcripts with overlaps_nucleus == 1 during preprocessing",
    )
    parser.add_argument(
        "--drop-zero-expression",
        default="true",
        help="Drop segmented cells with zero remaining transcripts during preprocessing",
    )
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild the processed cache")
    parser.add_argument(
        "--gene-encoder",
        default=CFG.GENE_ENCODER,
        choices=["projection", "scfoundation"],
        help="Gene encoder backbone used for Xenium and scRNA expression inputs",
    )
    parser.add_argument("--scfoundation-repo-dir", default=CFG.SCFOUNDATION_REPO_DIR, help="Local scFoundation repository path")
    parser.add_argument("--scfoundation-checkpoint", default=CFG.SCFOUNDATION_CHECKPOINT, help="Local scFoundation checkpoint path")
    parser.add_argument("--scfoundation-key", default=CFG.SCFOUNDATION_KEY, help="Top-level key inside the scFoundation checkpoint")
    parser.add_argument(
        "--scfoundation-pool-type",
        default=CFG.SCFOUNDATION_POOL_TYPE,
        choices=["all", "max"],
        help="Pooling method for scFoundation cell embeddings",
    )
    parser.add_argument("--scfoundation-tgthighres", default=CFG.SCFOUNDATION_TGTHIGHRES, help="scFoundation T token setting, e.g. t4 or a5")
    parser.add_argument("--scrna-data-path", default=CFG.SCRNA_DATA_PATH, help="Optional scRNA h5ad path for the auxiliary loss")
    parser.add_argument(
        "--use-scrna-loss",
        default=str(CFG.USE_SCRNA_LOSS).lower(),
        help="Whether to enable the scRNA auxiliary loss branch",
    )
    parser.add_argument("--scrna-loss-weight", type=float, default=CFG.SCRNA_LOSS_WEIGHT, help="Weight of the Xenium-scRNA auxiliary loss")
    parser.add_argument(
        "--scrna-knn-percent",
        type=float,
        default=CFG.SCRNA_KNN_PERCENT,
        help="Top percentage of scRNA bank neighbors used as positives; 0.1 means 0.1%% of the bank",
    )
    parser.add_argument("--scrna-temperature", type=float, default=CFG.TEMPERATURE, help="Temperature for the Xenium-scRNA auxiliary loss")
    parser.add_argument("--scrna-batch-size", type=int, default=256, help="Batch size for scRNA bank embedding refresh")
    parser.add_argument("--scrna-num-workers", type=int, default=-1, help="Worker count for scRNA loading; -1 reuses --num-workers")
    parser.add_argument("--scrna-max-cells", type=int, default=0, help="Optional cap on scRNA cells used in the auxiliary bank")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pretrained = parse_bool(args.pretrained)
    args.model = args.model.strip()
    args.gene_encoder = args.gene_encoder.strip().lower()
    args.resolved_model_name = resolve_image_model_name(args.model)
    args.hf_endpoint = args.hf_endpoint.strip()
    args.image_encoder_checkpoint = os.path.expanduser(args.image_encoder_checkpoint.strip()) if args.image_encoder_checkpoint else ""
    args.scfoundation_repo_dir = os.path.abspath(os.path.expanduser(args.scfoundation_repo_dir.strip()))
    args.scfoundation_checkpoint = os.path.abspath(os.path.expanduser(args.scfoundation_checkpoint.strip())) if args.scfoundation_checkpoint else ""
    args.scfoundation_key = args.scfoundation_key.strip()
    args.scrna_data_path = os.path.abspath(os.path.expanduser(args.scrna_data_path.strip())) if args.scrna_data_path else ""
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
        if not os.access(args.scfoundation_checkpoint, os.R_OK):
            raise PermissionError(f"scFoundation checkpoint is not readable: {args.scfoundation_checkpoint}")

    os.makedirs(args.run_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Image encoder: {args.resolved_model_name}")
    print(f"Gene encoder:  {args.gene_encoder}")
    if args.hf_endpoint:
        print(f"Using HF endpoint: {args.hf_endpoint}")
    if args.hf_hub_download_timeout and args.hf_hub_download_timeout > 0:
        print(f"HF_HUB_DOWNLOAD_TIMEOUT={args.hf_hub_download_timeout}")
    if args.hf_hub_etag_timeout and args.hf_hub_etag_timeout > 0:
        print(f"HF_HUB_ETAG_TIMEOUT={args.hf_hub_etag_timeout}")

    processed_paths = prepare_processed_dataset(
        hest_data_dir=args.hest_data_dir,
        output_dir=args.processed_dir,
        sample_id=args.sample_id,
        remove_control_features=parse_bool(args.remove_control_features),
        nucleus_only=parse_bool(args.nucleus_only),
        drop_zero_expression=parse_bool(args.drop_zero_expression),
        force_rebuild=args.force_rebuild_cache,
    )

    if args.gene_encoder == "scfoundation":
        encoder_target_gene_names = load_gene_names_from_tsv(resolve_scfoundation_gene_list_path(args.scfoundation_repo_dir))
        encoder_use_raw_counts = True
    else:
        with open(processed_paths.genes_path, "r", encoding="utf-8") as handle:
            encoder_target_gene_names = list(json.load(handle)["genes"])
        encoder_use_raw_counts = False

    train_dataset = XeniumSingleCellDataset(
        processed_dir=args.processed_dir,
        hest_data_dir=args.hest_data_dir,
        sample_id=args.sample_id,
        split="train",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size=args.crop_size,
        image_size=args.image_size,
        wsi_level=args.wsi_level,
        augment=True,
        max_cells=args.max_train_cells,
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )
    train_eval_dataset = XeniumSingleCellDataset(
        processed_dir=args.processed_dir,
        hest_data_dir=args.hest_data_dir,
        sample_id=args.sample_id,
        split="train",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size=args.crop_size,
        image_size=args.image_size,
        wsi_level=args.wsi_level,
        augment=False,
        max_cells=args.max_train_cells,
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )
    train_bank_dataset = XeniumSingleCellDataset(
        processed_dir=args.processed_dir,
        hest_data_dir=args.hest_data_dir,
        sample_id=args.sample_id,
        split="train",
        test_fold=args.test_fold,
        num_position_folds=args.num_position_folds,
        crop_size=args.crop_size,
        image_size=args.image_size,
        wsi_level=args.wsi_level,
        augment=False,
        include_image=False,
        max_cells=args.max_train_cells,
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )
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
        encoder_target_gene_names=encoder_target_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )

    print(f"Train cells: {len(train_dataset)}")
    print(f"Test cells:  {len(test_dataset)}")
    print(f"Gene dim:    {train_dataset.num_features}")
    print(f"Encoder dim: {len(encoder_target_gene_names)}")

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

    scrna_loader = None
    scrna_dataset = None
    scrna_enabled = parse_bool(args.use_scrna_loss) and bool(args.scrna_data_path) and args.scrna_loss_weight > 0
    if scrna_enabled:
        if not os.path.exists(args.scrna_data_path):
            raise FileNotFoundError(f"scRNA h5ad file not found: {args.scrna_data_path}")
        scrna_dataset = ScRNADataset(
            h5ad_path=args.scrna_data_path,
            target_gene_names=encoder_target_gene_names,
            use_raw_counts=encoder_use_raw_counts,
            max_cells=args.scrna_max_cells,
        )
        scrna_num_workers = args.num_workers if args.scrna_num_workers < 0 else args.scrna_num_workers
        scrna_loader = create_loader(scrna_dataset, args.scrna_batch_size, scrna_num_workers, shuffle=False)
        print(f"scRNA cells:  {len(scrna_dataset)}")
    else:
        print("scRNA auxiliary loss disabled")

    history: List[Dict[str, float]] = []
    best_train_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        scrna_bank_embeddings = None
        if scrna_loader is not None:
            scrna_bank_embeddings = collect_scrna_embeddings(model, scrna_loader, device)
            print(f"Refreshed scRNA bank: {tuple(scrna_bank_embeddings.shape)}")

        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scrna_bank_embeddings=scrna_bank_embeddings,
            scrna_loss_weight=args.scrna_loss_weight,
            scrna_knn_percent=args.scrna_knn_percent,
            scrna_temperature=args.scrna_temperature,
        )

        epoch_train_subset, num_epoch_train_queries = build_eval_subset(
            train_eval_dataset,
            max_cells=args.epoch_eval_max_cells,
            seed=args.seed + epoch,
        )
        epoch_test_subset, num_epoch_test_queries = build_eval_subset(
            test_dataset,
            max_cells=args.epoch_eval_max_cells,
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
            "image_gene_loss": float(train_stats["main_loss"]),
            "scrna_loss": float(train_stats["scrna_loss"]),
            "train_subset_overall_pearson": float(train_metrics["overall_pearson"]),
            "test_subset_overall_pearson": float(test_metrics["overall_pearson"]),
            "train_subset_queries": int(num_epoch_train_queries),
            "test_subset_queries": int(num_epoch_test_queries),
        }
        history.append(epoch_record)
        print(
            "train_loss={train_loss:.4f} image_gene_loss={main_loss:.4f} scrna_loss={scrna_loss:.4f} "
            "train_subset_pearson={train_p:.4f} test_subset_pearson={test_p:.4f}".format(
                train_loss=train_stats["total_loss"],
                main_loss=train_stats["main_loss"],
                scrna_loss=train_stats["scrna_loss"],
                train_p=train_metrics["overall_pearson"],
                test_p=test_metrics["overall_pearson"],
            )
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "args": vars(args),
            },
            os.path.join(args.run_dir, "last.pt"),
        )
        if train_stats["total_loss"] < best_train_loss:
            best_train_loss = train_stats["total_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "args": vars(args),
                },
                os.path.join(args.run_dir, "best_train_loss.pt"),
            )

    print("\nRunning final evaluation...")
    final_train_subset, num_final_train_queries = build_eval_subset(
        train_eval_dataset,
        max_cells=args.final_train_eval_max_cells,
        seed=args.seed + 2000,
    )
    final_test_subset, num_final_test_queries = build_eval_subset(
        test_dataset,
        max_cells=args.final_test_eval_max_cells,
        seed=args.seed + 3000,
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

    summary = {
        "sample_id": args.sample_id,
        "device": str(device),
        "epochs": int(args.epochs),
        "train_cells": int(len(train_dataset)),
        "test_cells": int(len(test_dataset)),
        "scrna_cells": int(len(scrna_dataset)) if scrna_dataset is not None else 0,
        "num_genes": int(train_dataset.num_features),
        "encoder_input_dim": int(len(encoder_target_gene_names)),
        "model": args.model,
        "resolved_model_name": args.resolved_model_name,
        "gene_encoder": args.gene_encoder,
        "pretrained": bool(args.pretrained),
        "image_encoder_checkpoint": args.image_encoder_checkpoint or None,
        "scfoundation_repo_dir": args.scfoundation_repo_dir if args.gene_encoder == "scfoundation" else None,
        "scfoundation_checkpoint": args.scfoundation_checkpoint if args.gene_encoder == "scfoundation" else None,
        "scfoundation_key": args.scfoundation_key if args.gene_encoder == "scfoundation" else None,
        "hf_endpoint": args.hf_endpoint or None,
        "hf_hub_download_timeout": int(args.hf_hub_download_timeout),
        "hf_hub_etag_timeout": int(args.hf_hub_etag_timeout),
        "test_fold": int(args.test_fold),
        "num_position_folds": int(args.num_position_folds),
        "max_train_cells": int(args.max_train_cells),
        "max_test_cells": int(args.max_test_cells),
        "top_k": int(args.top_k),
        "use_scrna_loss": bool(parse_bool(args.use_scrna_loss)),
        "scrna_data_path": args.scrna_data_path or None,
        "scrna_loss_weight": float(args.scrna_loss_weight),
        "scrna_knn_percent": float(args.scrna_knn_percent),
        "scrna_temperature": float(args.scrna_temperature),
        "scrna_max_cells": int(args.scrna_max_cells),
        "epoch_eval_max_cells": int(args.epoch_eval_max_cells),
        "final_train_eval_max_cells": int(args.final_train_eval_max_cells),
        "final_test_eval_max_cells": int(args.final_test_eval_max_cells),
        "final_train_metrics": final_train_metrics,
        "final_test_metrics": final_test_metrics,
        "history": history,
    }
    save_json(summary, os.path.join(args.run_dir, "metrics.json"))
    save_json(
        {
            "train_cells": train_dataset.cell_ids[train_dataset.indices].astype(int).tolist(),
            "test_cells": test_dataset.cell_ids[test_dataset.indices].astype(int).tolist(),
            "fold_edges": train_dataset.fold_edges,
            "num_final_train_queries": int(num_final_train_queries),
            "num_final_test_queries": int(num_final_test_queries),
        },
        os.path.join(args.run_dir, "split_manifest.json"),
    )

    print(f"Final train overall Pearson: {final_train_metrics['overall_pearson']:.4f}")
    print(f"Final test overall Pearson:  {final_test_metrics['overall_pearson']:.4f}")
    print(f"Saved outputs to {args.run_dir}")


if __name__ == "__main__":
    main()
