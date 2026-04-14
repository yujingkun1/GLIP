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
DEFAULT_SHARED_GENE_FILE = "/data/yujk/GLIP/configs/brca_shared_genes_ncbi784_ref_spa124_313.txt"


def parse_sample_ids(raw_sample_ids: str | Sequence[str] | None) -> List[str]:
    if raw_sample_ids is None:
        return []
    if isinstance(raw_sample_ids, str):
        return [sample_id.strip() for sample_id in raw_sample_ids.split(",") if sample_id.strip()]
    return [str(sample_id).strip() for sample_id in raw_sample_ids if str(sample_id).strip()]


@dataclass
class RunConfig:
    visium_heldout_sample: str
    visium_sample_ids: List[str]
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
    note: str


class WrappedVisiumDataset(Dataset):
    def __init__(self, dataset: Dataset, source_name: str) -> None:
        self.dataset = dataset
        self.source_name = source_name
        self.num_features = getattr(dataset, "num_features", None)
        self.num_genes = getattr(dataset, "num_genes", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        sample = self.dataset[index]
        sample["source"] = self.source_name
        return sample


class WrappedXeniumPseudoSpotDataset(Dataset):
    def __init__(self, dataset: XeniumPseudoSpotDataset, source_name: str) -> None:
        self.dataset = dataset
        self.source_name = source_name
        self.num_features = getattr(dataset, "num_features", None)
        self.num_genes = getattr(dataset, "num_features", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        sample = self.dataset[index]
        return {
            "image": sample["image"],
            "reduced_expression": sample["expression"].float(),
            "barcode": f"{self.source_name}_{int(sample['spot_id'])}",
            "sample_id": self.source_name,
            "spatial_coords": [float(sample["centroid_x"]), float(sample["centroid_y"])],
            "source": self.source_name,
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
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "reduced_expression": batch["reduced_expression"].to(device, non_blocking=True),
    }


def train_epoch(model, train_loader: DataLoader, optimizer, device: torch.device) -> AvgMeter:
    loss_meter = AvgMeter("train_loss")
    tqdm_object = tqdm(train_loader, total=len(train_loader), desc="joint_train")
    model.train()
    for batch in tqdm_object:
        moved = move_batch_to_device(batch, device)
        loss = model(moved)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = moved["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


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
        spot_embeddings = model.spot_projection(reduced_expression)
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
        image_features = model.image_encoder(images)
        image_embeddings = model.image_projection(image_features)
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


def build_model(model_name: str, resolved_model_name: str, spot_embedding_dim: int, pretrained: bool, checkpoint_path: str):
    choice = str(model_name).strip().lower()
    if choice == "clip":
        return CLIPModel_CLIP(spot_embedding=spot_embedding_dim)
    if choice == "vit":
        return CLIPModel_ViT(spot_embedding=spot_embedding_dim)
    if choice == "vit_l":
        return CLIPModel_ViT_L(spot_embedding=spot_embedding_dim)
    if choice == "resnet101":
        return CLIPModel_resnet101(spot_embedding=spot_embedding_dim)
    if choice == "resnet152":
        return CLIPModel_resnet152(spot_embedding=spot_embedding_dim)
    if choice == "uni" or resolved_model_name == UNI_MODEL_NAME:
        return CLIPModel_UNI(spot_embedding=spot_embedding_dim, pretrained=pretrained, checkpoint_path=checkpoint_path)
    return CLIPModel(spot_embedding=spot_embedding_dim, model_name=resolved_model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)


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
    train_indices = []
    test_indices = []
    for idx, entry in enumerate(dataset.entries):
        if entry["sample_id"] == args.visium_heldout_sample:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
    train_indices = maybe_subset(np.asarray(train_indices, dtype=np.int64), args.max_visium_train_spots)
    test_indices = maybe_subset(np.asarray(test_indices, dtype=np.int64), args.max_visium_test_spots)
    if len(test_indices) == 0:
        raise RuntimeError(f"No test visium spots for heldout sample {args.visium_heldout_sample}")
    return dataset, train_indices, test_indices


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
    parser.add_argument('--epochs', type=int, default=3)
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
    parser.add_argument('--note', default='Stage2 minimal naive joint training on pooled Visium + Xenium pseudo-spots in a shared 313-gene space.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.visium_sample_ids = parse_sample_ids(args.visium_sample_ids) or list(DEFAULT_VISIUM_SAMPLE_IDS)
    args.pretrained = parse_bool(args.pretrained)
    args.resolved_model_name = resolve_model_name(args.model)

    os.makedirs(args.run_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')

    visium_dataset, vis_train_indices, vis_test_indices = build_visium_subsets(args, args.shared_gene_file)
    x_train_ds, x_train_eval_ds, x_test_ds, shared_genes = build_xenium_datasets(args, args.shared_gene_file)

    vis_train = WrappedVisiumDataset(Subset(visium_dataset, vis_train_indices.tolist()), source_name='visium_train')
    vis_train_eval = WrappedVisiumDataset(Subset(visium_dataset, vis_train_indices.tolist()), source_name='visium_train')
    vis_test = WrappedVisiumDataset(Subset(visium_dataset, vis_test_indices.tolist()), source_name=f'visium_test_{args.visium_heldout_sample}')
    xen_train = WrappedXeniumPseudoSpotDataset(x_train_ds, source_name='xenium_train')
    xen_train_eval = WrappedXeniumPseudoSpotDataset(x_train_eval_ds, source_name='xenium_train')
    xen_test = WrappedXeniumPseudoSpotDataset(x_test_ds, source_name=f'xenium_test_{args.xenium_sample_id}')

    combined_train = ConcatDataset([vis_train, xen_train])
    combined_train_eval = ConcatDataset([vis_train_eval, xen_train_eval])
    vis_test_loader = create_loader(vis_test, args.eval_batch_size, args.num_workers, shuffle=False)
    xen_test_loader = create_loader(xen_test, args.eval_batch_size, args.num_workers, shuffle=False)
    train_loader = create_loader(combined_train, args.batch_size, args.num_workers, shuffle=True)
    train_eval_loader = create_loader(combined_train_eval, args.eval_batch_size, args.num_workers, shuffle=False)

    model = build_model(args.model, args.resolved_model_name, len(shared_genes), args.pretrained, args.image_encoder_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = RunConfig(
        visium_heldout_sample=args.visium_heldout_sample,
        visium_sample_ids=args.visium_sample_ids,
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
        note=args.note,
    )
    save_json(asdict(config), os.path.join(args.run_dir, 'joint_config.json'))
    save_json({
        'manual_baseline_override': True,
        'baseline_source': 'user-provided manual single-platform results',
        'shared_gene_count': len(shared_genes),
        'shared_gene_file': os.path.abspath(args.shared_gene_file),
        'visium_train_spots': len(vis_train),
        'visium_test_spots': len(vis_test),
        'xenium_train_spots': len(xen_train),
        'xenium_test_spots': len(xen_test),
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
            'joint_train_loss': float(train_stats.avg),
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

    joint_bank = collect_spot_bank(model, train_eval_loader, device)
    vis_queries = collect_image_queries(model, vis_test_loader, device)
    xen_queries = collect_image_queries(model, xen_test_loader, device)

    vis_predictions = predict_expression_from_retrieval(joint_bank, vis_queries, top_k=args.top_k, chunk_size=args.retrieval_chunk_size)
    xen_predictions = predict_expression_from_retrieval(joint_bank, xen_queries, top_k=args.top_k, chunk_size=args.retrieval_chunk_size)

    vis_metrics = compute_metrics(vis_predictions, vis_queries['expressions'].numpy())
    xen_metrics = compute_metrics(xen_predictions, xen_queries['expressions'].numpy())
    vis_metrics['top_k'] = int(args.top_k)
    xen_metrics['top_k'] = int(args.top_k)
    vis_metrics['heldout_sample'] = args.visium_heldout_sample
    xen_metrics['xenium_sample_id'] = args.xenium_sample_id

    summary = {
        'history': history,
        'shared_gene_count': len(shared_genes),
        'shared_genes_preview': shared_genes[:50],
        'best_epoch': int(best_payload['epoch']),
        'visium_test_metrics': vis_metrics,
        'xenium_test_metrics': xen_metrics,
        'joint_bank_size': int(joint_bank['embeddings'].shape[0]),
        'visium_test_queries': int(vis_queries['embeddings'].shape[0]),
        'xenium_test_queries': int(xen_queries['embeddings'].shape[0]),
        'manual_baseline_override': True,
        'no_extra_datasets_used': True,
    }
    save_json(summary, os.path.join(args.run_dir, 'metrics.json'))

    rows = [
        {'target': 'visium', **vis_metrics},
        {'target': 'xenium_pseudospot', **xen_metrics},
    ]
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
