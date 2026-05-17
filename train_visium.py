import argparse
import csv
import json
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from glip.visium import config as CFG
from glip.visium.dataset import CLIPDataset, CLIPSubset
from glip.visium.utils import AvgMeter, get_lr


DEFAULT_LOO_SAMPLE_IDS = [
    "SPA154", "SPA153", "SPA152", "SPA151", "SPA150", "SPA149",
    "SPA148", "SPA147", "SPA146", "SPA145", "SPA144", "SPA143",
    "SPA142", "SPA141", "SPA140", "SPA139", "SPA138", "SPA137",
    "SPA136", "SPA135", "SPA134", "SPA133", "SPA132", "SPA131",
    "SPA130", "SPA129", "SPA128", "SPA127", "SPA126", "SPA125",
    "SPA124", "SPA123", "SPA122", "SPA121", "SPA120", "SPA119",
]

UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"
H0MINI_MODEL_NAME = "hf-hub:bioptimus/H0-mini"

MODEL_NAME_ALIASES = {
    "uni": UNI_MODEL_NAME,
    "uni2-h": UNI_MODEL_NAME,
    "h0mini": H0MINI_MODEL_NAME,
    "h0-mini": H0MINI_MODEL_NAME,
}

DEFAULT_VISIUM_GENE_FILE = "/data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt"


parser = argparse.ArgumentParser(description="Train the Visium/ST-level BLEEP model inside GLIP")

parser.add_argument("--exp_name", type=str, default="/data/yujk/GLIP/runs_visium/hest_bleep_loo_uni", help="output directory")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--max_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--num_workers", type=int, default=8, help="data loader worker count")

parser.add_argument("--hest_data_dir", type=str, default="/data/yujk/hovernet2feature/HEST/hest_data", help="HEST root directory")
parser.add_argument("--sample_ids", type=str, default="", help="comma-separated HEST sample ids; empty uses the default SPA leave-one-out panel")
parser.add_argument("--cv_mode", type=str, default="leave_one_out", choices=["leave_one_out", "fixed_manifest"], help="cross-validation mode")
parser.add_argument("--fold_manifest", type=str, default="", help="optional fixed fold manifest JSON for cv_mode=fixed_manifest")
parser.add_argument("--fold_index", type=int, default=-1, help="fold index inside fold_manifest for cv_mode=fixed_manifest")
parser.add_argument("--gene_file", type=str, default=DEFAULT_VISIUM_GENE_FILE, help="gene list file, one gene per line")
parser.add_argument("--max_spots_per_sample", type=int, default=0, help="optional cap for spots per sample, 0 means no cap")
parser.add_argument("--top_k", type=int, default=50, help="top-k retrieved training spots used to predict each test spot")
parser.add_argument("--retrieval_chunk_size", type=int, default=1024, help="query chunk size used during retrieval evaluation")
parser.add_argument("--pretrained", type=str, default="true", help="whether to use timm pretrained weights")
parser.add_argument(
    "--image_encoder_checkpoint",
    type=str,
    default="/data/yujk/UNI2-h/pytorch_model.bin",
    help="optional local timm checkpoint path",
)
parser.add_argument("--hf_endpoint", type=str, default="", help="optional Hugging Face Hub endpoint, e.g. https://hf-mirror.com")
parser.add_argument("--hf_hub_download_timeout", type=int, default=0, help="optional HF Hub download timeout in seconds")
parser.add_argument("--hf_hub_etag_timeout", type=int, default=0, help="optional HF Hub metadata timeout in seconds")

parser.add_argument("--init_method", default="tcp://127.0.0.1:3456", type=str, help="torch distributed init method")
parser.add_argument("--dist-backend", default="nccl", type=str, help="torch distributed backend")
parser.add_argument("--world_size", default=1, type=int, help="world size for distributed training")
parser.add_argument("--distributed", action="store_true", help="enable distributed training")
parser.add_argument(
    "--model",
    type=str,
    default="uni",
    help="image encoder backbone or alias, e.g. resnet50, uni, vit, clip",
)
parser.add_argument("--device_id", type=int, default=1, help="CUDA device id to use for single-GPU training")
parser.add_argument("--trainable", action="store_true", help="Unfreeze H0-mini parameters for fine-tuning")
parser.add_argument("--lr", type=float, default=CFG.lr, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay, help="weight decay")


def parse_sample_ids(raw_sample_ids):
    if raw_sample_ids is None:
        return []
    if isinstance(raw_sample_ids, str):
        return [sample_id.strip() for sample_id in raw_sample_ids.split(",") if sample_id.strip()]
    return [str(sample_id).strip() for sample_id in raw_sample_ids if str(sample_id).strip()]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def resolve_image_model_name(model_name):
    normalized = str(model_name).strip()
    if not normalized:
        raise ValueError("Image encoder backbone cannot be empty.")
    if normalized.startswith("hf_hub:"):
        normalized = "hf-hub:" + normalized[len("hf_hub:"):]
    return MODEL_NAME_ALIASES.get(normalized.lower(), normalized)


def configure_hf_hub(args):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.hf_hub_download_timeout and args.hf_hub_download_timeout > 0:
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(int(args.hf_hub_download_timeout))

    if args.hf_hub_etag_timeout and args.hf_hub_etag_timeout > 0:
        os.environ["HF_HUB_ETAG_TIMEOUT"] = str(int(args.hf_hub_etag_timeout))


def resolve_cv_sample_ids(args):
    sample_ids = parse_sample_ids(args.sample_ids)
    if sample_ids:
        return sample_ids
    return list(DEFAULT_LOO_SAMPLE_IDS)


def load_fixed_fold_manifest(manifest_path, fold_index):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    folds = payload.get("folds") if isinstance(payload, dict) else payload
    if not isinstance(folds, list):
        raise ValueError(f"Invalid fold manifest format: {manifest_path}")

    for fold in folds:
        if int(fold.get("fold_index", -1)) == int(fold_index):
            train_samples = parse_sample_ids(fold.get("train_samples"))
            test_samples = parse_sample_ids(fold.get("test_samples"))
            if not train_samples or not test_samples:
                raise ValueError(f"Fold {fold_index} in {manifest_path} is missing train/test samples")
            sample_ids = parse_sample_ids(payload.get("sample_ids")) if isinstance(payload, dict) else []
            if not sample_ids:
                sample_ids = sorted(set(train_samples + test_samples))
            return {
                "fold_index": int(fold_index),
                "sample_ids": sample_ids,
                "train_samples": train_samples,
                "test_samples": test_samples,
                "split_name": payload.get("split_name", os.path.basename(manifest_path)) if isinstance(payload, dict) else os.path.basename(manifest_path),
            }
    raise ValueError(f"Fold index {fold_index} not found in manifest: {manifest_path}")


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(rows, fieldnames, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unwrap_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def create_loader(dataset, batch_size, num_workers, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_base_dataset(args, sample_ids):
    print("Building base HEST dataset")
    dataset = CLIPDataset(
        hest_data_dir=args.hest_data_dir,
        sample_ids=sample_ids,
        gene_file=args.gene_file or None,
        max_spots_per_sample=args.max_spots_per_sample or None,
        is_train=False,
        model_name=args.model,  # Pass model name for normalization
    )

    print(f"Total aligned spots across CV panel: {len(dataset)}")
    print(f"Gene dimension: {dataset.num_features}")
    return dataset


def build_fold_loaders(args, base_dataset, test_samples):
    train_indices = []
    test_indices = []
    test_samples = set(parse_sample_ids(test_samples))
    if not test_samples:
        raise RuntimeError("At least one test sample is required to build a fold.")

    for idx, entry in enumerate(base_dataset.entries):
        if entry["sample_id"] in test_samples:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    if not train_indices:
        raise RuntimeError(f"No training spots available for held-out samples {sorted(test_samples)}")
    if not test_indices:
        raise RuntimeError(f"No test spots available for held-out samples {sorted(test_samples)}")

    train_dataset = CLIPSubset(base_dataset, train_indices, is_train=True)
    train_eval_dataset = CLIPSubset(base_dataset, train_indices, is_train=False)
    test_dataset = CLIPSubset(base_dataset, test_indices, is_train=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None

    train_loader = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=len(train_dataset) >= args.batch_size,
    )
    train_eval_loader = create_loader(
        train_eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=None,
        drop_last=False,
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=None,
        drop_last=False,
    )

    return train_loader, train_eval_loader, test_loader, train_dataset, test_dataset


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def move_batch_to_device(batch, device):
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "reduced_expression": batch["reduced_expression"].to(device, non_blocking=True),
    }


def train_epoch(model, train_loader, optimizer, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        batch = move_batch_to_device(batch, device)
        loss = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def test_epoch(model, test_loader, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    for batch in tqdm_object:
        batch = move_batch_to_device(batch, device)
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter


def build_model(args, spot_embedding_dim):
    from glip.visium.models import (
        CLIPModel,
        CLIPModel_CLIP,
        CLIPModel_H0mini,
        CLIPModel_UNI,
        CLIPModel_ViT,
        CLIPModel_ViT_L,
        CLIPModel_resnet101,
        CLIPModel_resnet152,
    )

    model_choice = str(args.model).strip().lower()

    if model_choice == "clip":
        print("Image encoder is CLIP")
        return CLIPModel_CLIP(spot_embedding=spot_embedding_dim)
    if model_choice == "vit":
        print("Image encoder is ViT")
        return CLIPModel_ViT(spot_embedding=spot_embedding_dim)
    if model_choice == "vit_l":
        print("Image encoder is ViT_L")
        return CLIPModel_ViT_L(spot_embedding=spot_embedding_dim)
    if model_choice == "resnet101":
        print("Image encoder is ResNet101")
        return CLIPModel_resnet101(spot_embedding=spot_embedding_dim)
    if model_choice == "resnet152":
        print("Image encoder is ResNet152")
        return CLIPModel_resnet152(spot_embedding=spot_embedding_dim)
    if model_choice == "h0mini" or model_choice == "h0-mini" or args.resolved_model_name == H0MINI_MODEL_NAME:
        print(f"Image encoder is H0-mini ({H0MINI_MODEL_NAME})")
        return CLIPModel_H0mini(
            spot_embedding=spot_embedding_dim,
            pretrained=args.pretrained,
            checkpoint_path=args.image_encoder_checkpoint,
            output_mode="pooled",
            trainable=getattr(args, 'trainable', False),
        )
    if model_choice == "uni" or args.resolved_model_name == UNI_MODEL_NAME:
        print(f"Image encoder is UNI2-h ({UNI_MODEL_NAME})")
        return CLIPModel_UNI(spot_embedding=spot_embedding_dim)

    print(f"Image encoder is {args.resolved_model_name}")
    return CLIPModel(
        spot_embedding=spot_embedding_dim,
        model_name=args.resolved_model_name,
    )


def safe_pearson(x, y):
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
def collect_spot_bank(model, loader, device):
    model_ref = unwrap_model(model)
    embeddings = []
    expressions = []
    sample_ids = []
    barcodes = []

    for batch in loader:
        reduced_expression = batch["reduced_expression"].to(device, non_blocking=True)
        spot_embeddings = model_ref.spot_projection(reduced_expression)
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
def collect_image_queries(model, loader, device):
    model_ref = unwrap_model(model)
    embeddings = []
    expressions = []
    sample_ids = []
    barcodes = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        image_features = model_ref.image_encoder(images)
        image_embeddings = model_ref.image_projection(image_features)
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


def predict_expression_from_retrieval(train_spot_bank, test_image_queries, top_k=1, chunk_size=1024):
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


def compute_pearson_metrics(predictions, targets):
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


def evaluate_fold(model, train_eval_loader, test_loader, device, top_k, retrieval_chunk_size):
    model.eval()

    train_spot_bank = collect_spot_bank(model, train_eval_loader, device)
    test_image_queries = collect_image_queries(model, test_loader, device)
    predicted_expression = predict_expression_from_retrieval(
        train_spot_bank=train_spot_bank,
        test_image_queries=test_image_queries,
        top_k=top_k,
        chunk_size=retrieval_chunk_size,
    )
    target_expression = test_image_queries["expressions"].numpy()

    metrics = compute_pearson_metrics(predicted_expression, target_expression)
    metrics["top_k"] = int(max(1, top_k))

    return metrics


def setup_runtime(args):
    env_world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    args.world_size = env_world_size
    args.distributed = args.distributed or args.world_size > 1

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if args.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
            current_device = min(local_rank, max(ngpus_per_node - 1, 0))
        else:
            if args.device_id < 0 or args.device_id >= ngpus_per_node:
                raise ValueError(
                    f"Requested --device_id {args.device_id}, but available CUDA device range is 0 to {ngpus_per_node - 1}."
                )
            local_rank = args.device_id
            current_device = args.device_id
        torch.cuda.set_device(current_device)
        device = torch.device("cuda", current_device)
    else:
        if args.dist_backend == "nccl":
            raise ValueError("NCCL requires CUDA. Use --dist-backend gloo for CPU-only execution.")
        ngpus_per_node = 1
        local_rank = 0
        current_device = None
        device = torch.device("cpu")

    if args.distributed:
        rank = int(
            os.environ.get(
                "RANK",
                int(os.environ.get("SLURM_NODEID", 0)) * ngpus_per_node + local_rank,
            )
        )
        print(f"From Rank: {rank}, ==> Initializing Process Group...")
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=rank,
        )
        print("Process group ready!")
    else:
        rank = 0
        print(f"Using device: {device}")

    return device, rank, current_device


def summarize_fold_metrics(all_fold_metrics):
    overall_values = [fold["overall_pearson"] for fold in all_fold_metrics]
    gene_values = [fold["mean_gene_pearson"] for fold in all_fold_metrics]
    spot_values = [fold["mean_spot_pearson"] for fold in all_fold_metrics]

    return {
        "num_folds": len(all_fold_metrics),
        "average_overall_pearson": float(np.mean(overall_values)) if overall_values else 0.0,
        "std_overall_pearson": float(np.std(overall_values)) if overall_values else 0.0,
        "average_mean_gene_pearson": float(np.mean(gene_values)) if gene_values else 0.0,
        "std_mean_gene_pearson": float(np.std(gene_values)) if gene_values else 0.0,
        "average_mean_spot_pearson": float(np.mean(spot_values)) if spot_values else 0.0,
        "std_mean_spot_pearson": float(np.std(spot_values)) if spot_values else 0.0,
    }


def _build_loss_curve_svg(train_losses, test_losses, title):
    width = 900
    height = 520
    left = 80
    right = 30
    top = 60
    bottom = 70

    values = list(train_losses) + list(test_losses)
    if not values:
        values = [0.0, 1.0]

    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.5
        y_max += 0.5

    def scale_x(epoch_idx, total_epochs):
        plot_width = width - left - right
        if total_epochs <= 1:
            return left + plot_width / 2
        return left + (epoch_idx / (total_epochs - 1)) * plot_width

    def scale_y(value):
        plot_height = height - top - bottom
        return top + (y_max - value) / (y_max - y_min) * plot_height

    def polyline_points(values_seq):
        return " ".join(
            f"{scale_x(i, len(values_seq)):.2f},{scale_y(v):.2f}"
            for i, v in enumerate(values_seq)
        )

    total_epochs = max(len(train_losses), len(test_losses), 1)
    y_ticks = 5
    x_labels = "".join(
        f"<text x='{scale_x(i, total_epochs):.2f}' y='{height - bottom + 28}' font-size='12' text-anchor='middle'>{i + 1}</text>"
        for i in range(total_epochs)
    )
    y_labels = []
    y_grid = []
    for tick_idx in range(y_ticks + 1):
        frac = tick_idx / y_ticks
        value = y_min + (y_max - y_min) * frac
        y = scale_y(value)
        y_labels.append(
            f"<text x='{left - 12}' y='{y + 4:.2f}' font-size='12' text-anchor='end'>{value:.4f}</text>"
        )
        y_grid.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' stroke='#d9dde3' stroke-width='1' />"
        )

    train_polyline = polyline_points(train_losses) if train_losses else ""
    test_polyline = polyline_points(test_losses) if test_losses else ""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
  <text x="{width / 2:.2f}" y="32" font-size="22" text-anchor="middle" font-family="Arial, sans-serif">{title}</text>
  {''.join(y_grid)}
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222" stroke-width="2" />
  {''.join(y_labels)}
  {x_labels}
  <text x="{width / 2:.2f}" y="{height - 18}" font-size="14" text-anchor="middle" font-family="Arial, sans-serif">Epoch</text>
  <text x="22" y="{height / 2:.2f}" font-size="14" text-anchor="middle" transform="rotate(-90 22 {height / 2:.2f})" font-family="Arial, sans-serif">Loss</text>
  <polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{train_polyline}" />
  <polyline fill="none" stroke="#d62728" stroke-width="3" points="{test_polyline}" />
  <rect x="{width - 220}" y="74" width="16" height="16" fill="#1f77b4" />
  <text x="{width - 196}" y="87" font-size="13" font-family="Arial, sans-serif">Train Loss</text>
  <rect x="{width - 220}" y="102" width="16" height="16" fill="#d62728" />
  <text x="{width - 196}" y="115" font-size="13" font-family="Arial, sans-serif">Test Loss</text>
</svg>
"""


def save_fold_training_artifacts(fold_dir, heldout_sample, train_losses, test_losses):
    history_rows = []
    for epoch_idx in range(max(len(train_losses), len(test_losses))):
        history_rows.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": float(train_losses[epoch_idx]) if epoch_idx < len(train_losses) else None,
                "test_loss": float(test_losses[epoch_idx]) if epoch_idx < len(test_losses) else None,
            }
        )

    save_json(history_rows, os.path.join(fold_dir, "training_history.json"))
    save_csv(history_rows, ["epoch", "train_loss", "test_loss"], os.path.join(fold_dir, "training_history.csv"))

    svg = _build_loss_curve_svg(train_losses, test_losses, f"Loss Curve - {heldout_sample}")
    with open(os.path.join(fold_dir, "loss_curve.svg"), "w", encoding="utf-8") as handle:
        handle.write(svg)


def main():
    print("Starting...")
    args = parser.parse_args()
    device, rank, current_device = setup_runtime(args)
    args.pretrained = parse_bool(args.pretrained)
    args.model = args.model.strip()
    args.resolved_model_name = resolve_image_model_name(args.model)
    args.hf_endpoint = args.hf_endpoint.strip()
    args.image_encoder_checkpoint = os.path.expanduser(args.image_encoder_checkpoint.strip()) if args.image_encoder_checkpoint else ""
    configure_hf_hub(args)

    if args.image_encoder_checkpoint and not os.path.exists(args.image_encoder_checkpoint):
        raise FileNotFoundError(f"Local image encoder checkpoint not found: {args.image_encoder_checkpoint}")

    if args.image_encoder_checkpoint and args.pretrained:
        print("Local image encoder checkpoint provided; disabling remote pretrained download.")
        args.pretrained = False

    CFG.pretrained = args.pretrained
    CFG.model_name = args.resolved_model_name
    CFG.image_encoder_checkpoint = args.image_encoder_checkpoint

    if args.cv_mode == "fixed_manifest":
        if not args.fold_manifest:
            raise RuntimeError("--fold_manifest is required when --cv_mode fixed_manifest is used.")
        if args.fold_index < 0:
            raise RuntimeError("--fold_index must be >= 0 when --cv_mode fixed_manifest is used.")
        cv_setup = load_fixed_fold_manifest(args.fold_manifest, args.fold_index)
        sample_ids = cv_setup["sample_ids"]
    else:
        cv_setup = None
        sample_ids = resolve_cv_sample_ids(args)
        if len(sample_ids) < 2:
            raise RuntimeError("Leave-one-out requires at least two samples.")

    if rank == 0:
        if args.hf_endpoint:
            print(f"Using HF endpoint: {args.hf_endpoint}")
        if args.hf_hub_download_timeout and args.hf_hub_download_timeout > 0:
            print(f"HF_HUB_DOWNLOAD_TIMEOUT={args.hf_hub_download_timeout}")
        if args.hf_hub_etag_timeout and args.hf_hub_etag_timeout > 0:
            print(f"HF_HUB_ETAG_TIMEOUT={args.hf_hub_etag_timeout}")
        os.makedirs(args.exp_name, exist_ok=True)
        save_json(
            {
                "cv_mode": args.cv_mode,
                "sample_ids": sample_ids,
                "fold_manifest": args.fold_manifest or None,
                "fold_index": int(args.fold_index),
                "gene_file": args.gene_file or None,
                "model": args.model,
                "resolved_model_name": args.resolved_model_name,
                "top_k": int(args.top_k),
                "retrieval_chunk_size": int(args.retrieval_chunk_size),
                "pretrained": bool(args.pretrained),
                "image_encoder_checkpoint": args.image_encoder_checkpoint or None,
                "hf_endpoint": args.hf_endpoint or None,
                "hf_hub_download_timeout": int(args.hf_hub_download_timeout),
                "hf_hub_etag_timeout": int(args.hf_hub_etag_timeout),
                "fixed_manifest_split": cv_setup if cv_setup is not None else None,
            },
            os.path.join(args.exp_name, "cv_config.json"),
        )

    print(f"From Rank: {rank}, ==> Preparing data..")
    base_dataset = build_base_dataset(args, sample_ids)
    spot_embedding_dim = base_dataset.num_features
    CFG.spot_embedding = spot_embedding_dim

    all_fold_metrics = []
    completed_folds = {}
    metrics_list_name = "loo_fold_metrics.json" if args.cv_mode == "leave_one_out" else "cv_fold_metrics.json"
    summary_name = "loo_summary.json" if args.cv_mode == "leave_one_out" else "cv_summary.json"
    completed_metrics_path = os.path.join(args.exp_name, metrics_list_name)
    if args.cv_mode == "leave_one_out" and rank == 0 and os.path.exists(completed_metrics_path):
        try:
            existing_metrics = json.load(open(completed_metrics_path, "r", encoding="utf-8"))
            if isinstance(existing_metrics, list):
                for fold_metrics in existing_metrics:
                    heldout_sample = fold_metrics.get("heldout_sample")
                    if heldout_sample:
                        completed_folds[heldout_sample] = fold_metrics
                all_fold_metrics.extend(existing_metrics)
                if completed_folds:
                    print(
                        f"Resuming existing experiment: found {len(completed_folds)} completed fold(s) in "
                        f"{completed_metrics_path}"
                    )
        except Exception as exc:
            print(f"Warning: failed to load existing fold metrics from {completed_metrics_path}: {exc}")

    if args.cv_mode == "fixed_manifest":
        fold_jobs = [
            {
                "fold_idx": int(cv_setup["fold_index"]),
                "test_samples": list(cv_setup["test_samples"]),
                "train_samples": list(cv_setup["train_samples"]),
                "fold_name": f"fold_{int(cv_setup['fold_index']):02d}",
                "display_name": ",".join(cv_setup["test_samples"]),
            }
        ]
    else:
        fold_jobs = [
            {
                "fold_idx": int(fold_idx),
                "test_samples": [heldout_sample],
                "train_samples": [sample_id for sample_id in sample_ids if sample_id != heldout_sample],
                "fold_name": f"fold_{fold_idx:02d}_{heldout_sample}",
                "display_name": heldout_sample,
            }
            for fold_idx, heldout_sample in enumerate(sample_ids)
        ]

    for job_idx, fold_job in enumerate(fold_jobs):
        fold_idx = int(fold_job["fold_idx"])
        test_samples = list(fold_job["test_samples"])
        train_samples = list(fold_job["train_samples"])
        display_name = str(fold_job["display_name"])
        fold_name = str(fold_job["fold_name"])
        fold_dir = os.path.join(args.exp_name, fold_name)
        best_model_path = os.path.join(fold_dir, "best.pt")

        if rank == 0:
            if args.cv_mode == "leave_one_out" and display_name in completed_folds:
                print("")
                print(f"=== Fold {job_idx + 1}/{len(fold_jobs)} ===")
                print(f"Held-out sample: {display_name}")
                print("Skipping fold because pearson_metrics already exist in loo_fold_metrics.json")
                continue
            os.makedirs(fold_dir, exist_ok=True)
            print("")
            print(f"=== Fold {job_idx + 1}/{len(fold_jobs)} ===")
            print(f"Held-out sample(s): {display_name}")
            print(f"Train sample count: {len(train_samples)}")

        train_loader, train_eval_loader, test_loader, train_dataset, test_dataset = build_fold_loaders(
            args=args,
            base_dataset=base_dataset,
            test_samples=test_samples,
        )

        if rank == 0:
            print(f"Train spots: {len(train_dataset)}")
            print(f"Test spots: {len(test_dataset)}")

        print(f"From Rank: {rank}, ==> Making model for {display_name}..")
        model = build_model(args, spot_embedding_dim).to(device)

        if args.distributed:
            if device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
            else:
                model = nn.parallel.DistributedDataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float("inf")
        best_epoch = 0
        train_loss_history = []
        test_loss_history = []

        for epoch in range(args.max_epochs):
            if rank == 0:
                print(f"Epoch: {epoch + 1}")

            if args.distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, device)

            model.eval()
            with torch.no_grad():
                test_loss = test_epoch(model, test_loader, device)

            if rank == 0:
                train_loss_history.append(float(train_loss.avg))
                test_loss_history.append(float(test_loss.avg))

            if rank == 0:
                print(f"Epoch {epoch + 1}: train loss {train_loss.avg:.6f}, test loss {test_loss.avg:.6f}")

            if test_loss.avg < best_loss and rank == 0:
                best_loss = test_loss.avg
                best_epoch = epoch + 1
                torch.save(unwrap_model(model).state_dict(), best_model_path)
                print(f"Saved best model with loss {best_loss:.6f}")

        if rank == 0:
            save_fold_training_artifacts(
                fold_dir=fold_dir,
                heldout_sample=display_name,
                train_losses=train_loss_history,
                test_losses=test_loss_history,
            )

            if os.path.exists(best_model_path):
                state_dict = torch.load(best_model_path, map_location=device)
                unwrap_model(model).load_state_dict(state_dict)

            fold_metrics = evaluate_fold(
                model=model,
                train_eval_loader=train_eval_loader,
                test_loader=test_loader,
                device=device,
                top_k=args.top_k,
                retrieval_chunk_size=args.retrieval_chunk_size,
            )
            fold_metrics["fold_idx"] = int(fold_idx)
            fold_metrics["heldout_sample"] = display_name
            fold_metrics["heldout_samples"] = test_samples
            fold_metrics["train_samples"] = train_samples
            fold_metrics["num_train_spots"] = int(len(train_dataset))
            fold_metrics["num_test_spots"] = int(len(test_dataset))
            fold_metrics["best_epoch"] = int(best_epoch)
            fold_metrics["best_test_loss"] = float(best_loss)
            fold_metrics["training_history_json"] = os.path.join(fold_dir, "training_history.json")
            fold_metrics["training_history_csv"] = os.path.join(fold_dir, "training_history.csv")
            fold_metrics["loss_curve_svg"] = os.path.join(fold_dir, "loss_curve.svg")

            save_json(fold_metrics, os.path.join(fold_dir, "pearson_metrics.json"))
            all_fold_metrics.append(fold_metrics)
            save_json(all_fold_metrics, os.path.join(args.exp_name, metrics_list_name))

            print(
                f"Fold {job_idx + 1} Pearson: overall={fold_metrics['overall_pearson']:.6f}, "
                f"gene={fold_metrics['mean_gene_pearson']:.6f}, "
                f"spot={fold_metrics['mean_spot_pearson']:.6f}"
            )

        del model, optimizer, train_loader, train_eval_loader, test_loader, train_dataset, test_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rank == 0:
        summary = summarize_fold_metrics(all_fold_metrics)
        summary["folds"] = all_fold_metrics
        save_json(summary, os.path.join(args.exp_name, summary_name))
        print("")
        print("=== CV Summary ===")
        print(f"Average overall Pearson: {summary['average_overall_pearson']:.6f} ± {summary['std_overall_pearson']:.6f}")
        print(
            f"Average mean gene Pearson: {summary['average_mean_gene_pearson']:.6f} ± "
            f"{summary['std_mean_gene_pearson']:.6f}"
        )
        print(
            f"Average mean spot Pearson: {summary['average_mean_spot_pearson']:.6f} ± "
            f"{summary['std_mean_spot_pearson']:.6f}"
        )

    cleanup()


if __name__ == "__main__":
    main()
