#!/usr/bin/env python3
"""Compare joint-model platform alignment with UMAP for image/gene embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from glip.utils import parse_bool, seed_everything
from tools.build_xenium_pseudospots import compute_embedding_separation_metrics, run_umap
from train_joint_brca_naive import (
    WrappedVisiumDataset,
    WrappedXeniumPseudoSpotDataset,
    build_model,
    build_visium_subsets,
    build_xenium_datasets,
    create_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare joint embedding UMAPs for no-OT vs OT/UOT models")
    parser.add_argument(
        "--run-dir-a",
        default="/data/yujk/GLIP/runs_joint/brca_stage2_naive_joint_targetbank_full_resnet50_g227_random5fold",
        help="Parent joint run directory for model A",
    )
    parser.add_argument(
        "--label-a",
        default="No OT",
        help="Display label for model A",
    )
    parser.add_argument(
        "--run-dir-b",
        default="/data/yujk/GLIP/runs_joint/brca_stage2_image_ot_targetbank_full_resnet50_g227_random5fold_bs64_uot_rho2p0",
        help="Parent joint run directory for model B",
    )
    parser.add_argument(
        "--label-b",
        default="Image UOT",
        help="Display label for model B",
    )
    parser.add_argument("--fold-index", type=int, default=0, help="Fold index to compare")
    parser.add_argument(
        "--output-dir",
        default="/data/yujk/GLIP/runs/embedding_umap_compare",
        help="Directory where plots/tables are written",
    )
    parser.add_argument(
        "--visium-hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data",
        help="HEST Visium root",
    )
    parser.add_argument(
        "--xenium-hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data_Xenium",
        help="HEST Xenium root",
    )
    parser.add_argument(
        "--pseudo-output-base-dir",
        default="/data/yujk/GLIP/processed/pseudospots",
        help="Pseudo-spot cache base directory",
    )
    parser.add_argument("--device", default="", help="Torch device, empty means auto")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="Embedding extraction batch size")
    parser.add_argument("--include-test", default="false", help="Whether to include test split points in UMAP")
    parser.add_argument("--max-visium-train-spots", type=int, default=3000, help="Visium train cap for plotting")
    parser.add_argument("--max-visium-test-spots", type=int, default=1500, help="Visium test cap for plotting")
    parser.add_argument("--max-xenium-train-spots", type=int, default=0, help="Xenium train cap for plotting")
    parser.add_argument("--max-xenium-test-spots", type=int, default=0, help="Xenium test cap for plotting")
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def resolve_device(raw_device: str) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_fold_dir(run_dir: str, fold_index: int) -> Path:
    run_path = Path(run_dir)
    fold_path = run_path / f"fold_{int(fold_index):02d}"
    if fold_path.is_dir():
        return fold_path
    if (run_path / "joint_config.json").is_file():
        return run_path
    raise FileNotFoundError(f"Unable to find fold directory under {run_dir} for fold {fold_index}")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_dataset_args(config: Dict, cli_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        visium_hest_data_dir=cli_args.visium_hest_data_dir,
        xenium_hest_data_dir=cli_args.xenium_hest_data_dir,
        pseudo_output_base_dir=cli_args.pseudo_output_base_dir,
        visium_sample_ids=config["visium_sample_ids"],
        visium_heldout_sample=config.get("visium_heldout_sample", ""),
        visium_fold_manifest=config.get("visium_fold_manifest", ""),
        visium_fold_index=int(config.get("visium_fold_index", -1)),
        xenium_sample_id=config["xenium_sample_id"],
        xenium_reference_visium_sample_id=config["xenium_reference_visium_sample_id"],
        shared_gene_file=config["shared_gene_file"],
        xenium_test_fold=int(config["xenium_test_fold"]),
        xenium_num_position_folds=int(config["xenium_num_position_folds"]),
        image_size=int(config.get("image_size", 224)),
        max_spots_per_sample=0,
        max_visium_train_spots=int(cli_args.max_visium_train_spots),
        max_visium_test_spots=int(cli_args.max_visium_test_spots),
        max_xenium_train_spots=int(cli_args.max_xenium_train_spots),
        max_xenium_test_spots=int(cli_args.max_xenium_test_spots),
    )


def load_joint_model(fold_dir: Path, device: torch.device, gene_dim: int) -> Tuple[torch.nn.Module, Dict]:
    config = load_json(fold_dir / "joint_config.json")
    checkpoint = torch.load(fold_dir / "best.pt", map_location=device)
    model = build_model(
        config["model"],
        config["resolved_model_name"],
        spot_embedding_dim=int(gene_dim),
        pretrained=bool(config["pretrained"]),
        checkpoint_path=config["image_encoder_checkpoint"],
        use_platform_token=bool(config["module_platform_token"]),
        use_image_ot=bool(config["module_image_ot"]),
        use_gene_ot=bool(config["module_gene_ot"]),
        ot_transport=str(config.get("ot_transport", "ot")),
        ot_image_weight=float(config["ot_image_weight"]),
        ot_gene_weight=float(config["ot_gene_weight"]),
        ot_sinkhorn_eps=float(config["ot_sinkhorn_eps"]),
        ot_sinkhorn_iters=int(config["ot_sinkhorn_iters"]),
        uot_marginal_weight=float(config.get("uot_marginal_weight", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


def rebuild_model_with_gene_dim(fold_dir: Path, device: torch.device, gene_dim: int) -> Tuple[torch.nn.Module, Dict]:
    config = load_json(fold_dir / "joint_config.json")
    checkpoint = torch.load(fold_dir / "best.pt", map_location=device)
    model = build_model(
        config["model"],
        config["resolved_model_name"],
        spot_embedding_dim=int(gene_dim),
        pretrained=bool(config["pretrained"]),
        checkpoint_path=config["image_encoder_checkpoint"],
        use_platform_token=bool(config["module_platform_token"]),
        use_image_ot=bool(config["module_image_ot"]),
        use_gene_ot=bool(config["module_gene_ot"]),
        ot_transport=str(config.get("ot_transport", "ot")),
        ot_image_weight=float(config["ot_image_weight"]),
        ot_gene_weight=float(config["ot_gene_weight"]),
        ot_sinkhorn_eps=float(config["ot_sinkhorn_eps"]),
        ot_sinkhorn_iters=int(config["ot_sinkhorn_iters"]),
        uot_marginal_weight=float(config.get("uot_marginal_weight", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


def build_loaders(config: Dict, cli_args: argparse.Namespace) -> Tuple[Dict[str, torch.utils.data.DataLoader], int]:
    dataset_args = build_dataset_args(config, cli_args)
    visium_dataset, vis_train_indices, vis_test_indices, _, _, _ = build_visium_subsets(dataset_args, dataset_args.shared_gene_file)
    x_train_ds, x_train_eval_ds, x_test_ds, shared_genes = build_xenium_datasets(dataset_args, dataset_args.shared_gene_file)

    loaders = {
        "visium_train": create_loader(
            WrappedVisiumDataset(torch.utils.data.Subset(visium_dataset, vis_train_indices.tolist()), "visium_train", platform_id=0),
            cli_args.eval_batch_size,
            0,
            shuffle=False,
        ),
        "visium_test": create_loader(
            WrappedVisiumDataset(torch.utils.data.Subset(visium_dataset, vis_test_indices.tolist()), "visium_test", platform_id=0),
            cli_args.eval_batch_size,
            0,
            shuffle=False,
        ),
        "xenium_train": create_loader(
            WrappedXeniumPseudoSpotDataset(x_train_eval_ds, "xenium_train", platform_id=1),
            cli_args.eval_batch_size,
            0,
            shuffle=False,
        ),
        "xenium_test": create_loader(
            WrappedXeniumPseudoSpotDataset(x_test_ds, "xenium_test", platform_id=1),
            cli_args.eval_batch_size,
            0,
            shuffle=False,
        ),
    }
    return loaders, len(shared_genes)


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_label: str,
    split_name: str,
    platform_label: str,
) -> pd.DataFrame:
    image_embeddings: List[np.ndarray] = []
    gene_embeddings: List[np.ndarray] = []
    barcodes: List[str] = []
    sample_ids: List[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        expressions = batch["reduced_expression"].to(device, non_blocking=True)
        platform_ids = batch.get("platform_id")
        if platform_ids is not None:
            platform_ids = platform_ids.to(device, non_blocking=True)
        image_batch = model.encode_images(images, platform_ids=platform_ids).detach().cpu().numpy()
        gene_batch = model.encode_spots(expressions, platform_ids=platform_ids).detach().cpu().numpy()
        image_embeddings.append(image_batch)
        gene_embeddings.append(gene_batch)
        barcodes.extend([str(value) for value in batch["barcode"]])
        sample_ids.extend([str(value) for value in batch["sample_id"]])

    image_array = np.concatenate(image_embeddings, axis=0)
    gene_array = np.concatenate(gene_embeddings, axis=0)
    frame = pd.DataFrame(
        {
            "model": model_label,
            "split": split_name,
            "platform": platform_label,
            "sample_id": sample_ids,
            "barcode": barcodes,
            "image_embedding": list(image_array),
            "gene_embedding": list(gene_array),
        }
    )
    return frame


def subsample_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    sampled_parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    limits = {
        ("visium", "train"): None,
        ("visium", "test"): None,
        ("xenium", "train"): None,
        ("xenium", "test"): None,
    }
    return frame.reset_index(drop=True)


def apply_plot_limits(frame: pd.DataFrame, cli_args: argparse.Namespace, seed: int) -> pd.DataFrame:
    sampled_parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    limit_map = {
        ("visium", "train"): int(cli_args.max_visium_train_spots),
        ("visium", "test"): int(cli_args.max_visium_test_spots),
        ("xenium", "train"): int(cli_args.max_xenium_train_spots),
        ("xenium", "test"): int(cli_args.max_xenium_test_spots),
    }
    for (platform, split), group in frame.groupby(["platform", "split"], sort=False):
        limit = limit_map[(platform, split)]
        if limit and len(group) > limit:
            chosen = np.sort(rng.choice(group.index.to_numpy(), size=limit, replace=False))
            sampled_parts.append(group.loc[chosen])
        else:
            sampled_parts.append(group)
    sampled = pd.concat(sampled_parts, axis=0, ignore_index=True)
    return sampled.reset_index(drop=True)


def compute_alignment_metrics(features: np.ndarray, platform_labels: Sequence[str]) -> Dict[str, float | None]:
    metrics = compute_embedding_separation_metrics(features, platform_labels)
    encoded = np.asarray(pd.Categorical(platform_labels).codes, dtype=np.int64)
    vis_mask = encoded == 0
    xen_mask = encoded == 1
    if vis_mask.any() and xen_mask.any():
        vis = np.asarray(features[vis_mask], dtype=np.float32)
        xen = np.asarray(features[xen_mask], dtype=np.float32)
        vis_mean = vis.mean(axis=0)
        xen_mean = xen.mean(axis=0)
        metrics["centroid_distance"] = float(np.linalg.norm(vis_mean - xen_mean))
        similarity = cosine_similarity(F.normalize(torch.from_numpy(vis).float(), dim=1).numpy(), F.normalize(torch.from_numpy(xen).float(), dim=1).numpy())
        metrics["cross_domain_mean_cosine"] = float(similarity.mean())
    else:
        metrics["centroid_distance"] = None
        metrics["cross_domain_mean_cosine"] = None
    return metrics


def build_umap_table(
    frame: pd.DataFrame,
    embedding_column: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> Tuple[pd.DataFrame, Dict]:
    features = np.stack(frame[embedding_column].to_numpy(), axis=0).astype(np.float32)
    umap_embedding, pca_features, umap_meta = run_umap(
        features=features,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    result = frame.drop(columns=["image_embedding", "gene_embedding"]).copy()
    result["umap_1"] = umap_embedding[:, 0]
    result["umap_2"] = umap_embedding[:, 1]
    metrics = {
        **umap_meta,
        **compute_alignment_metrics(pca_features, result["platform"].tolist()),
    }
    return result, metrics


def save_comparison_plot(
    df: pd.DataFrame,
    metric_map: Dict[str, Dict],
    output_path: str,
    title: str,
) -> None:
    models = list(dict.fromkeys(df["model"].tolist()))
    fig, axes = plt.subplots(1, len(models), figsize=(7.2 * len(models), 6.4), squeeze=False)
    palette = {"visium": "#3566B8", "xenium": "#D65F5F"}
    markers = {"train": "o", "test": "X"}
    for ax, model_name in zip(axes.flat, models):
        subset = df.loc[df["model"] == model_name].copy()
        sns.scatterplot(
            data=subset,
            x="umap_1",
            y="umap_2",
            hue="platform",
            style="split",
            palette=palette,
            markers=markers,
            s=26,
            alpha=0.78,
            linewidth=0.0,
            ax=ax,
        )
        stats = metric_map[model_name]
        ax.set_title(
            f"{model_name}\n"
            f"sil={stats['silhouette']:.3f}  knn={stats['same_label_knn_fraction_k15']:.3f}\n"
            f"centroid={stats['centroid_distance']:.3f}  cross-cos={stats['cross_domain_mean_cosine']:.3f}"
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title)
    plt.tight_layout(rect=(0.0, 0.0, 0.92, 0.96))
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.include_test = parse_bool(args.include_test)
    seed_everything(args.seed)
    sns.set_theme(style="whitegrid", context="talk")
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)

    model_specs = [
        (args.label_a, resolve_fold_dir(args.run_dir_a, args.fold_index)),
        (args.label_b, resolve_fold_dir(args.run_dir_b, args.fold_index)),
    ]

    all_frames: List[pd.DataFrame] = []
    summary_payload = {
        "fold_index": int(args.fold_index),
        "models": [],
    }

    for model_label, fold_dir in model_specs:
        config = load_json(fold_dir / "joint_config.json")
        loaders, gene_dim = build_loaders(config, args)
        model, _ = load_joint_model(fold_dir, device, gene_dim)
        frame_parts = [
            collect_embeddings(model, loaders["visium_train"], device, model_label, "train", "visium"),
            collect_embeddings(model, loaders["xenium_train"], device, model_label, "train", "xenium"),
        ]
        if args.include_test:
            frame_parts.extend(
                [
                    collect_embeddings(model, loaders["visium_test"], device, model_label, "test", "visium"),
                    collect_embeddings(model, loaders["xenium_test"], device, model_label, "test", "xenium"),
                ]
            )
        model_frame = pd.concat(frame_parts, axis=0, ignore_index=True)
        model_frame = apply_plot_limits(model_frame, args, seed=args.seed)
        all_frames.append(model_frame)
        summary_payload["models"].append(
            {
                "label": model_label,
                "fold_dir": str(fold_dir),
                "config_path": str(fold_dir / "joint_config.json"),
                "checkpoint_path": str(fold_dir / "best.pt"),
            }
        )

    combined = pd.concat(all_frames, axis=0, ignore_index=True)

    image_tables: List[pd.DataFrame] = []
    gene_tables: List[pd.DataFrame] = []
    image_metrics: Dict[str, Dict] = {}
    gene_metrics: Dict[str, Dict] = {}

    for model_label in [args.label_a, args.label_b]:
        subset = combined.loc[combined["model"] == model_label].reset_index(drop=True)
        image_df, img_metrics = build_umap_table(
            subset,
            embedding_column="image_embedding",
            seed=args.seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        gene_df, gen_metrics = build_umap_table(
            subset,
            embedding_column="gene_embedding",
            seed=args.seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        image_tables.append(image_df)
        gene_tables.append(gene_df)
        image_metrics[model_label] = img_metrics
        gene_metrics[model_label] = gen_metrics

    image_output = pd.concat(image_tables, axis=0, ignore_index=True)
    gene_output = pd.concat(gene_tables, axis=0, ignore_index=True)

    image_output.to_csv(os.path.join(args.output_dir, "image_embedding_umap.csv"), index=False)
    gene_output.to_csv(os.path.join(args.output_dir, "gene_embedding_umap.csv"), index=False)

    save_comparison_plot(
        image_output,
        image_metrics,
        os.path.join(args.output_dir, "image_embedding_umap.png"),
        title=f"Image Embedding UMAP Comparison (fold_{args.fold_index:02d})",
    )
    save_comparison_plot(
        gene_output,
        gene_metrics,
        os.path.join(args.output_dir, "gene_embedding_umap.png"),
        title=f"Gene Embedding UMAP Comparison (fold_{args.fold_index:02d})",
    )

    summary_payload["image_embedding"] = image_metrics
    summary_payload["gene_embedding"] = gene_metrics
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print("Saved outputs to", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
