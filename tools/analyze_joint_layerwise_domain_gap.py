#!/usr/bin/env python3
"""Analyze layerwise image-domain gap for joint BRCA models."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from glip.utils import parse_bool, seed_everything
from tools.build_xenium_pseudospots import run_umap
from tools.compare_joint_embedding_umap import (
    apply_plot_limits,
    build_loaders,
    compute_alignment_metrics,
    rebuild_model_with_gene_dim,
    resolve_device,
    resolve_fold_dir,
)


STAGE_ORDER = [
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "pooled_backbone",
    "projected",
    "final_embedding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layerwise Visium/Xenium domain gap for joint models")
    parser.add_argument(
        "--run-dir-a",
        default="/data/yujk/GLIP/runs_joint/brca_stage2_naive_joint_targetbank_full_resnet50_g227_random5fold",
        help="Parent joint run directory for model A",
    )
    parser.add_argument("--label-a", default="No OT", help="Display label for model A")
    parser.add_argument(
        "--run-dir-b",
        default="/data/yujk/GLIP/runs_joint/brca_stage2_image_ot_targetbank_full_resnet50_g227_random5fold_bs64",
        help="Parent joint run directory for model B",
    )
    parser.add_argument("--label-b", default="Image OT bs64", help="Display label for model B")
    parser.add_argument("--fold-index", type=int, default=0, help="Fold index to compare")
    parser.add_argument(
        "--output-dir",
        default="/data/yujk/GLIP/runs/layerwise_domain_gap_noot_vs_ot_fold00",
        help="Directory where plots/tables are written",
    )
    parser.add_argument("--device", default="", help="Torch device, empty means auto")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="Feature extraction batch size")
    parser.add_argument("--include-test", default="true", help="Whether to include test split points")
    parser.add_argument("--max-visium-train-spots", type=int, default=2000, help="Visium train cap")
    parser.add_argument("--max-visium-test-spots", type=int, default=1000, help="Visium test cap")
    parser.add_argument("--max-xenium-train-spots", type=int, default=2000, help="Xenium train cap")
    parser.add_argument("--max-xenium-test-spots", type=int, default=1000, help="Xenium test cap")
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    return parser.parse_args()


def mean_pool_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return F.adaptive_avg_pool2d(tensor, output_size=1).flatten(1)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Unsupported tensor shape for pooling: {tuple(tensor.shape)}")


def ensure_resnet_backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone = model.base_model.image_encoder.model
    required = ["conv1", "bn1", "act1", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    if not all(hasattr(backbone, name) for name in required):
        raise TypeError("Layerwise analysis currently supports ResNet-style image backbones only.")
    return backbone


def extract_resnet_stage_embeddings(
    model: torch.nn.Module,
    images: torch.Tensor,
    platform_ids: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    backbone = ensure_resnet_backbone(model)
    x = backbone.conv1(images)
    x = backbone.bn1(x)
    x = backbone.act1(x)
    x = backbone.maxpool(x)
    outputs: Dict[str, torch.Tensor] = {"stem": mean_pool_if_needed(x)}

    x = backbone.layer1(x)
    outputs["layer1"] = mean_pool_if_needed(x)
    x = backbone.layer2(x)
    outputs["layer2"] = mean_pool_if_needed(x)
    x = backbone.layer3(x)
    outputs["layer3"] = mean_pool_if_needed(x)
    x = backbone.layer4(x)
    outputs["layer4"] = mean_pool_if_needed(x)

    pooled = backbone.global_pool(x) if hasattr(backbone, "global_pool") else mean_pool_if_needed(x)
    outputs["pooled_backbone"] = mean_pool_if_needed(pooled)

    projected = model.base_model.image_projection(outputs["pooled_backbone"])
    outputs["projected"] = projected
    outputs["final_embedding"] = model._apply_platform_token(projected, platform_ids)
    return outputs


@torch.no_grad()
def collect_layerwise_features(
    model: torch.nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    model_label: str,
    include_test: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    meta_rows: List[Dict[str, str]] = []
    stage_buffers: Dict[str, List[np.ndarray]] = {stage: [] for stage in STAGE_ORDER}

    for split_name, loader in loaders.items():
        split_suffix = "test" if split_name.endswith("test") else "train"
        if (not include_test) and split_suffix == "test":
            continue
        platform_label = "visium" if split_name.startswith("visium") else "xenium"

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            platform_ids = batch.get("platform_id")
            if platform_ids is not None:
                platform_ids = platform_ids.to(device, non_blocking=True)

            stage_outputs = extract_resnet_stage_embeddings(model, images, platform_ids)
            batch_size = images.size(0)
            for stage_name, stage_tensor in stage_outputs.items():
                stage_buffers[stage_name].append(stage_tensor.detach().cpu().numpy())

            sample_ids = [str(value) for value in batch["sample_id"]]
            barcodes = [str(value) for value in batch["barcode"]]
            for idx in range(batch_size):
                meta_rows.append(
                    {
                        "model": model_label,
                        "platform": platform_label,
                        "split": split_suffix,
                        "sample_id": sample_ids[idx],
                        "barcode": barcodes[idx],
                    }
                )

    meta_frame = pd.DataFrame(meta_rows)
    stage_arrays = {
        stage_name: np.concatenate(chunks, axis=0).astype(np.float32)
        for stage_name, chunks in stage_buffers.items()
    }
    return meta_frame, stage_arrays


def compute_platform_auc(features: np.ndarray, labels: Sequence[str], seed: int) -> float | None:
    encoded = np.asarray(pd.Categorical(labels).codes, dtype=np.int64)
    unique_codes, counts = np.unique(encoded, return_counts=True)
    if unique_codes.size != 2 or counts.min() < 2:
        return None

    n_splits = max(2, min(5, int(counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: List[float] = []
    classifier = LogisticRegression(max_iter=2000)
    for train_idx, test_idx in cv.split(features, encoded):
        classifier.fit(features[train_idx], encoded[train_idx])
        probabilities = classifier.predict_proba(features[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(encoded[test_idx], probabilities)))
    return float(np.mean(aucs)) if aucs else None


def build_stage_umap_table(
    frame: pd.DataFrame,
    stage_name: str,
    stage_features: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> Tuple[pd.DataFrame, Dict[str, float | None]]:
    umap_embedding, pca_features, umap_meta = run_umap(
        features=stage_features,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    result = frame.copy()
    result["stage"] = stage_name
    result["umap_1"] = umap_embedding[:, 0]
    result["umap_2"] = umap_embedding[:, 1]
    metrics: Dict[str, float | None] = {
        **umap_meta,
        **compute_alignment_metrics(pca_features, result["platform"].tolist()),
    }
    metrics["platform_auc"] = compute_platform_auc(pca_features, result["platform"].tolist(), seed=seed)
    return result, metrics


def save_stage_umap_plot(
    df: pd.DataFrame,
    metric_map: Dict[str, Dict[str, float | None]],
    stage_name: str,
    output_path: str,
) -> None:
    models = list(dict.fromkeys(df["model"].tolist()))
    fig, axes = plt.subplots(1, len(models), figsize=(7.2 * len(models), 6.2), squeeze=False)
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
            s=34,
            alpha=0.82,
            linewidth=0.0,
            ax=ax,
        )
        metrics = metric_map[model_name]
        auc_value = metrics.get("platform_auc")
        auc_text = "NA" if auc_value is None else f"{auc_value:.3f}"
        ax.set_title(
            f"{model_name}\n"
            f"sil={metrics['silhouette']:.3f}  centroid={metrics['centroid_distance']:.2f}\n"
            f"cos={metrics['cross_domain_mean_cosine']:.3f}  auc={auc_text}"
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    fig.suptitle(f"Layerwise Platform Gap: {stage_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_metric_curves(metrics_df: pd.DataFrame, output_path: str) -> None:
    metric_specs = [
        ("silhouette", "Silhouette"),
        ("centroid_distance", "Centroid Distance"),
        ("cross_domain_mean_cosine", "Cross-domain Mean Cosine"),
        ("platform_auc", "Platform AUC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), squeeze=False)
    axes_flat = axes.flatten()
    for ax, (metric_name, title) in zip(axes_flat, metric_specs):
        subset = metrics_df.copy()
        sns.lineplot(
            data=subset,
            x="stage",
            y=metric_name,
            hue="model",
            marker="o",
            linewidth=2.2,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Stage")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=25)
        if metric_name == "platform_auc":
            ax.set_ylim(0.45, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_metric_delta_bar(metrics_df: pd.DataFrame, output_path: str) -> None:
    records: List[Dict[str, float | str]] = []
    for model_name, group in metrics_df.groupby("model", sort=False):
        ordered = group.set_index("stage").reindex(STAGE_ORDER)
        for metric_name in ["silhouette", "centroid_distance", "cross_domain_mean_cosine", "platform_auc"]:
            start = ordered.loc["stem", metric_name]
            end = ordered.loc["final_embedding", metric_name]
            if pd.isna(start) or pd.isna(end):
                continue
            records.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "delta_final_minus_stem": float(end - start),
                }
            )
    delta_df = pd.DataFrame(records)
    plt.figure(figsize=(9.2, 5.4))
    sns.barplot(data=delta_df, x="metric", y="delta_final_minus_stem", hue="model")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Metric")
    plt.ylabel("Final - Stem")
    plt.title("How Platform Gap Changes From Stem To Final Embedding")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_json(payload: Dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.include_test = parse_bool(args.include_test)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        {"label": args.label_a, "run_dir": args.run_dir_a},
        {"label": args.label_b, "run_dir": args.run_dir_b},
    ]

    all_metric_rows: List[Dict[str, float | str | None]] = []
    summary_payload: Dict[str, Dict] = {
        "fold_index": int(args.fold_index),
        "device": str(device),
        "include_test": bool(args.include_test),
        "models": [],
        "stages": {},
    }

    for run_spec in run_specs:
        fold_dir = resolve_fold_dir(run_spec["run_dir"], args.fold_index)
        config = json.loads((fold_dir / "joint_config.json").read_text(encoding="utf-8"))
        loaders, gene_dim = build_loaders(config, args)
        model, _ = rebuild_model_with_gene_dim(fold_dir, device, gene_dim)
        meta_frame, stage_features = collect_layerwise_features(
            model=model,
            loaders=loaders,
            device=device,
            model_label=run_spec["label"],
            include_test=bool(args.include_test),
        )
        sampled_meta = apply_plot_limits(meta_frame, args, seed=args.seed)
        sampled_indices = sampled_meta.index.to_numpy()
        per_model_dir = output_dir / run_spec["label"].lower().replace(" ", "_")
        per_model_dir.mkdir(parents=True, exist_ok=True)

        summary_payload["models"].append(
            {
                "label": run_spec["label"],
                "fold_dir": str(fold_dir),
                "config_path": str(fold_dir / "joint_config.json"),
                "checkpoint_path": str(fold_dir / "best.pt"),
                "num_points": int(len(sampled_meta)),
            }
        )

        for stage_name in STAGE_ORDER:
            sampled_features = stage_features[stage_name][sampled_indices]
            stage_umap_df, stage_metrics = build_stage_umap_table(
                frame=sampled_meta.reset_index(drop=True),
                stage_name=stage_name,
                stage_features=sampled_features,
                seed=args.seed,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
            )
            stage_umap_df.to_csv(per_model_dir / f"{stage_name}_umap.csv", index=False)
            stage_record = {"model": run_spec["label"], "stage": stage_name, **stage_metrics}
            all_metric_rows.append(stage_record)
            summary_payload["stages"].setdefault(stage_name, {})[run_spec["label"]] = stage_metrics

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df["stage"] = pd.Categorical(metrics_df["stage"], categories=STAGE_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values(["stage", "model"]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "layerwise_metrics.csv", index=False)

    for stage_name in STAGE_ORDER:
        stage_metric_map = {}
        stage_umap_frames: List[pd.DataFrame] = []
        for run_spec in run_specs:
            per_model_dir = output_dir / run_spec["label"].lower().replace(" ", "_")
            stage_umap_frames.append(pd.read_csv(per_model_dir / f"{stage_name}_umap.csv"))
            stage_metrics = metrics_df.loc[
                (metrics_df["model"] == run_spec["label"]) & (metrics_df["stage"] == stage_name)
            ].iloc[0]
            stage_metric_map[run_spec["label"]] = {
                key: (None if pd.isna(value) else float(value))
                for key, value in stage_metrics.to_dict().items()
                if key not in {"model", "stage"}
            }
        stage_df = pd.concat(stage_umap_frames, axis=0, ignore_index=True)
        save_stage_umap_plot(
            df=stage_df,
            metric_map=stage_metric_map,
            stage_name=stage_name,
            output_path=str(output_dir / f"{stage_name}_umap_compare.png"),
        )

    save_metric_curves(metrics_df, str(output_dir / "layerwise_metric_curves.png"))
    save_metric_delta_bar(metrics_df, str(output_dir / "layerwise_metric_delta.png"))
    save_json(summary_payload, output_dir / "summary.json")


if __name__ == "__main__":
    main()
