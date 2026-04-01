#!/usr/bin/env python3
"""Visualize Xenium/Visium image distributions and quantify domain gap with UMAP."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import anndata as ad
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
import umap
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from glip.xenium.data import (
    _SimpleWSI,
    build_processed_paths,
    prepare_processed_dataset,
)
from glip.utils import parse_bool, sample_indices, save_json, seed_everything
from infer_visium import load_checkpoint_model


@dataclass
class SpatialLayout:
    spot_centers: np.ndarray
    spot_radius_px: float
    pixel_size_um: float


@dataclass
class SampleRecords:
    dataframe: pd.DataFrame
    crops: List[np.ndarray]
    total_available_cells: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Xenium and Visium cell image distributions with summary stats and UMAP"
    )
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
    parser.add_argument(
        "--processed-dir",
        default="/data/yujk/GLIP/processed",
        help="Processed Xenium cache directory",
    )
    parser.add_argument("--xenium-sample-id", default="NCBI784", help="Xenium sample id")
    parser.add_argument(
        "--visium-sample-ids",
        default="SPA124",
        help="Comma-separated Visium sample ids. SPA124 is a good default because it is also breast tissue.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/yujk/GLIP/runs/domain_gap_analysis",
        help="Directory where plots and summary tables will be written",
    )
    parser.add_argument(
        "--crop-size-um",
        type=float,
        default=64.0,
        help="Physical crop width in microns. Converted to pixels separately per sample.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Final resized crop size used for handcrafted features and optional encoder features",
    )
    parser.add_argument(
        "--max-xenium-cells",
        type=int,
        default=2000,
        help="Maximum sampled Xenium cells, 0 means all available processed cells",
    )
    parser.add_argument(
        "--max-visium-cells-per-sample",
        type=int,
        default=2000,
        help="Maximum sampled assigned Visium cells per sample, 0 means all assigned cells",
    )
    parser.add_argument(
        "--montage-cells-per-group",
        type=int,
        default=16,
        help="Number of cell crops shown per sample in the montage figure",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default="/data/yujk/GLIP/runs/ncbi784_default/best_train_loss.pt",
        help="Optional GLIP checkpoint used to extract learned image features. Empty string disables encoder UMAP.",
    )
    parser.add_argument("--encoder-batch-size", type=int, default=64, help="Batch size for encoder feature extraction")
    parser.add_argument("--device", default="", help="Torch device, empty means auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--in-tissue-only",
        default="true",
        help="For Visium, only keep spots marked in-tissue before assigning CellViT cells",
    )
    parser.add_argument(
        "--assignment-radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the Visium spot radius during CellViT cell-to-spot assignment",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    return parser.parse_args()


def parse_sample_id_list(raw_value: str) -> List[str]:
    parsed = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one Visium sample id must be provided.")
    return parsed


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_pixel_size_um(metadata_path: str) -> float:
    metadata = load_json(metadata_path)
    pixel_size_um = parse_optional_float(metadata.get("pixel_size_um_embedded"))
    if pixel_size_um is None:
        pixel_size_um = parse_optional_float(metadata.get("pixel_size_um_estimated"))
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError(f"Unable to resolve pixel size from {metadata_path}")
    return float(pixel_size_um)


def resolve_visium_spatial_layout(
    *,
    hest_data_dir: str,
    sample_id: str,
    in_tissue_only: bool,
) -> SpatialLayout:
    metadata_path = os.path.join(hest_data_dir, "metadata", f"{sample_id}.json")
    st_path = os.path.join(hest_data_dir, "st", f"{sample_id}.h5ad")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing Visium metadata JSON: {metadata_path}")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"Missing Visium AnnData file: {st_path}")

    metadata = load_json(metadata_path)
    pixel_size_um = resolve_pixel_size_um(metadata_path)
    spot_diameter_um = parse_optional_float(metadata.get("spot_diameter"))
    if spot_diameter_um is None or spot_diameter_um <= 0:
        raise ValueError(f"Unable to resolve spot diameter from {metadata_path}")

    adata = ad.read_h5ad(st_path, backed="r")
    try:
        spot_mask = np.ones(int(adata.n_obs), dtype=bool)
        if in_tissue_only and "in_tissue" in adata.obs.columns:
            spot_mask &= adata.obs["in_tissue"].to_numpy().astype(np.int64, copy=False) > 0
        valid_indices = np.flatnonzero(spot_mask)
        if valid_indices.size == 0:
            raise RuntimeError(f"No eligible Visium spots found for {sample_id}")
        spot_centers = np.asarray(adata.obsm["spatial"][valid_indices], dtype=np.float32)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass

    return SpatialLayout(
        spot_centers=spot_centers,
        spot_radius_px=float(spot_diameter_um / pixel_size_um / 2.0),
        pixel_size_um=float(pixel_size_um),
    )


def resolve_segmentation_path(hest_data_dir: str, sample_id: str, segmentation_type: str) -> str:
    if segmentation_type == "xenium":
        seg_dir = os.path.join(hest_data_dir, "xenium_seg")
        candidates = [
            os.path.join(seg_dir, f"{sample_id}_xenium_cell_seg.parquet"),
            os.path.join(seg_dir, f"{sample_id}_xenium_cell_seg.geojson"),
            os.path.join(seg_dir, f"{sample_id}_xenium_cell_seg.geojson.zip"),
        ]
    elif segmentation_type == "cellvit":
        seg_dir = os.path.join(hest_data_dir, "cellvit_seg")
        candidates = [
            os.path.join(seg_dir, f"{sample_id}_cellvit_seg.parquet"),
            os.path.join(seg_dir, f"{sample_id}_cellvit_seg.geojson"),
            os.path.join(seg_dir, f"{sample_id}_cellvit_seg.geojson.zip"),
        ]
    else:
        raise ValueError(f"Unsupported segmentation type: {segmentation_type}")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Unable to find segmentation for {sample_id} under {seg_dir}")


def load_geometry_frame(seg_path: str) -> gpd.GeoDataFrame:
    if seg_path.endswith(".parquet"):
        return gpd.read_parquet(seg_path)
    if seg_path.endswith(".geojson"):
        return gpd.read_file(seg_path)
    if seg_path.endswith(".zip"):
        with zipfile.ZipFile(seg_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".geojson")]
            if not members:
                raise FileNotFoundError(f"No .geojson file found inside {seg_path}")
            with tempfile.TemporaryDirectory(prefix="domain_gap_geojson_") as tmpdir:
                extracted_path = archive.extract(members[0], path=tmpdir)
                return gpd.read_file(extracted_path)
    raise ValueError(f"Unsupported segmentation format: {seg_path}")


def open_slide(image_path: str):
    try:
        import openslide

        return openslide.OpenSlide(image_path)
    except Exception:
        return _SimpleWSI(image_path)


def read_crop_at_physical_scale(
    *,
    slide,
    center_x: float,
    center_y: float,
    pixel_size_um: float,
    crop_size_um: float,
    image_size: int,
) -> np.ndarray:
    crop_size_px = max(1, int(round(float(crop_size_um) / float(pixel_size_um))))
    left = int(round(float(center_x) - crop_size_px / 2.0))
    top = int(round(float(center_y) - crop_size_px / 2.0))
    image = slide.read_region((left, top), 0, (crop_size_px, crop_size_px)).convert("RGB")
    if image.size != (int(image_size), int(image_size)):
        image = image.resize((int(image_size), int(image_size)), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def compute_geometry_stats(geometry, pixel_size_um: float) -> Dict[str, float]:
    bounds = geometry.bounds
    bbox_width_px = float(bounds[2] - bounds[0])
    bbox_height_px = float(bounds[3] - bounds[1])
    area_px2 = float(geometry.area)
    perimeter_px = float(geometry.length)
    bbox_area_px2 = max(bbox_width_px * bbox_height_px, 1e-6)
    return {
        "area_px2": area_px2,
        "area_um2": area_px2 * float(pixel_size_um) * float(pixel_size_um),
        "perimeter_px": perimeter_px,
        "perimeter_um": perimeter_px * float(pixel_size_um),
        "bbox_width_px": bbox_width_px,
        "bbox_height_px": bbox_height_px,
        "bbox_width_um": bbox_width_px * float(pixel_size_um),
        "bbox_height_um": bbox_height_px * float(pixel_size_um),
        "aspect_ratio": bbox_width_px / max(bbox_height_px, 1e-6),
        "fill_ratio": area_px2 / bbox_area_px2,
        "equivalent_diameter_um": math.sqrt(max(area_px2, 0.0) * 4.0 / math.pi) * float(pixel_size_um),
    }


def compute_crop_stats_and_feature_vector(crop_rgb: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    crop_float = crop_rgb.astype(np.float32) / 255.0
    gray = crop_float.mean(axis=2)

    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = gray[:, 1:] - gray[:, :-1]
    grad_y[1:, :] = gray[1:, :] - gray[:-1, :]
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    stats = {
        "crop_r_mean": float(crop_float[:, :, 0].mean()),
        "crop_g_mean": float(crop_float[:, :, 1].mean()),
        "crop_b_mean": float(crop_float[:, :, 2].mean()),
        "crop_r_std": float(crop_float[:, :, 0].std()),
        "crop_g_std": float(crop_float[:, :, 1].std()),
        "crop_b_std": float(crop_float[:, :, 2].std()),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gradient_mean": float(grad_mag.mean()),
        "gradient_std": float(grad_mag.std()),
    }

    histogram_features = []
    for channel_idx in range(3):
        hist, _ = np.histogram(crop_float[:, :, channel_idx], bins=8, range=(0.0, 1.0), density=True)
        histogram_features.append(hist.astype(np.float32))
    gray_hist, _ = np.histogram(gray, bins=8, range=(0.0, 1.0), density=True)
    grad_hist, _ = np.histogram(grad_mag, bins=8, range=(0.0, 1.0), density=True)
    histogram_features.append(gray_hist.astype(np.float32))
    histogram_features.append(grad_hist.astype(np.float32))
    return stats, np.concatenate(histogram_features, axis=0)


def build_handcrafted_feature_matrix(records_df: pd.DataFrame, histogram_features: np.ndarray) -> np.ndarray:
    morphology_columns = [
        "area_um2",
        "perimeter_um",
        "bbox_width_um",
        "bbox_height_um",
        "aspect_ratio",
        "fill_ratio",
        "equivalent_diameter_um",
        "crop_r_mean",
        "crop_g_mean",
        "crop_b_mean",
        "crop_r_std",
        "crop_g_std",
        "crop_b_std",
        "gray_mean",
        "gray_std",
        "gradient_mean",
        "gradient_std",
    ]
    morphology = records_df[morphology_columns].to_numpy(dtype=np.float32, copy=True)
    return np.concatenate([morphology, histogram_features.astype(np.float32)], axis=1)


def compute_embedding_separation_metrics(features: np.ndarray, labels: Sequence[str]) -> Dict[str, Optional[float]]:
    encoded = np.asarray(pd.Categorical(labels).codes, dtype=np.int64)
    unique_codes = np.unique(encoded)
    metrics: Dict[str, Optional[float]] = {
        "silhouette": None,
        "same_label_knn_fraction_k15": None,
    }
    if features.shape[0] < 3 or unique_codes.size < 2:
        return metrics

    if features.shape[0] > unique_codes.size:
        metrics["silhouette"] = float(silhouette_score(features, encoded))

    k = min(15, features.shape[0] - 1)
    if k >= 1:
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn.fit(features)
        neighbor_indices = knn.kneighbors(return_distance=False)[:, 1:]
        same_label_fraction = np.mean(encoded[:, None] == encoded[neighbor_indices])
        metrics["same_label_knn_fraction_k15"] = float(same_label_fraction)
    return metrics


def run_umap(
    *,
    features: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    pca_dim = max(2, min(32, standardized.shape[0] - 1, standardized.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca_features = pca.fit_transform(standardized)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(int(n_neighbors), pca_features.shape[0] - 1)),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=seed,
    )
    umap_embedding = reducer.fit_transform(pca_features)
    explained_variance = float(np.sum(pca.explained_variance_ratio_))
    return umap_embedding, pca_features, {"pca_components": int(pca_dim), "pca_explained_variance_ratio_sum": explained_variance}


def save_umap_scatter(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    style_col: Optional[str],
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8.5, 7.0))
    ax = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=style_col,
        s=28,
        alpha=0.82,
        linewidth=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_crop_montage(
    *,
    records_df: pd.DataFrame,
    crops: List[np.ndarray],
    output_path: str,
    cells_per_group: int,
) -> None:
    if records_df.empty:
        return

    group_order = list(dict.fromkeys(records_df["sample_id"].tolist()))
    cols = int(math.ceil(math.sqrt(max(1, int(cells_per_group)))))
    rows_per_group = int(math.ceil(max(1, int(cells_per_group)) / cols))
    total_rows = rows_per_group * len(group_order)
    fig, axes = plt.subplots(total_rows, cols, figsize=(cols * 2.1, total_rows * 2.1))
    axes = np.asarray(axes, dtype=object).reshape(total_rows, cols)

    for group_idx, sample_id in enumerate(group_order):
        group_df = records_df.loc[records_df["sample_id"] == sample_id].head(cells_per_group)
        start_row = group_idx * rows_per_group
        technology = group_df["technology"].iloc[0]
        for plot_idx in range(rows_per_group * cols):
            row = start_row + plot_idx // cols
            col = plot_idx % cols
            ax = axes[row, col]
            ax.axis("off")
            if plot_idx < len(group_df):
                image_idx = int(group_df.iloc[plot_idx]["record_index"])
                ax.imshow(crops[image_idx])
                if plot_idx == 0:
                    ax.set_title(f"{sample_id} ({technology})", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_distribution_plots(records_df: pd.DataFrame, output_dir: str) -> None:
    metrics = [
        ("area_um2", "Cell Area (um^2)"),
        ("equivalent_diameter_um", "Equivalent Diameter (um)"),
        ("bbox_width_um", "BBox Width (um)"),
        ("bbox_height_um", "BBox Height (um)"),
        ("gray_mean", "Gray Mean"),
        ("gradient_mean", "Gradient Mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (column, label) in zip(axes.flat, metrics):
        sns.kdeplot(
            data=records_df,
            x=column,
            hue="technology",
            common_norm=False,
            fill=False,
            linewidth=2.0,
            ax=ax,
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_by_technology.png"), dpi=220)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (column, label) in zip(axes.flat, metrics):
        sns.boxplot(
            data=records_df,
            x="sample_id",
            y=column,
            hue="technology",
            dodge=False,
            fliersize=1.5,
            ax=ax,
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel(label)
        ax.set_title(label)
        if ax.legend_ is not None:
            ax.legend_.remove()
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_by_sample.png"), dpi=220)
    plt.close()


def normalize_for_encoder(batch_images: Sequence[np.ndarray]) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for image in batch_images:
        tensor = TF.to_tensor(Image.fromarray(image))
        tensor = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def extract_encoder_features(
    *,
    checkpoint_path: str,
    device: torch.device,
    crops: List[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    model, _, _, _ = load_checkpoint_model(checkpoint_path, device)
    model.eval()
    outputs: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(crops), max(1, int(batch_size))), desc="encoder_features", leave=False):
            batch_images = normalize_for_encoder(crops[start : start + max(1, int(batch_size))]).to(device, non_blocking=True)
            embeddings = model.encode_images(batch_images)
            outputs.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def summarize_records(records_df: pd.DataFrame, total_available_by_sample: Dict[str, int]) -> pd.DataFrame:
    summary = (
        records_df.groupby(["technology", "sample_id"], as_index=False)
        .agg(
            sampled_cells=("cell_id", "count"),
            area_um2_mean=("area_um2", "mean"),
            area_um2_median=("area_um2", "median"),
            diameter_um_mean=("equivalent_diameter_um", "mean"),
            bbox_width_um_mean=("bbox_width_um", "mean"),
            bbox_height_um_mean=("bbox_height_um", "mean"),
            gray_mean_mean=("gray_mean", "mean"),
            gradient_mean_mean=("gradient_mean", "mean"),
        )
    )
    summary["total_available_cells"] = summary["sample_id"].map(total_available_by_sample).astype(np.int64)
    return summary


def load_xenium_records(
    *,
    xenium_sample_id: str,
    hest_data_dir: str,
    processed_dir: str,
    crop_size_um: float,
    image_size: int,
    max_cells: int,
    seed: int,
) -> Tuple[SampleRecords, np.ndarray]:
    prepare_processed_dataset(
        hest_data_dir=hest_data_dir,
        output_dir=processed_dir,
        sample_id=xenium_sample_id,
        remove_control_features=True,
        nucleus_only=False,
        drop_zero_expression=True,
        force_rebuild=False,
    )

    processed_paths = build_processed_paths(output_dir=processed_dir, sample_id=xenium_sample_id)
    metadata_df = pd.read_parquet(processed_paths.metadata_path)
    total_available_cells = int(metadata_df.shape[0])
    selected_index = sample_indices(total_available_cells, max_cells, seed) if max_cells != 0 else np.arange(total_available_cells)
    selected_metadata = metadata_df.iloc[selected_index].reset_index(drop=True)

    seg_path = resolve_segmentation_path(hest_data_dir, xenium_sample_id, "xenium")
    seg_gdf = load_geometry_frame(seg_path)
    geometry_lookup = seg_gdf["geometry"]
    pixel_size_um = resolve_pixel_size_um(os.path.join(hest_data_dir, "metadata", f"{xenium_sample_id}.json"))
    wsi_path = os.path.join(hest_data_dir, "wsis", f"{xenium_sample_id}.tif")
    slide = open_slide(wsi_path)

    crops: List[np.ndarray] = []
    histogram_features: List[np.ndarray] = []
    records: List[Dict] = []

    try:
        iterator = tqdm(selected_metadata.itertuples(index=False), total=len(selected_metadata), desc=f"{xenium_sample_id}_cells", leave=False)
        for record_index, row in enumerate(iterator):
            geometry = geometry_lookup.loc[int(row.cell_id)]
            crop = read_crop_at_physical_scale(
                slide=slide,
                center_x=float(row.centroid_x),
                center_y=float(row.centroid_y),
                pixel_size_um=pixel_size_um,
                crop_size_um=crop_size_um,
                image_size=image_size,
            )
            geometry_stats = compute_geometry_stats(geometry, pixel_size_um)
            crop_stats, hist_feature = compute_crop_stats_and_feature_vector(crop)

            records.append(
                {
                    "record_index": int(len(crops)),
                    "sample_id": xenium_sample_id,
                    "technology": "Xenium",
                    "cell_id": int(row.cell_id),
                    "centroid_x": float(row.centroid_x),
                    "centroid_y": float(row.centroid_y),
                    "pixel_size_um": float(pixel_size_um),
                    "crop_size_um": float(crop_size_um),
                    "cell_class": "NA",
                    **geometry_stats,
                    **crop_stats,
                }
            )
            crops.append(crop)
            histogram_features.append(hist_feature)
    finally:
        try:
            slide.close()
        except Exception:
            pass

    records_df = pd.DataFrame(records)
    records_df["feature_origin"] = "handcrafted"
    feature_matrix = np.stack(histogram_features, axis=0) if histogram_features else np.zeros((0, 40), dtype=np.float32)
    return SampleRecords(dataframe=records_df, crops=crops, total_available_cells=total_available_cells), feature_matrix


def load_visium_records(
    *,
    sample_id: str,
    hest_data_dir: str,
    crop_size_um: float,
    image_size: int,
    max_cells: int,
    in_tissue_only: bool,
    radius_scale: float,
    seed: int,
) -> Tuple[SampleRecords, np.ndarray]:
    spatial_layout = resolve_visium_spatial_layout(
        hest_data_dir=hest_data_dir,
        sample_id=sample_id,
        in_tissue_only=in_tissue_only,
    )

    seg_path = resolve_segmentation_path(hest_data_dir, sample_id, "cellvit")
    seg_gdf = load_geometry_frame(seg_path)
    if "geometry" not in seg_gdf.columns:
        raise KeyError(f"Missing geometry column in {seg_path}")

    centroids = seg_gdf.geometry.centroid
    centroid_xy = np.column_stack(
        [
            centroids.x.to_numpy(dtype=np.float32),
            centroids.y.to_numpy(dtype=np.float32),
        ]
    )
    cell_ids = (
        seg_gdf["cell_id"].to_numpy(dtype=np.int64, copy=False)
        if "cell_id" in seg_gdf.columns
        else np.arange(len(seg_gdf), dtype=np.int64)
    )
    class_values = (
        seg_gdf["class"].astype(str).to_numpy(dtype=object, copy=False)
        if "class" in seg_gdf.columns
        else np.asarray(["NA"] * len(seg_gdf), dtype=object)
    )

    tree = cKDTree(np.asarray(spatial_layout.spot_centers, dtype=np.float32))
    max_distance = float(spatial_layout.spot_radius_px * max(float(radius_scale), 1e-6))
    distances, nearest = tree.query(centroid_xy, distance_upper_bound=max_distance)
    valid_mask = np.isfinite(distances) & (nearest < spatial_layout.spot_centers.shape[0])
    valid_indices = np.flatnonzero(valid_mask)
    total_available_cells = int(valid_indices.shape[0])
    if max_cells and max_cells > 0 and valid_indices.size > max_cells:
        sampled = sample_indices(valid_indices.size, max_cells, seed)
        valid_indices = valid_indices[sampled]

    wsi_path = os.path.join(hest_data_dir, "wsis", f"{sample_id}.tif")
    slide = open_slide(wsi_path)

    crops: List[np.ndarray] = []
    histogram_features: List[np.ndarray] = []
    records: List[Dict] = []

    try:
        iterator = tqdm(valid_indices.tolist(), total=len(valid_indices), desc=f"{sample_id}_cells", leave=False)
        for seg_idx in iterator:
            geometry = seg_gdf.iloc[seg_idx].geometry
            crop = read_crop_at_physical_scale(
                slide=slide,
                center_x=float(centroid_xy[seg_idx, 0]),
                center_y=float(centroid_xy[seg_idx, 1]),
                pixel_size_um=spatial_layout.pixel_size_um,
                crop_size_um=crop_size_um,
                image_size=image_size,
            )
            geometry_stats = compute_geometry_stats(geometry, spatial_layout.pixel_size_um)
            crop_stats, hist_feature = compute_crop_stats_and_feature_vector(crop)

            records.append(
                {
                    "record_index": int(len(crops)),
                    "sample_id": sample_id,
                    "technology": "Visium",
                    "cell_id": int(cell_ids[seg_idx]),
                    "centroid_x": float(centroid_xy[seg_idx, 0]),
                    "centroid_y": float(centroid_xy[seg_idx, 1]),
                    "pixel_size_um": float(spatial_layout.pixel_size_um),
                    "crop_size_um": float(crop_size_um),
                    "cell_class": str(class_values[seg_idx]),
                    **geometry_stats,
                    **crop_stats,
                }
            )
            crops.append(crop)
            histogram_features.append(hist_feature)
    finally:
        try:
            slide.close()
        except Exception:
            pass

    records_df = pd.DataFrame(records)
    records_df["feature_origin"] = "handcrafted"
    feature_matrix = np.stack(histogram_features, axis=0) if histogram_features else np.zeros((0, 40), dtype=np.float32)
    return SampleRecords(dataframe=records_df, crops=crops, total_available_cells=total_available_cells), feature_matrix


def main() -> None:
    args = parse_args()
    args.visium_sample_ids = parse_sample_id_list(args.visium_sample_ids)
    args.in_tissue_only = parse_bool(args.in_tissue_only)
    args.encoder_checkpoint = os.path.abspath(os.path.expanduser(args.encoder_checkpoint.strip())) if args.encoder_checkpoint else ""
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    sns.set_theme(style="whitegrid", context="talk")

    all_records: List[pd.DataFrame] = []
    all_crops: List[np.ndarray] = []
    all_histogram_features: List[np.ndarray] = []
    total_available_by_sample: Dict[str, int] = {}

    xenium_records, xenium_hist = load_xenium_records(
        xenium_sample_id=args.xenium_sample_id,
        hest_data_dir=args.xenium_hest_data_dir,
        processed_dir=args.processed_dir,
        crop_size_um=args.crop_size_um,
        image_size=args.image_size,
        max_cells=args.max_xenium_cells,
        seed=args.seed,
    )
    total_available_by_sample[args.xenium_sample_id] = xenium_records.total_available_cells
    all_records.append(xenium_records.dataframe.copy())
    all_crops.extend(xenium_records.crops)
    all_histogram_features.append(xenium_hist)

    record_offset = len(xenium_records.crops)
    for visium_idx, sample_id in enumerate(args.visium_sample_ids):
        visium_records, visium_hist = load_visium_records(
            sample_id=sample_id,
            hest_data_dir=args.visium_hest_data_dir,
            crop_size_um=args.crop_size_um,
            image_size=args.image_size,
            max_cells=args.max_visium_cells_per_sample,
            in_tissue_only=args.in_tissue_only,
            radius_scale=args.assignment_radius_scale,
            seed=args.seed + 1000 + visium_idx,
        )
        visium_df = visium_records.dataframe.copy()
        visium_df["record_index"] += record_offset
        record_offset += len(visium_records.crops)

        total_available_by_sample[sample_id] = visium_records.total_available_cells
        all_records.append(visium_df)
        all_crops.extend(visium_records.crops)
        all_histogram_features.append(visium_hist)

    records_df = pd.concat(all_records, axis=0, ignore_index=True)
    histogram_features = np.concatenate(all_histogram_features, axis=0)
    handcrafted_features = build_handcrafted_feature_matrix(records_df, histogram_features)

    records_df.to_csv(os.path.join(args.output_dir, "cell_records.csv"), index=False)
    sample_summary_df = summarize_records(records_df, total_available_by_sample)
    sample_summary_df.to_csv(os.path.join(args.output_dir, "sample_summary.csv"), index=False)

    save_crop_montage(
        records_df=records_df,
        crops=all_crops,
        output_path=os.path.join(args.output_dir, "cell_crop_montage.png"),
        cells_per_group=args.montage_cells_per_group,
    )
    save_distribution_plots(records_df, args.output_dir)

    summary_payload: Dict[str, object] = {
        "xenium_sample_id": args.xenium_sample_id,
        "visium_sample_ids": args.visium_sample_ids,
        "crop_size_um": float(args.crop_size_um),
        "image_size": int(args.image_size),
        "total_available_by_sample": {key: int(value) for key, value in total_available_by_sample.items()},
        "sampled_by_sample": {
            str(sample_id): int(count)
            for sample_id, count in records_df.groupby("sample_id")["cell_id"].count().to_dict().items()
        },
        "feature_spaces": {},
    }

    handcrafted_umap, handcrafted_pca, handcrafted_meta = run_umap(
        features=handcrafted_features,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    records_df["handcrafted_umap_1"] = handcrafted_umap[:, 0]
    records_df["handcrafted_umap_2"] = handcrafted_umap[:, 1]
    records_df["feature_origin"] = "handcrafted"
    save_umap_scatter(
        df=records_df,
        x_col="handcrafted_umap_1",
        y_col="handcrafted_umap_2",
        hue_col="technology",
        style_col="sample_id",
        title="Handcrafted Feature UMAP by Technology",
        output_path=os.path.join(args.output_dir, "umap_handcrafted_by_technology.png"),
    )
    save_umap_scatter(
        df=records_df,
        x_col="handcrafted_umap_1",
        y_col="handcrafted_umap_2",
        hue_col="sample_id",
        style_col="technology",
        title="Handcrafted Feature UMAP by Sample",
        output_path=os.path.join(args.output_dir, "umap_handcrafted_by_sample.png"),
    )
    summary_payload["feature_spaces"]["handcrafted"] = {
        **handcrafted_meta,
        "technology_separation": compute_embedding_separation_metrics(handcrafted_pca, records_df["technology"].tolist()),
        "sample_separation": compute_embedding_separation_metrics(handcrafted_pca, records_df["sample_id"].tolist()),
    }

    if args.encoder_checkpoint:
        encoder_features = extract_encoder_features(
            checkpoint_path=args.encoder_checkpoint,
            device=resolve_device(args.device),
            crops=all_crops,
            batch_size=args.encoder_batch_size,
        )
        encoder_umap, encoder_pca, encoder_meta = run_umap(
            features=encoder_features,
            seed=args.seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        records_df["encoder_umap_1"] = encoder_umap[:, 0]
        records_df["encoder_umap_2"] = encoder_umap[:, 1]
        save_umap_scatter(
            df=records_df,
            x_col="encoder_umap_1",
            y_col="encoder_umap_2",
            hue_col="technology",
            style_col="sample_id",
            title="Checkpoint Image Feature UMAP by Technology",
            output_path=os.path.join(args.output_dir, "umap_encoder_by_technology.png"),
        )
        save_umap_scatter(
            df=records_df,
            x_col="encoder_umap_1",
            y_col="encoder_umap_2",
            hue_col="sample_id",
            style_col="technology",
            title="Checkpoint Image Feature UMAP by Sample",
            output_path=os.path.join(args.output_dir, "umap_encoder_by_sample.png"),
        )
        summary_payload["feature_spaces"]["encoder"] = {
            **encoder_meta,
            "checkpoint_path": args.encoder_checkpoint,
            "technology_separation": compute_embedding_separation_metrics(encoder_pca, records_df["technology"].tolist()),
            "sample_separation": compute_embedding_separation_metrics(encoder_pca, records_df["sample_id"].tolist()),
        }
    else:
        summary_payload["feature_spaces"]["encoder"] = {
            "enabled": False,
        }

    records_df.to_csv(os.path.join(args.output_dir, "cell_records_with_umap.csv"), index=False)
    save_json(summary_payload, os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
