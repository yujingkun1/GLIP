"""Pseudo-spot construction and dataset loading for Xenium."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from . import config as CFG
from .data import (
    _SimpleWSI,
    align_expression_from_index_map,
    build_processed_paths,
    build_target_to_source_index,
    prepare_processed_dataset,
)
from glip.utils import assign_position_folds, save_json

Image.MAX_IMAGE_PIXELS = None


@dataclass
class PseudoSpotPaths:
    output_dir: str
    counts_path: str
    metadata_path: str
    genes_path: str
    manifest_path: str
    membership_path: str


def build_pseudospot_output_dir(base_output_dir: str, xenium_sample_id: str, reference_sample_id: str) -> str:
    return os.path.join(os.path.abspath(base_output_dir), f"{xenium_sample_id}_ref_{reference_sample_id}")


def build_pseudospot_paths(output_dir: str) -> PseudoSpotPaths:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return PseudoSpotPaths(
        output_dir=output_dir,
        counts_path=os.path.join(output_dir, "pseudospot_expression_counts.npy"),
        metadata_path=os.path.join(output_dir, "pseudospot_metadata.parquet"),
        genes_path=os.path.join(output_dir, "pseudospot_genes.json"),
        manifest_path=os.path.join(output_dir, "pseudospot_manifest.json"),
        membership_path=os.path.join(output_dir, "pseudospot_membership.npz"),
    )


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def resolve_pixel_size_um(hest_data_dir: str, sample_id: str) -> float:
    metadata_path = os.path.join(hest_data_dir, "metadata", f"{sample_id}.json")
    metadata = _load_json(metadata_path)
    pixel_size_um = _parse_optional_float(metadata.get("pixel_size_um_embedded"))
    if pixel_size_um is None:
        pixel_size_um = _parse_optional_float(metadata.get("pixel_size_um_estimated"))
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError(f"Unable to resolve pixel size from {metadata_path}")
    return float(pixel_size_um)


def resolve_reference_spot_parameters(
    visium_hest_data_dir: str,
    reference_sample_id: str,
) -> Dict[str, object]:
    metadata_path = os.path.join(visium_hest_data_dir, "metadata", f"{reference_sample_id}.json")
    metadata = _load_json(metadata_path)

    spot_diameter_um = _parse_optional_float(metadata.get("spot_diameter"))
    if spot_diameter_um is None or spot_diameter_um <= 0:
        raise ValueError(f"Unable to resolve spot_diameter from {metadata_path}")

    inter_spot_dist_um = _parse_optional_float(metadata.get("inter_spot_dist"))
    if inter_spot_dist_um is None or inter_spot_dist_um <= 0:
        inter_spot_dist_um = float(spot_diameter_um)

    technology = str(metadata.get("st_technology") or "UNKNOWN")
    return {
        "reference_sample_id": reference_sample_id,
        "technology": technology,
        "spot_diameter_um": float(spot_diameter_um),
        "inter_spot_dist_um": float(inter_spot_dist_um),
        "metadata_path": metadata_path,
    }


def resolve_grid_layout(layout: str, reference_technology: str) -> str:
    normalized = str(layout).strip().lower()
    if normalized == "auto":
        if "visium" in str(reference_technology).strip().lower():
            return "hex"
        return "square"
    if normalized not in {"square", "hex"}:
        raise ValueError("grid_layout must be one of: auto, square, hex")
    return normalized


def generate_grid_centers(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    spot_radius_px: float,
    center_spacing_px: float,
    grid_layout: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = float(x_min) - float(spot_radius_px)
    x_max = float(x_max) + float(spot_radius_px)
    y_min = float(y_min) - float(spot_radius_px)
    y_max = float(y_max) + float(spot_radius_px)
    center_spacing_px = max(float(center_spacing_px), 1.0)

    centers: List[Tuple[float, float]] = []
    grid_rows: List[int] = []
    grid_cols: List[int] = []

    if grid_layout == "hex":
        y_step = center_spacing_px * math.sqrt(3.0) / 2.0
        row = 0
        y = y_min + float(spot_radius_px)
        while y <= y_max:
            x_offset = 0.0 if row % 2 == 0 else center_spacing_px / 2.0
            col = 0
            x = x_min + float(spot_radius_px) + x_offset
            while x <= x_max:
                centers.append((float(x), float(y)))
                grid_rows.append(row)
                grid_cols.append(col)
                x += center_spacing_px
                col += 1
            y += y_step
            row += 1
    else:
        row = 0
        y = y_min + float(spot_radius_px)
        while y <= y_max:
            col = 0
            x = x_min + float(spot_radius_px)
            while x <= x_max:
                centers.append((float(x), float(y)))
                grid_rows.append(row)
                grid_cols.append(col)
                x += center_spacing_px
                col += 1
            y += center_spacing_px
            row += 1

    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(grid_rows, dtype=np.int32),
        np.asarray(grid_cols, dtype=np.int32),
    )


def assign_cells_to_regular_grid(
    *,
    cell_centroids: np.ndarray,
    grid_centers: np.ndarray,
    grid_rows: np.ndarray,
    grid_cols: np.ndarray,
    center_spacing_px: float,
    spot_radius_px: float,
    grid_layout: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each cell to its nearest regular-grid center within the pseudo-spot radius."""

    if cell_centroids.size == 0 or grid_centers.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.int64)

    row_count = int(grid_rows.max()) + 1
    row_y = np.zeros(row_count, dtype=np.float32)
    row_start_x = np.zeros(row_count, dtype=np.float32)
    row_ncols = np.zeros(row_count, dtype=np.int32)

    for row_idx in range(row_count):
        row_mask = grid_rows == row_idx
        row_centers = grid_centers[row_mask]
        if row_centers.size == 0:
            continue
        row_y[row_idx] = float(row_centers[0, 1])
        row_start_x[row_idx] = float(row_centers[:, 0].min())
        row_ncols[row_idx] = int(row_centers.shape[0])

    max_cols = int(row_ncols.max()) if row_ncols.size else 0
    row_col_to_index = np.full((row_count, max_cols), -1, dtype=np.int64)
    row_col_to_index[grid_rows.astype(np.int64), grid_cols.astype(np.int64)] = np.arange(grid_centers.shape[0], dtype=np.int64)

    center_spacing_px = max(float(center_spacing_px), 1.0)
    if str(grid_layout).strip().lower() == "hex":
        row_step = center_spacing_px * math.sqrt(3.0) / 2.0
    else:
        row_step = center_spacing_px

    origin_y = float(row_y.min())
    base_row = np.rint((cell_centroids[:, 1] - origin_y) / max(row_step, 1e-6)).astype(np.int64)
    candidate_rows = np.stack([base_row - 1, base_row, base_row + 1], axis=1)
    candidate_rows_clipped = np.clip(candidate_rows, 0, max(row_count - 1, 0))
    valid_row_mask = (candidate_rows >= 0) & (candidate_rows < row_count)

    cell_x = cell_centroids[:, 0][:, None]
    cell_y = cell_centroids[:, 1][:, None]
    candidate_start_x = row_start_x[candidate_rows_clipped]
    candidate_row_y = row_y[candidate_rows_clipped]
    candidate_ncols = row_ncols[candidate_rows_clipped]

    candidate_cols = np.rint((cell_x - candidate_start_x) / center_spacing_px).astype(np.int64)
    candidate_cols = np.maximum(candidate_cols, 0)
    candidate_cols = np.minimum(candidate_cols, np.maximum(candidate_ncols.astype(np.int64) - 1, 0))

    candidate_indices = row_col_to_index[candidate_rows_clipped, candidate_cols]
    candidate_center_x = candidate_start_x + candidate_cols.astype(np.float32) * float(center_spacing_px)
    distance_sq = (cell_x - candidate_center_x) ** 2 + (cell_y - candidate_row_y) ** 2
    distance_sq[~valid_row_mask] = np.inf
    distance_sq[candidate_indices < 0] = np.inf

    best_choice = np.argmin(distance_sq, axis=1)
    best_distance_sq = distance_sq[np.arange(cell_centroids.shape[0]), best_choice]
    assigned_indices = candidate_indices[np.arange(cell_centroids.shape[0]), best_choice]
    valid_mask = np.isfinite(best_distance_sq) & (best_distance_sq <= float(spot_radius_px) ** 2) & (assigned_indices >= 0)
    return valid_mask.astype(bool, copy=False), assigned_indices.astype(np.int64, copy=False)


def maybe_downcast_counts(counts: np.ndarray) -> np.ndarray:
    max_value = int(counts.max()) if counts.size else 0
    if max_value <= np.iinfo(np.uint16).max:
        return counts.astype(np.uint16, copy=False)
    return counts.astype(np.uint32, copy=False)


def prepare_pseudospot_dataset(
    *,
    xenium_hest_data_dir: str,
    visium_hest_data_dir: str,
    processed_dir: str,
    pseudo_output_dir: str,
    xenium_sample_id: str = "NCBI784",
    reference_sample_id: str = "MEND141",
    min_cells_per_spot: int = 3,
    num_position_folds: int = 5,
    remove_control_features: bool = True,
    nucleus_only: bool = False,
    drop_zero_expression: bool = True,
    grid_layout: str = "auto",
    use_reference_inter_spot_dist: bool = True,
    force_rebuild: bool = False,
) -> PseudoSpotPaths:
    paths = build_pseudospot_paths(pseudo_output_dir)
    expected_outputs = [
        paths.counts_path,
        paths.metadata_path,
        paths.genes_path,
        paths.manifest_path,
        paths.membership_path,
    ]
    if not force_rebuild and all(os.path.exists(path) for path in expected_outputs):
        return paths

    processed_paths = build_processed_paths(output_dir=processed_dir, sample_id=xenium_sample_id)
    if force_rebuild or not (
        os.path.exists(processed_paths.counts_path)
        and os.path.exists(processed_paths.metadata_path)
        and os.path.exists(processed_paths.genes_path)
    ):
        prepare_processed_dataset(
            hest_data_dir=xenium_hest_data_dir,
            output_dir=processed_dir,
            sample_id=xenium_sample_id,
            remove_control_features=remove_control_features,
            nucleus_only=nucleus_only,
            drop_zero_expression=drop_zero_expression,
            force_rebuild=force_rebuild,
        )

    counts = np.load(processed_paths.counts_path, mmap_mode="r")
    metadata_df = pd.read_parquet(processed_paths.metadata_path)
    with open(processed_paths.genes_path, "r", encoding="utf-8") as handle:
        gene_names = list(json.load(handle)["genes"])

    reference = resolve_reference_spot_parameters(visium_hest_data_dir, reference_sample_id)
    xenium_pixel_size_um = resolve_pixel_size_um(xenium_hest_data_dir, xenium_sample_id)
    spot_radius_px = float(reference["spot_diameter_um"]) / float(xenium_pixel_size_um) / 2.0
    center_spacing_um = (
        float(reference["inter_spot_dist_um"])
        if use_reference_inter_spot_dist
        else float(reference["spot_diameter_um"])
    )
    center_spacing_px = center_spacing_um / float(xenium_pixel_size_um)
    resolved_grid_layout = resolve_grid_layout(grid_layout, str(reference["technology"]))
    default_crop_size_um = float(center_spacing_um)

    cell_centroids = metadata_df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32, copy=True)
    cell_ids = metadata_df["cell_id"].to_numpy(dtype=np.int64, copy=False)
    grid_centers, grid_rows, grid_cols = generate_grid_centers(
        x_min=float(cell_centroids[:, 0].min()),
        x_max=float(cell_centroids[:, 0].max()),
        y_min=float(cell_centroids[:, 1].min()),
        y_max=float(cell_centroids[:, 1].max()),
        spot_radius_px=spot_radius_px,
        center_spacing_px=center_spacing_px,
        grid_layout=resolved_grid_layout,
    )

    valid_mask, nearest_spot = assign_cells_to_regular_grid(
        cell_centroids=cell_centroids,
        grid_centers=grid_centers,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        center_spacing_px=center_spacing_px,
        spot_radius_px=spot_radius_px,
        grid_layout=resolved_grid_layout,
    )
    valid_cell_indices = np.flatnonzero(valid_mask).astype(np.int64)
    assigned_spot_indices = nearest_spot[valid_mask].astype(np.int64, copy=False)

    if valid_cell_indices.size == 0:
        raise RuntimeError("No Xenium cells were assigned to any pseudo-spot.")

    spot_counts_full = np.bincount(assigned_spot_indices, minlength=grid_centers.shape[0]).astype(np.int64, copy=False)
    keep_spot_mask = spot_counts_full >= max(1, int(min_cells_per_spot))
    kept_spot_indices = np.flatnonzero(keep_spot_mask).astype(np.int64)
    if kept_spot_indices.size == 0:
        raise RuntimeError("No pseudo-spots satisfied min_cells_per_spot.")

    old_to_new_spot = np.full(grid_centers.shape[0], -1, dtype=np.int64)
    old_to_new_spot[kept_spot_indices] = np.arange(kept_spot_indices.shape[0], dtype=np.int64)
    kept_cell_mask = keep_spot_mask[assigned_spot_indices]
    kept_cell_indices = valid_cell_indices[kept_cell_mask]
    kept_assigned_spots = assigned_spot_indices[kept_cell_mask]
    new_spot_indices = old_to_new_spot[kept_assigned_spots]

    sort_order = np.argsort(new_spot_indices, kind="stable")
    membership_cell_indices = kept_cell_indices[sort_order]
    membership_cell_ids = cell_ids[membership_cell_indices]
    membership_spot_indices = new_spot_indices[sort_order]
    spot_indptr = np.zeros(kept_spot_indices.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(membership_spot_indices, minlength=kept_spot_indices.shape[0]), out=spot_indptr[1:])

    pseudospot_counts = np.zeros((kept_spot_indices.shape[0], counts.shape[1]), dtype=np.uint32)
    for spot_idx in range(kept_spot_indices.shape[0]):
        start = int(spot_indptr[spot_idx])
        end = int(spot_indptr[spot_idx + 1])
        pseudospot_counts[spot_idx] = np.asarray(counts[membership_cell_indices[start:end]].sum(axis=0), dtype=np.uint32)

    pseudospot_counts = maybe_downcast_counts(pseudospot_counts)
    transcript_count = pseudospot_counts.sum(axis=1).astype(np.int64, copy=False)
    num_genes_by_counts = (pseudospot_counts > 0).sum(axis=1).astype(np.int64, copy=False)
    fold_ids, fold_edges = assign_position_folds(grid_centers[kept_spot_indices, 0], num_folds=num_position_folds)

    pseudospot_metadata = pd.DataFrame(
        {
            "spot_id": np.arange(kept_spot_indices.shape[0], dtype=np.int64),
            "center_x": grid_centers[kept_spot_indices, 0].astype(np.float32, copy=False),
            "center_y": grid_centers[kept_spot_indices, 1].astype(np.float32, copy=False),
            "grid_row": grid_rows[kept_spot_indices].astype(np.int32, copy=False),
            "grid_col": grid_cols[kept_spot_indices].astype(np.int32, copy=False),
            "cell_count": np.diff(spot_indptr).astype(np.int32, copy=False),
            "transcript_count": transcript_count,
            "n_genes_by_counts": num_genes_by_counts,
            "position_fold": fold_ids.astype(np.int16, copy=False),
        }
    )

    np.save(paths.counts_path, pseudospot_counts)
    pseudospot_metadata.to_parquet(paths.metadata_path, index=False)
    save_json({"genes": gene_names}, paths.genes_path)
    np.savez_compressed(
        paths.membership_path,
        indptr=spot_indptr,
        cell_data_indices=membership_cell_indices.astype(np.int64, copy=False),
        cell_ids=membership_cell_ids.astype(np.int64, copy=False),
        spot_indices=membership_spot_indices.astype(np.int64, copy=False),
    )
    save_json(
        {
            "xenium_sample_id": xenium_sample_id,
            "reference_sample_id": reference_sample_id,
            "reference_technology": reference["technology"],
            "xenium_hest_data_dir": os.path.abspath(xenium_hest_data_dir),
            "visium_hest_data_dir": os.path.abspath(visium_hest_data_dir),
            "processed_dir": os.path.abspath(processed_dir),
            "xenium_pixel_size_um": float(xenium_pixel_size_um),
            "reference_spot_diameter_um": float(reference["spot_diameter_um"]),
            "reference_inter_spot_dist_um": float(reference["inter_spot_dist_um"]),
            "center_spacing_um": float(center_spacing_um),
            "center_spacing_px": float(center_spacing_px),
            "spot_radius_px": float(spot_radius_px),
            "grid_layout": resolved_grid_layout,
            "use_reference_inter_spot_dist": bool(use_reference_inter_spot_dist),
            "min_cells_per_spot": int(min_cells_per_spot),
            "num_position_folds": int(num_position_folds),
            "default_crop_size_um": float(default_crop_size_um),
            "num_cells_total": int(metadata_df.shape[0]),
            "num_cells_assigned": int(kept_cell_indices.shape[0]),
            "num_pseudospots": int(pseudospot_metadata.shape[0]),
            "num_genes": int(len(gene_names)),
            "position_fold_edges": fold_edges,
            "bbox_x": [
                float(cell_centroids[:, 0].min()),
                float(cell_centroids[:, 0].max()),
            ],
            "bbox_y": [
                float(cell_centroids[:, 1].min()),
                float(cell_centroids[:, 1].max()),
            ],
            "counts_dtype": str(pseudospot_counts.dtype),
            "wsi_path": os.path.join(os.path.abspath(xenium_hest_data_dir), "wsis", f"{xenium_sample_id}.tif"),
        },
        paths.manifest_path,
    )
    return paths


class XeniumPseudoSpotDataset(Dataset):
    def __init__(
        self,
        *,
        pseudospot_dir: str,
        split: str = "train",
        test_fold: int = 4,
        num_position_folds: int = 5,
        crop_size_um: float = 0.0,
        image_size: int = CFG.IMAGE_SIZE,
        augment: bool = False,
        max_spots: int = 0,
        include_image: bool = True,
        encoder_target_gene_names: Optional[Sequence[str]] = None,
        encoder_use_raw_counts: bool = False,
    ) -> None:
        super().__init__()
        self.paths = build_pseudospot_paths(pseudospot_dir)
        if not os.path.exists(self.paths.counts_path):
            raise FileNotFoundError(f"Missing pseudo-spot counts: {self.paths.counts_path}")
        if not os.path.exists(self.paths.metadata_path):
            raise FileNotFoundError(f"Missing pseudo-spot metadata: {self.paths.metadata_path}")
        if not os.path.exists(self.paths.genes_path):
            raise FileNotFoundError(f"Missing pseudo-spot gene file: {self.paths.genes_path}")
        if not os.path.exists(self.paths.manifest_path):
            raise FileNotFoundError(f"Missing pseudo-spot manifest: {self.paths.manifest_path}")

        with open(self.paths.genes_path, "r", encoding="utf-8") as handle:
            genes_payload = json.load(handle)
        with open(self.paths.manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self.gene_names = list(genes_payload["genes"])
        self.num_features = len(self.gene_names)
        self.encoder_target_gene_names = (
            [str(gene_name) for gene_name in encoder_target_gene_names]
            if encoder_target_gene_names is not None
            else list(self.gene_names)
        )
        self.encoder_use_raw_counts = bool(encoder_use_raw_counts)
        self._encoder_target_to_source = build_target_to_source_index(self.gene_names, self.encoder_target_gene_names)
        self.image_size = int(image_size)
        self.crop_size_um = (
            float(crop_size_um)
            if crop_size_um and crop_size_um > 0
            else float(self.manifest.get("default_crop_size_um", 0.0))
        )
        if self.crop_size_um <= 0:
            raise ValueError("Pseudo-spot crop_size_um must be positive.")
        self.pixel_size_um = float(self.manifest["xenium_pixel_size_um"])
        self.wsi_path = str(self.manifest["wsi_path"])
        self.include_image = bool(include_image)
        self.augment = bool(augment)
        self._slide = None

        metadata_df = pd.read_parquet(self.paths.metadata_path)
        self.spot_ids = metadata_df["spot_id"].to_numpy(dtype=np.int64, copy=False)
        self.center_x = metadata_df["center_x"].to_numpy(dtype=np.float32, copy=False)
        self.center_y = metadata_df["center_y"].to_numpy(dtype=np.float32, copy=False)
        self.cell_count = metadata_df["cell_count"].to_numpy(dtype=np.int32, copy=False)
        self.transcript_count = metadata_df["transcript_count"].to_numpy(dtype=np.int64, copy=False)
        self.counts = np.load(self.paths.counts_path, mmap_mode="r")

        fold_ids, fold_edges = assign_position_folds(self.center_x, num_folds=num_position_folds)
        self.fold_ids = fold_ids.astype(np.int64, copy=False)
        self.fold_edges = fold_edges

        if split == "train":
            selection_mask = self.fold_ids != int(test_fold)
        elif split == "test":
            selection_mask = self.fold_ids == int(test_fold)
        elif split == "all":
            selection_mask = np.ones_like(self.fold_ids, dtype=bool)
        else:
            raise ValueError("split must be one of: train, test, all")

        self.indices = np.flatnonzero(selection_mask).astype(np.int64)
        if max_spots and max_spots > 0:
            self.indices = self.indices[: int(max_spots)]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_slide"] = None
        return state

    def _get_slide(self):
        if self._slide is None:
            try:
                import openslide

                self._slide = openslide.OpenSlide(self.wsi_path)
            except Exception:
                self._slide = _SimpleWSI(self.wsi_path)
        return self._slide

    def _read_crop(self, center_x: float, center_y: float) -> Image.Image:
        slide = self._get_slide()
        crop_size_px = max(1, int(round(self.crop_size_um / max(self.pixel_size_um, 1e-6))))
        left = int(round(float(center_x) - crop_size_px / 2.0))
        top = int(round(float(center_y) - crop_size_px / 2.0))
        region = slide.read_region((left, top), 0, (crop_size_px, crop_size_px)).convert("RGB")
        if region.size != (self.image_size, self.image_size):
            region = region.resize((self.image_size, self.image_size), Image.BILINEAR)
        return region

    def _transform(self, image: Image.Image) -> torch.Tensor:
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
            image = TF.rotate(image, float(np.random.choice([0, 90, 180, -90])))
        tensor = TF.to_tensor(image)
        return TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data_index = int(self.indices[index])
        raw_expression = self.counts[data_index].astype(np.float32, copy=False)
        expression = np.log1p(raw_expression)
        encoder_source_expression = raw_expression if self.encoder_use_raw_counts else expression
        encoder_expression = align_expression_from_index_map(
            encoder_source_expression,
            self._encoder_target_to_source,
        )

        sample = {
            "expression": torch.from_numpy(expression),
            "encoder_expression": torch.from_numpy(encoder_expression),
            "cell_id": int(self.spot_ids[data_index]),
            "spot_id": int(self.spot_ids[data_index]),
            "centroid_x": float(self.center_x[data_index]),
            "centroid_y": float(self.center_y[data_index]),
            "fold_id": int(self.fold_ids[data_index]),
            "cell_count": int(self.cell_count[data_index]),
        }
        if self.include_image:
            image = self._read_crop(self.center_x[data_index], self.center_y[data_index])
            sample["image"] = self._transform(image)
        return sample

    def close(self) -> None:
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception:
                pass
            self._slide = None

    def __del__(self) -> None:
        self.close()
