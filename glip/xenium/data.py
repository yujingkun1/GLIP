"""NCBI784 Xenium preprocessing and dataset code."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import openslide
import pandas as pd
import pyarrow.parquet as pq
import shapely.wkb as shapely_wkb
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from . import config as CFG
from glip.utils import assign_position_folds, save_json

Image.MAX_IMAGE_PIXELS = None


@dataclass
class ProcessedPaths:
    sample_id: str
    output_dir: str
    counts_path: str
    metadata_path: str
    genes_path: str
    manifest_path: str


def build_processed_paths(output_dir: str, sample_id: str) -> ProcessedPaths:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return ProcessedPaths(
        sample_id=sample_id,
        output_dir=output_dir,
        counts_path=os.path.join(output_dir, f"{sample_id}_expression_counts.npy"),
        metadata_path=os.path.join(output_dir, f"{sample_id}_cell_metadata.parquet"),
        genes_path=os.path.join(output_dir, f"{sample_id}_genes.json"),
        manifest_path=os.path.join(output_dir, f"{sample_id}_manifest.json"),
    )


def decode_feature_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def is_control_feature(feature_name: str) -> bool:
    return any(feature_name.startswith(prefix) for prefix in CFG.CONTROL_FEATURE_PREFIXES)


def _decode_h5ad_string_array(values) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _load_gene_names_from_h5ad(h5ad_path: str) -> List[str]:
    with h5py.File(h5ad_path, "r") as handle:
        if "var/_index" not in handle:
            raise KeyError(f"Missing var/_index in {h5ad_path}")
        return _decode_h5ad_string_array(handle["var/_index"][:])


def load_gene_names_from_tsv(tsv_path: str, column: str = "gene_name") -> List[str]:
    gene_df = pd.read_csv(tsv_path, sep="	")
    if column in gene_df.columns:
        return [str(gene_name) for gene_name in gene_df[column].tolist()]
    first_column = gene_df.columns[0]
    return [str(gene_name) for gene_name in gene_df[first_column].tolist()]


def build_target_to_source_index(source_gene_names: Sequence[str], target_gene_names: Sequence[str]) -> np.ndarray:
    source_index = {str(gene_name): idx for idx, gene_name in enumerate(source_gene_names)}
    target_to_source = np.full(len(target_gene_names), -1, dtype=np.int64)
    for target_idx, gene_name in enumerate(target_gene_names):
        source_idx = source_index.get(str(gene_name))
        if source_idx is not None:
            target_to_source[target_idx] = source_idx
    return target_to_source


def align_expression_from_index_map(expression: np.ndarray, target_to_source_index: np.ndarray) -> np.ndarray:
    expression = np.asarray(expression, dtype=np.float32)
    aligned = np.zeros(target_to_source_index.shape[0], dtype=np.float32)
    valid_mask = target_to_source_index >= 0
    if valid_mask.any():
        aligned[valid_mask] = expression[target_to_source_index[valid_mask]]
    return aligned


def resolve_gene_panel(
    hest_data_dir: str,
    sample_id: str,
    remove_control_features: bool,
) -> List[str]:
    h5ad_path = os.path.join(hest_data_dir, "st", f"{sample_id}.h5ad")
    transcripts_path = os.path.join(hest_data_dir, "transcripts", f"{sample_id}_transcripts.parquet")

    gene_names = None
    if os.path.exists(h5ad_path):
        try:
            gene_names = _load_gene_names_from_h5ad(h5ad_path)
        except Exception as exc:
            if not os.path.exists(transcripts_path):
                raise RuntimeError(
                    f"Failed to read gene names from {h5ad_path} and no transcript parquet fallback exists at {transcripts_path}"
                ) from exc
            print(f"Warning: failed to read {h5ad_path} via anndata ({exc}); falling back to transcript parquet.")

    if gene_names is None:
        parquet_file = pq.ParquetFile(transcripts_path)
        gene_names = []
        seen = set()
        for row_group_idx in range(parquet_file.num_row_groups):
            feature_list = parquet_file.read_row_group(row_group_idx, columns=["feature_name"]).column(0).to_pylist()
            for feature_name in feature_list:
                feature_name = decode_feature_name(feature_name)
                if feature_name in seen:
                    continue
                seen.add(feature_name)
                gene_names.append(feature_name)

    if remove_control_features:
        gene_names = [gene_name for gene_name in gene_names if not is_control_feature(gene_name)]
    return gene_names


def load_xenium_segmentation_metadata(segmentation_path: str) -> pd.DataFrame:
    seg_df = pd.read_parquet(segmentation_path)
    if "geometry" not in seg_df.columns:
        raise KeyError(f"Expected a geometry column in {segmentation_path}")

    cell_ids = seg_df.index.to_numpy(dtype=np.int64, copy=False)
    centroid_x = np.zeros(cell_ids.shape[0], dtype=np.float32)
    centroid_y = np.zeros(cell_ids.shape[0], dtype=np.float32)
    min_x = np.zeros(cell_ids.shape[0], dtype=np.float32)
    min_y = np.zeros(cell_ids.shape[0], dtype=np.float32)
    max_x = np.zeros(cell_ids.shape[0], dtype=np.float32)
    max_y = np.zeros(cell_ids.shape[0], dtype=np.float32)

    for idx, geometry in enumerate(seg_df["geometry"].tolist()):
        geom = shapely_wkb.loads(geometry) if isinstance(geometry, (bytes, bytearray)) else geometry
        centroid = geom.centroid
        bounds = geom.bounds
        centroid_x[idx] = float(centroid.x)
        centroid_y[idx] = float(centroid.y)
        min_x[idx] = float(bounds[0])
        min_y[idx] = float(bounds[1])
        max_x[idx] = float(bounds[2])
        max_y[idx] = float(bounds[3])

    return pd.DataFrame(
        {
            "cell_id": cell_ids.astype(np.int64),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "bbox_min_x": min_x,
            "bbox_min_y": min_y,
            "bbox_max_x": max_x,
            "bbox_max_y": max_y,
        }
    )


def aggregate_transcripts_to_counts(
    transcripts_path: str,
    metadata_df: pd.DataFrame,
    gene_names: Sequence[str],
    *,
    nucleus_only: bool = False,
) -> np.ndarray:
    cell_ids = metadata_df["cell_id"].to_numpy(dtype=np.int64, copy=False)
    if cell_ids.size == 0:
        return np.zeros((0, len(gene_names)), dtype=np.uint32)

    max_cell_id = int(cell_ids.max())
    cell_id_to_row = np.full(max_cell_id + 1, -1, dtype=np.int64)
    cell_id_to_row[cell_ids] = np.arange(cell_ids.shape[0], dtype=np.int64)

    gene_to_index = {gene_name: idx for idx, gene_name in enumerate(gene_names)}
    counts = np.zeros((cell_ids.shape[0], len(gene_names)), dtype=np.uint32)

    parquet_file = pq.ParquetFile(transcripts_path)
    columns = ["cell_id", "feature_name"]
    if nucleus_only:
        columns.append("overlaps_nucleus")

    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=columns)
        df = table.to_pandas()
        if df.empty:
            continue

        transcript_cell_ids = pd.to_numeric(df["cell_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        valid_mask = transcript_cell_ids > 0

        if nucleus_only:
            nucleus_mask = pd.to_numeric(df["overlaps_nucleus"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
            valid_mask &= nucleus_mask

        feature_values = df["feature_name"]
        local_feature_map = {}
        for raw_feature_value in pd.unique(feature_values):
            decoded_name = decode_feature_name(raw_feature_value)
            local_feature_map[raw_feature_value] = gene_to_index.get(decoded_name, -1)
        feature_indices = feature_values.map(local_feature_map).fillna(-1).astype(np.int64).to_numpy()
        valid_mask &= feature_indices >= 0

        if not valid_mask.any():
            continue

        valid_cell_ids = transcript_cell_ids[valid_mask]
        within_segmentation_mask = valid_cell_ids <= max_cell_id
        if not within_segmentation_mask.any():
            continue

        valid_cell_ids = valid_cell_ids[within_segmentation_mask]
        valid_feature_indices = feature_indices[valid_mask][within_segmentation_mask]
        row_indices = cell_id_to_row[valid_cell_ids]
        in_metadata_mask = row_indices >= 0
        if not in_metadata_mask.any():
            continue

        np.add.at(
            counts,
            (row_indices[in_metadata_mask], valid_feature_indices[in_metadata_mask]),
            1,
        )

    return counts


def maybe_downcast_counts(counts: np.ndarray) -> np.ndarray:
    max_value = int(counts.max()) if counts.size else 0
    if max_value <= np.iinfo(np.uint16).max:
        return counts.astype(np.uint16, copy=False)
    return counts.astype(np.uint32, copy=False)


def prepare_processed_dataset(
    *,
    hest_data_dir: str,
    output_dir: str,
    sample_id: str = "NCBI784",
    remove_control_features: bool = True,
    nucleus_only: bool = False,
    drop_zero_expression: bool = True,
    force_rebuild: bool = False,
) -> ProcessedPaths:
    processed_paths = build_processed_paths(output_dir=output_dir, sample_id=sample_id)

    expected_outputs = [
        processed_paths.counts_path,
        processed_paths.metadata_path,
        processed_paths.genes_path,
        processed_paths.manifest_path,
    ]
    if not force_rebuild and all(os.path.exists(path) for path in expected_outputs):
        return processed_paths

    transcripts_path = os.path.join(hest_data_dir, "transcripts", f"{sample_id}_transcripts.parquet")
    segmentation_path = os.path.join(hest_data_dir, "xenium_seg", f"{sample_id}_xenium_cell_seg.parquet")

    if not os.path.exists(transcripts_path):
        raise FileNotFoundError(f"Transcripts parquet not found: {transcripts_path}")
    if not os.path.exists(segmentation_path):
        raise FileNotFoundError(f"Xenium segmentation parquet not found: {segmentation_path}")

    gene_names = resolve_gene_panel(
        hest_data_dir=hest_data_dir,
        sample_id=sample_id,
        remove_control_features=remove_control_features,
    )
    metadata_df = load_xenium_segmentation_metadata(segmentation_path)
    counts = aggregate_transcripts_to_counts(
        transcripts_path=transcripts_path,
        metadata_df=metadata_df,
        gene_names=gene_names,
        nucleus_only=nucleus_only,
    )

    transcript_count = counts.sum(axis=1).astype(np.int64, copy=False)
    if drop_zero_expression:
        keep_mask = transcript_count > 0
        metadata_df = metadata_df.loc[keep_mask].reset_index(drop=True)
        counts = counts[keep_mask]
        transcript_count = transcript_count[keep_mask]

    fold_ids, fold_edges = assign_position_folds(metadata_df["centroid_x"].to_numpy(), num_folds=5)
    metadata_df["transcript_count"] = transcript_count
    metadata_df["position_fold_5"] = fold_ids.astype(np.int16)

    counts = maybe_downcast_counts(counts)
    np.save(processed_paths.counts_path, counts)
    metadata_df.to_parquet(processed_paths.metadata_path, index=False)
    save_json({"genes": list(gene_names)}, processed_paths.genes_path)
    save_json(
        {
            "sample_id": sample_id,
            "hest_data_dir": os.path.abspath(hest_data_dir),
            "transcripts_path": transcripts_path,
            "segmentation_path": segmentation_path,
            "num_cells_after_filtering": int(metadata_df.shape[0]),
            "num_genes": int(len(gene_names)),
            "remove_control_features": bool(remove_control_features),
            "nucleus_only": bool(nucleus_only),
            "drop_zero_expression": bool(drop_zero_expression),
            "counts_dtype": str(counts.dtype),
            "position_fold_edges_5": fold_edges,
            "x_range": [
                float(metadata_df["centroid_x"].min()) if not metadata_df.empty else 0.0,
                float(metadata_df["centroid_x"].max()) if not metadata_df.empty else 0.0,
            ],
        },
        processed_paths.manifest_path,
    )

    return processed_paths


class _SimpleWSI:
    def __init__(self, image_path: str) -> None:
        self._image = Image.open(image_path).convert("RGB")
        self.level_count = 1
        self.level_dimensions = [self._image.size]
        self.level_downsamples = [1.0]
        self.dimensions = self._image.size

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> Image.Image:
        x, y = location
        width, height = size
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(self._image.size[0], int(x + width))
        bottom = min(self._image.size[1], int(y + height))
        canvas = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
        if right > left and bottom > top:
            crop = self._image.crop((left, top, right, bottom))
            canvas.paste(crop, (max(0, left - int(x)), max(0, top - int(y))))
        return canvas

    def close(self) -> None:
        self._image.close()


class XeniumSingleCellDataset(Dataset):
    def __init__(
        self,
        *,
        processed_dir: str,
        hest_data_dir: str,
        sample_id: str = "NCBI784",
        split: str = "train",
        test_fold: int = 4,
        num_position_folds: int = 5,
        crop_size: int = CFG.CROP_SIZE,
        image_size: int = CFG.IMAGE_SIZE,
        wsi_level: int = 0,
        augment: bool = False,
        max_cells: int = 0,
        include_image: bool = True,
        encoder_target_gene_names: Optional[Sequence[str]] = None,
        encoder_use_raw_counts: bool = False,
    ) -> None:
        super().__init__()

        processed_paths = build_processed_paths(output_dir=processed_dir, sample_id=sample_id)
        if not os.path.exists(processed_paths.counts_path):
            raise FileNotFoundError(f"Missing processed counts file: {processed_paths.counts_path}")
        if not os.path.exists(processed_paths.metadata_path):
            raise FileNotFoundError(f"Missing processed metadata file: {processed_paths.metadata_path}")
        if not os.path.exists(processed_paths.genes_path):
            raise FileNotFoundError(f"Missing processed genes file: {processed_paths.genes_path}")

        self.sample_id = sample_id
        self.split = str(split)
        self.crop_size = int(crop_size)
        self.image_size = int(image_size)
        self.wsi_level = int(wsi_level)
        self.augment = bool(augment)
        self.include_image = bool(include_image)
        self._slide = None

        with open(processed_paths.genes_path, "r", encoding="utf-8") as handle:
            genes_payload = json.load(handle)
        self.gene_names = list(genes_payload["genes"])
        self.num_features = len(self.gene_names)
        self.encoder_target_gene_names = (
            [str(gene_name) for gene_name in encoder_target_gene_names]
            if encoder_target_gene_names is not None
            else list(self.gene_names)
        )
        self.encoder_use_raw_counts = bool(encoder_use_raw_counts)
        self.encoder_num_features = len(self.encoder_target_gene_names)
        self._encoder_target_to_source = build_target_to_source_index(self.gene_names, self.encoder_target_gene_names)

        metadata_df = pd.read_parquet(processed_paths.metadata_path)
        self.cell_ids = metadata_df["cell_id"].to_numpy(dtype=np.int64, copy=False)
        self.centroid_x = metadata_df["centroid_x"].to_numpy(dtype=np.float32, copy=False)
        self.centroid_y = metadata_df["centroid_y"].to_numpy(dtype=np.float32, copy=False)
        self.transcript_count = metadata_df["transcript_count"].to_numpy(dtype=np.int64, copy=False)
        self.counts = np.load(processed_paths.counts_path, mmap_mode="r")

        fold_ids, fold_edges = assign_position_folds(self.centroid_x, num_folds=num_position_folds)
        self.fold_ids = fold_ids.astype(np.int64)
        self.fold_edges = fold_edges

        if self.split == "train":
            selection_mask = self.fold_ids != int(test_fold)
        elif self.split == "test":
            selection_mask = self.fold_ids == int(test_fold)
        elif self.split == "all":
            selection_mask = np.ones_like(self.fold_ids, dtype=bool)
        else:
            raise ValueError("split must be one of: train, test, all")

        self.indices = np.flatnonzero(selection_mask).astype(np.int64)
        if max_cells and max_cells > 0:
            self.indices = self.indices[: int(max_cells)]

        self.wsi_path = os.path.join(hest_data_dir, "wsis", f"{sample_id}.tif")
        if not os.path.exists(self.wsi_path):
            raise FileNotFoundError(f"WSI file not found: {self.wsi_path}")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_slide"] = None
        return state

    def _get_slide(self):
        if self._slide is None:
            try:
                self._slide = openslide.OpenSlide(self.wsi_path)
            except Exception:
                self._slide = _SimpleWSI(self.wsi_path)
        return self._slide

    def _read_crop(self, center_x: float, center_y: float) -> Image.Image:
        slide = self._get_slide()
        downsample = float(slide.level_downsamples[self.wsi_level])
        read_size = max(1, int(round(self.crop_size / downsample)))
        left = int(round(float(center_x) - self.crop_size / 2.0))
        top = int(round(float(center_y) - self.crop_size / 2.0))
        region = slide.read_region((left, top), self.wsi_level, (read_size, read_size)).convert("RGB")
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
        tensor = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return tensor

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
            "cell_id": int(self.cell_ids[data_index]),
            "centroid_x": float(self.centroid_x[data_index]),
            "centroid_y": float(self.centroid_y[data_index]),
            "fold_id": int(self.fold_ids[data_index]),
        }
        if self.include_image:
            image = self._read_crop(self.centroid_x[data_index], self.centroid_y[data_index])
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


class ScRNADataset(Dataset):
    def __init__(
        self,
        *,
        h5ad_path: str,
        target_gene_names: Sequence[str],
        use_raw_counts: bool = False,
        max_cells: int = 0,
    ) -> None:
        super().__init__()
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"scRNA h5ad file not found: {h5ad_path}")

        self.h5ad_path = os.path.abspath(h5ad_path)
        self.target_gene_names = [str(gene_name) for gene_name in target_gene_names]
        self.use_raw_counts = bool(use_raw_counts)
        self._h5 = None

        with h5py.File(self.h5ad_path, "r") as handle:
            self.source_gene_names = _decode_h5ad_string_array(handle["var/_index"][:])
            self.obs_names = _decode_h5ad_string_array(handle["obs/_index"][:])
            self.num_source_features = int(handle["var/_index"].shape[0])
            self.num_features = len(self.target_gene_names)
            total_cells = int(handle["obs/_index"].shape[0])

        self.indices = np.arange(total_cells, dtype=np.int64)
        if max_cells and max_cells > 0:
            self.indices = self.indices[: int(max_cells)]
        self._target_to_source = build_target_to_source_index(self.source_gene_names, self.target_gene_names)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5ad_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data_index = int(self.indices[index])
        handle = self._get_h5()
        indptr = handle["X/indptr"]
        start = int(indptr[data_index])
        end = int(indptr[data_index + 1])
        raw_expression = np.zeros(self.num_source_features, dtype=np.float32)
        if end > start:
            row_indices = np.asarray(handle["X/indices"][start:end], dtype=np.int64)
            row_data = np.asarray(handle["X/data"][start:end], dtype=np.float32)
            raw_expression[row_indices] = row_data
        expression = raw_expression if self.use_raw_counts else np.log1p(raw_expression)
        encoder_expression = align_expression_from_index_map(expression, self._target_to_source)
        return {
            "encoder_expression": torch.from_numpy(encoder_expression),
            "sc_index": int(data_index),
            "obs_name": self.obs_names[data_index],
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()
